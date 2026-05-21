#!/usr/bin/env python3
"""
Demucs + pyannote.audio による BGM/SE 除去 + 話者分離スクリプト

長い音声ファイルから BGM/SE を除去し、「誰が・いつ・話したか」を自動特定する。
話者別の音声ファイル（無音カット済み）を高品質で出力する。

使い方:
  # BGM/SE 除去あり（デフォルト）
  uv run python scripts/diarize_speakers.py --input data/input/audio.wav

  # BGM/SE 除去をスキップ
  uv run python scripts/diarize_speakers.py --input data/input/audio.wav --no-separate

  # 話者数を指定
  uv run python scripts/diarize_speakers.py --input audio.wav --num-speakers 3

前提条件:
  - HuggingFace アカウントが必要
  - pyannote/speaker-diarization-3.1 と pyannote/segmentation-3.0 の利用条件に同意
  - HF_TOKEN 環境変数 または --hf-token でトークンを指定

出力:
  <output-dir>/speakers/SPEAKER_XX.wav    - 話者別の音声ファイル（無音カット済み）
  <output-dir>/<stem>_diarization.json    - JSON 形式の話者分離結果
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

# ---- torchaudio 2.9+ 互換パッチ ----
if not hasattr(torchaudio, "AudioMetaData"):

    @dataclass
    class AudioMetaData:
        sample_rate: int
        num_frames: int
        num_channels: int
        bits_per_sample: int
        encoding: str

    torchaudio.AudioMetaData = AudioMetaData

if not hasattr(torchaudio, "info"):

    def _torchaudio_info(filepath, backend=None):
        info = sf.info(filepath)
        return torchaudio.AudioMetaData(
            sample_rate=int(info.samplerate),
            num_frames=int(info.frames),
            num_channels=int(info.channels),
            bits_per_sample=0,
            encoding=info.subtype or "",
        )

    torchaudio.info = _torchaudio_info

if not hasattr(torchaudio, "list_audio_backends"):

    def _list_audio_backends():
        return ["soundfile"]

    torchaudio.list_audio_backends = _list_audio_backends

# ---- PyTorch 2.6+ 互換パッチ ----
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load


@contextlib.contextmanager
def _patch_htdemucs_view():
    """CUDA上でhtdemucsが生成する非連続テンソルのview()エラーを回避する。

    htdemucs内部の転置などにより非連続になったテンソルに.view()が呼ばれると
    RuntimeErrorになる。スコープ内でのみ torch.Tensor.view を
    「非連続なら .contiguous() してから view」に差し替え、抜けたら元に戻す。
    """
    _orig = torch.Tensor.view

    def _safe_view(self, *args):
        if not self.is_contiguous():
            return _orig(self.contiguous(), *args)
        return _orig(self, *args)

    torch.Tensor.view = _safe_view
    try:
        yield
    finally:
        torch.Tensor.view = _orig


# ---- デフォルト設定 ----
DEFAULT_MODEL = "pyannote/speaker-diarization-3.1"
DEFAULT_SEPARATION_MODEL = "htdemucs"


def resolve_hf_token(cli_token: str | None) -> str:
    """HuggingFace トークンを解決する（CLI → 環境変数 → ログイン済みトークン）。"""
    if cli_token:
        return cli_token

    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env_token:
        return env_token

    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        return token_path.read_text().strip()

    print(
        "ERROR: HuggingFace トークンが見つかりません。\n"
        "  以下のいずれかの方法でトークンを設定してください:\n"
        "  1. --hf-token hf_... を指定\n"
        "  2. HF_TOKEN 環境変数を設定\n"
        "  3. huggingface-cli login を実行\n\n"
        "  事前に以下のモデルページで利用条件に同意してください:\n"
        "  - https://huggingface.co/pyannote/speaker-diarization-3.1\n"
        "  - https://huggingface.co/pyannote/segmentation-3.0",
        file=sys.stderr,
    )
    sys.exit(1)


def separate_vocals(
    audio_path: Path,
    model_name: str = DEFAULT_SEPARATION_MODEL,
    device: str = "cuda",
    segment: int | None = None,
    chunk_minutes: float = 5.0,
) -> tuple[Path, int]:
    """Demucs で音声を分離し、ボーカル（台詞）のみを一時 WAV ファイルに書き出す。

    全音声をメモリに乗せず 5 分チャンク単位で処理することで、
    apply_model が確保する巨大な出力テンソル（~13 GB）によるクラッシュを防ぐ。

    Returns:
        (tmp_vocals_path, sample_rate)
    """
    import soundfile as sf
    from demucs.apply import apply_model
    from demucs.audio import convert_audio
    from demucs.pretrained import get_model

    print(f"[demucs] loading model: {model_name}")
    model = get_model(model_name)
    model.eval()

    # BagOfModels には .segment が存在しないため、サブモデルから取得する
    # HTDemucs は training_length = segment * samplerate の整数倍の入力しか受け付けない
    # apply_model に segment=None を渡すと各サブモデルが自分の .segment を使う（正しい挙動）
    # ユーザー指定値は無視して常に None（= モデルのネイティブ値）を使う
    if segment is not None:
        # サブモデルの training unit を取得して警告のみ表示
        sub = model.models[0] if hasattr(model, "models") else model
        native = float(sub.segment)
        print(f"[demucs] --separation-segment {segment}s は無視し、"
              f"モデルのネイティブ値 {native:.2f}s を使用します")
    native_str = (f"{float(model.models[0].segment):.2f}s"
                  if hasattr(model, "models") else "auto")

    # soundfile でファイル情報のみ取得（全ロードしない）
    info = sf.info(str(audio_path))
    orig_sr = info.samplerate
    total_frames = info.frames
    duration = total_frames / orig_sr
    print(f"[demucs] audio info: {orig_sr}Hz, {info.channels}ch, "
          f"{duration:.1f}s ({duration / 60:.1f}min)")

    chunk_samples = int(chunk_minutes * 60 * orig_sr)
    target_sr = model.samplerate
    target_channels = model.audio_channels
    vocals_idx = model.sources.index("vocals")

    tmp_vocals_path = audio_path.parent / f".tmp_vocals_{audio_path.stem}.wav"

    model.to(device)
    print(f"[demucs] separating on {device}, chunk={chunk_minutes:.0f}min, segment={native_str} ...")
    t0 = time.time()

    out_sf = None
    processed_frames = 0
    try:
        with sf.SoundFile(str(audio_path)) as in_sf:
            while True:
                data = in_sf.read(chunk_samples, dtype="float32", always_2d=True)
                if len(data) == 0:
                    break

                # (frames, channels) → (channels, frames) tensor
                wav = torch.from_numpy(data.T)
                wav = convert_audio(wav, orig_sr, target_sr, target_channels)
                ref = wav.mean(0)
                wav_norm = (wav - ref.mean()) / (ref.std() + 1e-8)

                with torch.no_grad(), _patch_htdemucs_view():
                    # segment=None → 各サブモデルがネイティブの segment 値を使う
                    out = apply_model(model, wav_norm[None], device=device, segment=None)

                vocals = out[0, vocals_idx] * (ref.std() + 1e-8) + ref.mean()
                vocals_np = vocals.cpu().numpy().T  # (samples, channels)

                if out_sf is None:
                    out_sf = sf.SoundFile(
                        str(tmp_vocals_path), mode="w",
                        samplerate=target_sr, channels=vocals_np.shape[1],
                        format="WAV", subtype="FLOAT",
                    )
                out_sf.write(vocals_np)

                del out, wav_norm, wav, vocals
                torch.cuda.empty_cache()

                processed_frames += len(data)
                pct = processed_frames / total_frames * 100
                elapsed = time.time() - t0
                print(f"[demucs] {pct:.1f}% ({processed_frames / orig_sr:.0f}/{duration:.0f}s, "
                      f"{elapsed:.0f}s elapsed)")
    finally:
        if out_sf is not None:
            out_sf.close()

    del model
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"[demucs] completed in {elapsed:.0f}s ({elapsed / 60:.1f}min)")
    print(f"[demucs] vocals saved to: {tmp_vocals_path}")

    return tmp_vocals_path, target_sr


def prepare_waveform(
    audio_input: Path | tuple[torch.Tensor, int],
    target_sr: int = 16000,
) -> dict:
    """音声をメモリにロードし、モノラル16kHzに変換する。

    pyannote 話者分離用の入力を準備する。
    """
    if isinstance(audio_input, tuple):
        waveform, sr = audio_input
        duration = waveform.shape[1] / sr
        print(f"[audio] using pre-loaded waveform: {sr}Hz, {waveform.shape[0]}ch, "
              f"{duration:.1f}s")
    else:
        print(f"[audio] loading: {audio_input}")
        waveform, sr = torchaudio.load(str(audio_input))
        duration = waveform.shape[1] / sr
        print(f"[audio]   {sr}Hz, {waveform.shape[0]}ch, {duration:.1f}s")

    # モノラル化
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # リサンプリング
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr

    return {"waveform": waveform, "sample_rate": sr}


def apply_silero_vad(
    waveform: torch.Tensor,
    sample_rate: int,
    device: str,
    threshold: float = 0.5,
    min_silence_ms: int = 300,
) -> torch.Tensor:
    """Silero VAD で非発話区間をゼロマスクする。

    waveform: (1, samples) @ sample_rate Hz（prepare_waveform 出力と同形式）
    Returns: マスク済みの同形状テンソル
    """
    print("[silero-vad] loading model...")
    vad_model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        verbose=False,
    )
    get_speech_timestamps = utils[0]
    vad_model = vad_model.to(device)

    audio_1d = waveform.squeeze(0).to(device)
    print("[silero-vad] detecting speech regions...")
    speech_ts = get_speech_timestamps(
        audio_1d,
        vad_model,
        sampling_rate=sample_rate,
        threshold=threshold,
        min_silence_duration_ms=min_silence_ms,
    )

    mask = torch.zeros_like(waveform)
    for ts in speech_ts:
        mask[:, ts["start"]: ts["end"]] = 1.0
    masked = (waveform * mask).cpu()

    total_speech = sum(ts["end"] - ts["start"] for ts in speech_ts)
    ratio = total_speech / waveform.shape[-1]
    print(
        f"[silero-vad] 発話区間: {ratio:.1%} "
        f"({len(speech_ts)} 区間, {total_speech / sample_rate:.0f}s / {waveform.shape[-1] / sample_rate:.0f}s)"
    )

    del vad_model
    torch.cuda.empty_cache()
    return masked


def run_whisperx_pipeline(
    audio_np: "np.ndarray",
    hf_token: str,
    device: str,
    whisper_model: str = "large-v2",
    language: str = "ja",
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    clustering_threshold: float | None = None,
) -> "tuple[pyannote.core.Annotation, list[dict]]":
    """WhisperX で ASR + 単語アライメント + 話者分離を実行する。

    audio_np: float32 numpy array, 16kHz モノラル
    Returns:
        (diarization, segments)
        diarization: pyannote Annotation（extract_speaker_audio と互換）
        segments: テキスト付きセグメントリスト（transcript JSON 用）
    """
    import numpy as np
    import whisperx
    from pyannote.core import Annotation, Segment

    compute_type = "float16" if device == "cuda" else "int8"

    print(f"[whisperx] loading ASR model: {whisper_model} ({compute_type})")
    asr_model = whisperx.load_model(
        whisper_model, device, compute_type=compute_type, language=language,
    )

    print("[whisperx] transcribing...")
    result = asr_model.transcribe(audio_np, batch_size=8)
    detected_lang = result.get("language", language)
    print(f"[whisperx] 言語: {detected_lang}, セグメント数: {len(result['segments'])}")

    del asr_model
    torch.cuda.empty_cache()

    print("[whisperx] aligning word timestamps...")
    align_model, metadata = whisperx.load_align_model(
        language_code=detected_lang, device=device,
    )
    result = whisperx.align(
        result["segments"], align_model, metadata, audio_np, device,
        return_char_alignments=False,
    )
    del align_model
    torch.cuda.empty_cache()

    print("[whisperx] diarizing...")
    from whisperx.diarize import DiarizationPipeline as WhisperxDiarizationPipeline

    diarize_kwargs: dict = {}
    if num_speakers is not None:
        diarize_kwargs["num_speakers"] = num_speakers
    if min_speakers is not None:
        diarize_kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        diarize_kwargs["max_speakers"] = max_speakers

    diarize_pipeline = WhisperxDiarizationPipeline(use_auth_token=hf_token, device=device)

    if clustering_threshold is not None:
        try:
            params = diarize_pipeline.model.parameters(instantiated=True)
            params["clustering"]["threshold"] = clustering_threshold
            diarize_pipeline.model.instantiate(params)
            print(f"[whisperx] clustering threshold: {clustering_threshold}")
        except Exception as e:
            print(f"[warn] clustering threshold 設定失敗: {e}", file=sys.stderr)

    diarize_segments = diarize_pipeline(audio_np, **diarize_kwargs)
    result_with_speakers = whisperx.assign_word_speakers(diarize_segments, result)

    # pyannote Annotation に変換
    annotation = Annotation()
    for seg in result_with_speakers["segments"]:
        speaker = seg.get("speaker", "UNKNOWN")
        annotation[Segment(seg["start"], seg["end"])] = speaker

    print(f"[whisperx] {len(annotation.labels())} 話者を検出")
    return annotation, result_with_speakers["segments"]


def write_transcript_json(
    segments: list[dict],
    output_path: Path,
    audio_path: Path,
) -> None:
    """WhisperX の文字起こし結果を JSON で保存する。"""
    data = {
        "audio_file": audio_path.name,
        "segments": [
            {
                "speaker": seg.get("speaker", "UNKNOWN"),
                "start": round(seg["start"], 3),
                "end": round(seg["end"], 3),
                "text": seg.get("text", "").strip(),
            }
            for seg in segments
        ],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[output] transcript: {output_path}")


def load_pipeline(
    model_id: str,
    hf_token: str,
    device: str,
    clustering_threshold: float | None = None,
) -> "pyannote.audio.Pipeline":
    """pyannote Pipeline をロードしてデバイスに転送する。"""
    from pyannote.audio import Pipeline

    print(f"[pyannote] loading model: {model_id}")
    pipeline = Pipeline.from_pretrained(model_id, use_auth_token=hf_token)

    if clustering_threshold is not None:
        try:
            params = pipeline.parameters(instantiated=True)
            params["clustering"]["threshold"] = clustering_threshold
            pipeline.instantiate(params)
            print(f"[pyannote] clustering threshold: {clustering_threshold} "
                  f"（低い値=より細かく区別 / 高い値=より統合）")
        except Exception as e:
            print(f"[warn] clustering threshold の設定に失敗しました: {e}", file=sys.stderr)

    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA が利用できないため CPU に切り替えます", file=sys.stderr)
        device = "cpu"

    if device == "cuda":
        free_mem = torch.cuda.mem_get_info()[0] / (1024 ** 3)
        if free_mem < 2.0:
            print(
                f"[warn] GPU 空き VRAM が {free_mem:.1f}GB です（2GB 未満）。\n"
                "  CPU にフォールバックします。GPU を使用する場合は\n"
                "  他の GPU プロセスを終了してから再実行してください。",
                file=sys.stderr,
            )
            device = "cpu"

    pipeline.to(torch.device(device))
    print(f"[pyannote] device: {device}")
    return pipeline


def run_diarization(
    pipeline: "pyannote.audio.Pipeline",
    audio_input: dict,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
) -> "pyannote.core.Annotation":
    """音声に対して話者分離を実行する。"""
    kwargs: dict = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    print("[pyannote] running diarization...")
    if kwargs:
        print(f"[pyannote]   hints: num_speakers={num_speakers}, "
              f"min={min_speakers}, max={max_speakers}")
    else:
        print("[pyannote]   speaker count: auto-detect")

    t0 = time.time()
    diarization = pipeline(audio_input, **kwargs)
    elapsed = time.time() - t0
    print(f"[pyannote] completed in {elapsed:.0f}s ({elapsed / 60:.1f}min)")

    speaker_count = len(diarization.labels())
    print(f"[pyannote] detected {speaker_count} speaker(s)")
    return diarization


def write_json(
    diarization: "pyannote.core.Annotation",
    output_path: Path,
    audio_path: Path,
    model_id: str,
) -> None:
    """JSON 形式で話者分離結果を出力する。"""
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "duration": round(turn.end - turn.start, 3),
        })

    data = {
        "audio_file": audio_path.name,
        "duration": round(diarization.get_timeline().extent().end, 3),
        "num_speakers": len(diarization.labels()),
        "model": model_id,
        "segments": segments,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[output] JSON: {output_path}")


def print_summary(diarization: "pyannote.core.Annotation") -> None:
    """話者別の発話時間サマリーをコンソールに表示する。"""
    speakers = sorted(diarization.labels())
    print(f"\n{'=' * 50}")
    print(f"  話者分離サマリー: {len(speakers)} 話者")
    print(f"{'=' * 50}")

    for speaker in speakers:
        timeline = diarization.label_timeline(speaker)
        total = sum(turn.duration for turn in timeline)
        print(f"  {speaker}: {total:.1f}s ({len(timeline)} セグメント)")

    print(f"{'=' * 50}\n")


def extract_speaker_audio(
    diarization: "pyannote.core.Annotation",
    output_dir: Path,
    min_duration: float = 0.5,
    hq_waveform: torch.Tensor | None = None,
    hq_sample_rate: int | None = None,
    lq_waveform: torch.Tensor | None = None,
    lq_sample_rate: int | None = None,
) -> None:
    """話者別フォルダを作成し、各セグメントを個別の WAV ファイルとして出力する。

    出力構造:
        <output_dir>/speakers/SPEAKER_00/0001.wav
        <output_dir>/speakers/SPEAKER_00/0002.wav
        <output_dir>/speakers/SPEAKER_01/0001.wav
        ...

    hq_waveform が指定された場合は高品質版から抽出する。
    """
    if hq_waveform is not None and hq_sample_rate is not None:
        out_waveform = hq_waveform
        out_sr = hq_sample_rate
        print(f"[extract] 高品質音声から抽出: {out_sr}Hz, {hq_waveform.shape[0]}ch")
    elif lq_waveform is not None and lq_sample_rate is not None:
        out_waveform = lq_waveform
        out_sr = lq_sample_rate
        print(f"[extract] 標準音声から抽出: {out_sr}Hz, {lq_waveform.shape[0]}ch")
    else:
        print("[extract] ERROR: 波形がありません", file=sys.stderr)
        return

    speakers_dir = output_dir / "speakers"
    speakers = sorted(diarization.labels())
    min_samples = int(min_duration * out_sr)
    total_files = 0

    print("[extract] 話者別セグメントを抽出中...")

    for speaker in speakers:
        speaker_dir = speakers_dir / speaker
        speaker_dir.mkdir(parents=True, exist_ok=True)

        timeline = diarization.label_timeline(speaker)
        idx = 0

        for turn in timeline:
            start_sample = int(turn.start * out_sr)
            end_sample = min(int(turn.end * out_sr), out_waveform.shape[-1])

            if out_waveform.dim() == 2 and out_waveform.shape[0] > 1:
                seg = out_waveform[:, start_sample:end_sample]
            else:
                seg = out_waveform[0, start_sample:end_sample].unsqueeze(0)

            if seg.shape[-1] < min_samples:
                continue

            idx += 1
            out_path = speaker_dir / f"{idx:04d}.wav"
            torchaudio.save(str(out_path), seg, out_sr)

        if idx == 0:
            speaker_dir.rmdir()
            continue

        total_dur = sum(t.duration for t in timeline if int(t.duration * out_sr) >= min_samples)
        print(f"  {speaker}: {idx} セグメント, {total_dur:.1f}s → {speaker_dir}/")
        total_files += idx

    print(f"[extract] {len(speakers)} 話者 / {total_files} ファイルを {speakers_dir} に出力しました")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demucs + pyannote.audio による BGM/SE 除去 + 話者分離",
    )
    parser.add_argument("--input", type=Path, required=True, help="入力音声ファイル（WAV/MP3）")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="出力ディレクトリ（デフォルト: data/output/diarization/<stem>）",
    )
    parser.add_argument("--hf-token", type=str, default=None, help="HuggingFace トークン")
    parser.add_argument("--num-speakers", type=int, default=None, help="話者数（省略時は自動推定）")
    parser.add_argument("--min-speakers", type=int, default=None, help="最小話者数（省略時は自動推定）")
    parser.add_argument("--max-speakers", type=int, default=None, help="最大話者数（省略時は自動推定）")
    parser.add_argument("--device", type=str, default="cuda", help="推論デバイス (cuda/cpu)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="pyannote モデル ID")
    parser.add_argument(
        "--clustering-threshold", type=float, default=None,
        help="話者クラスタリング閾値（低い値=より細かく区別/過統合防止、高い値=より統合/過分割防止）",
    )

    # WhisperX オプション
    parser.add_argument(
        "--use-whisperx", action="store_true",
        help="WhisperX（ASR + 単語アライメント + 話者分離）を使用する",
    )
    parser.add_argument(
        "--whisper-model", type=str, default="large-v2",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="WhisperX のモデルサイズ（デフォルト: large-v2）",
    )
    parser.add_argument(
        "--language", type=str, default="ja",
        help="音声の言語コード（デフォルト: ja）",
    )

    # Silero VAD オプション
    parser.add_argument(
        "--silero-vad", action="store_true",
        help="Silero VAD で非発話区間をマスクしてから話者分離する（デフォルト無効）",
    )

    # 音源分離オプション
    parser.add_argument(
        "--no-separate", action="store_true",
        help="BGM/SE 分離をスキップする（入力音声をそのまま話者分離に使用）",
    )
    parser.add_argument(
        "--separation-model", type=str, default=DEFAULT_SEPARATION_MODEL,
        choices=["htdemucs", "htdemucs_ft", "htdemucs_6s"],
        help="Demucs モデル（デフォルト: htdemucs）",
    )
    parser.add_argument(
        "--separation-segment", type=int, default=None,
        help="Demucs セグメント長（秒）-- GPU メモリ不足の場合は小さい値に設定",
    )
    args = parser.parse_args()

    # 入力チェック
    if not args.input.exists():
        print(f"ERROR: {args.input} が見つかりません", file=sys.stderr)
        sys.exit(1)

    # 出力ディレクトリ
    stem = args.input.stem
    output_dir = args.output_dir or Path(f"data/output/diarization/{stem}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # HF トークン解決
    hf_token = resolve_hf_token(args.hf_token)

    # ---- Step 1: 音源分離 (BGM/SE 除去) ----
    hq_vocals_path: Path | None = None
    hq_sr: int | None = None

    if not args.no_separate:
        hq_vocals_path, hq_sr = separate_vocals(
            args.input,
            model_name=args.separation_model,
            device=args.device,
            segment=args.separation_segment,
        )
        audio_input = prepare_waveform(hq_vocals_path)
    else:
        print("[demucs] BGM/SE 分離をスキップ (--no-separate)")
        audio_input = prepare_waveform(args.input)

    # ---- Step 1.5: Silero VAD 前処理（opt-in） ----
    if args.silero_vad:
        audio_input["waveform"] = apply_silero_vad(
            audio_input["waveform"], audio_input["sample_rate"], args.device,
        )

    # ---- Step 2: 話者分離 ----
    if args.use_whisperx:
        print(
            "[whisperx] 注意: WhisperX は ASR ベースのセグメント分割のため、"
            "セグメントが 10〜30 秒程度の長さになります。\n"
            "           TTS 学習データ用途には pyannote（デフォルト）の方が適しています。"
        )
        import numpy as np
        audio_np = audio_input["waveform"].squeeze(0).numpy().astype("float32")
        diarization, wx_segments = run_whisperx_pipeline(
            audio_np, hf_token, args.device,
            whisper_model=args.whisper_model,
            language=args.language,
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            clustering_threshold=args.clustering_threshold,
        )
        # 文字起こし結果を保存
        transcript_path = output_dir / f"{stem}_transcript.json"
        write_transcript_json(wx_segments, transcript_path, args.input)
    else:
        pipeline = load_pipeline(args.model, hf_token, args.device, args.clustering_threshold)
        diarization = run_diarization(
            pipeline, audio_input,
            args.num_speakers, args.min_speakers, args.max_speakers,
        )

    # ---- Step 3: 話者別音声抽出 ----
    # pyannote 完了後に HQ 音声をロード（demucs との同時メモリ使用を避ける）
    hq_waveform: torch.Tensor | None = None
    if hq_vocals_path is not None:
        print(f"[extract] HQ 音声をロード中: {hq_vocals_path}")
        hq_waveform, hq_sr = torchaudio.load(str(hq_vocals_path))

    try:
        extract_speaker_audio(
            diarization,
            output_dir=output_dir,
            hq_waveform=hq_waveform,
            hq_sample_rate=hq_sr,
            lq_waveform=audio_input["waveform"],
            lq_sample_rate=audio_input["sample_rate"],
        )
    finally:
        # 一時ファイルを削除
        if hq_vocals_path is not None and hq_vocals_path.exists():
            hq_vocals_path.unlink()
            print(f"[cleanup] 一時ファイルを削除: {hq_vocals_path}")

    # ---- Step 4: JSON 出力 + サマリー ----
    json_path = output_dir / f"{stem}_diarization.json"
    write_json(diarization, json_path, args.input, args.model)
    print_summary(diarization)


if __name__ == "__main__":
    main()
