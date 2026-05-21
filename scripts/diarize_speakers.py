#!/usr/bin/env python3
"""
pyannote.audio による話者分離（Speaker Diarization）スクリプト

長い音声ファイルから「誰が・いつ・話したか」を自動特定する。
RTTM 形式と JSON 形式の両方で結果を出力する。

使い方:
  uv run python scripts/diarize_speakers.py --input data/input/audio.mp3
  uv run python scripts/diarize_speakers.py --input audio.wav --max-speakers 3 --device cpu

  話者数は自動推定されます。--num-speakers, --min-speakers, --max-speakers は
  任意のヒントとして機能します（指定しなくても動作します）。

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
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

# ---- torchaudio 2.9+ 互換パッチ ----
# pyannote.audio 3.x は torchaudio < 2.9 に依存しているが、
# 本プロジェクトは torchaudio 2.10+ を使用するため必要な API を復元する。

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
# pyannote のモデルチェックポイントは PyTorch 2.6+ の weights_only=True デフォルトと
# 互換性がないため、torch.load のデフォルトを weights_only=False にする。
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load


# ---- デフォルト設定 ----
DEFAULT_MODEL = "pyannote/speaker-diarization-3.1"


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


def load_pipeline(model_id: str, hf_token: str, device: str) -> "pyannote.audio.Pipeline":
    """pyannote Pipeline をロードしてデバイスに転送する。"""
    from pyannote.audio import Pipeline

    print(f"[pyannote] loading model: {model_id}")
    pipeline = Pipeline.from_pretrained(model_id, use_auth_token=hf_token)

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


def prepare_waveform(audio_path: Path, target_sr: int = 16000) -> dict:
    """音声をメモリにロードし、モノラル16kHzに変換する。

    pyannote にファイルパスではなく波形 dict を渡すことで、
    毎回ファイル全体を再ロードするオーバーヘッドを回避する。
    """
    print(f"[audio] loading: {audio_path}")
    waveform, sr = torchaudio.load(str(audio_path))
    duration = waveform.shape[1] / sr
    print(f"[audio]   {sr}Hz, {waveform.shape[0]}ch, {duration:.1f}s ({duration / 60:.1f}min)")

    # モノラル化
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
        print(f"[audio]   converted to mono")

    # リサンプリング
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        print(f"[audio]   resampled {sr}Hz -> {target_sr}Hz")
        sr = target_sr

    return {"waveform": waveform, "sample_rate": sr}


def run_diarization(
    pipeline: "pyannote.audio.Pipeline",
    audio_input: Path | dict,
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

    print(f"[pyannote] running diarization...")
    if kwargs:
        print(f"[pyannote]   hints: num_speakers={num_speakers}, "
              f"min={min_speakers}, max={max_speakers}")
    else:
        print("[pyannote]   speaker count: auto-detect")

    import time
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
    waveform: torch.Tensor,
    sample_rate: int,
    output_dir: Path,
    min_duration: float = 0.5,
) -> None:
    """話者別に音声を抽出し、無音部分をカットして WAV で出力する。"""
    speakers_dir = output_dir / "speakers"
    speakers_dir.mkdir(parents=True, exist_ok=True)

    speakers = sorted(diarization.labels())
    print(f"[extract] 話者別音声を抽出中...")

    for speaker in speakers:
        timeline = diarization.label_timeline(speaker)
        chunks = []

        for turn in timeline:
            start_sample = int(turn.start * sample_rate)
            end_sample = int(turn.end * sample_rate)
            end_sample = min(end_sample, waveform.shape[1])
            segment = waveform[0, start_sample:end_sample]

            if segment.shape[0] < int(min_duration * sample_rate):
                continue

            chunks.append(segment)

        if not chunks:
            continue

        # 全セグメントを結合（無音なし）
        combined = torch.cat(chunks, dim=0).unsqueeze(0)
        duration = combined.shape[1] / sample_rate

        out_path = speakers_dir / f"{speaker}.wav"
        torchaudio.save(str(out_path), combined, sample_rate)
        print(f"  {speaker}: {duration:.1f}s -> {out_path.name}")

    print(f"[extract] {len(speakers)} 話者の音声ファイルを {speakers_dir} に出力しました")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="pyannote.audio による話者分離（Speaker Diarization）",
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
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="モデル ID")
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

    # パイプラインロード
    pipeline = load_pipeline(args.model, hf_token, args.device)

    # 音声をメモリにロード（モノラル16kHz変換済み）
    audio_input = prepare_waveform(args.input)

    # 話者分離実行
    diarization = run_diarization(
        pipeline, audio_input,
        args.num_speakers, args.min_speakers, args.max_speakers,
    )

    # 話者別音声抽出
    extract_speaker_audio(
        diarization,
        waveform=audio_input["waveform"],
        sample_rate=audio_input["sample_rate"],
        output_dir=output_dir,
    )

    # JSON 出力
    json_path = output_dir / f"{stem}_diarization.json"
    write_json(diarization, json_path, args.input, args.model)

    # サマリー表示
    print_summary(diarization)


if __name__ == "__main__":
    main()
