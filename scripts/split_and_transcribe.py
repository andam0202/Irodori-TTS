#!/usr/bin/env python3
"""
Whisper のセグメントタイムスタンプを使って音声を自然な文単位に分割し、
同時に文字起こしを行い metadata.csv を生成するスクリプト。

改善点（v2）:
  - 単語レベルのタイムスタンプを境界に使用（より正確）
  - ffmpeg silenceremove で冒頭ノイズ・無音をトリム
  - 抽出後の実尺チェックで短すぎるクリップを除外
  - 出力先を --output-dir で変更可能

出力:
  <output-dir>/seg_XXXXX.wav  - 分割済み WAV
  <output-dir>/metadata.csv   - (file_name, transcription)
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path

from faster_whisper import WhisperModel

# ---- デフォルト設定 ----
DEFAULT_MP3    = Path("data/input/mamimi/tanakamamimi_clipped_full.mp3")
DEFAULT_OUTDIR = Path("data/mamimi_v2/wavs")
MODEL_SIZE     = "large-v3"

MIN_DUR        = 2.0    # 秒: これ未満のセグメントは結合
MAX_DUR        = 15.0   # 秒: これ超えは単語境界で再分割
MIN_TEXT       = 3      # 文字数: これ未満は除外
MIN_CLIP_DUR   = 1.5    # 秒: silenceremove 後の実尺がこれ未満なら除外
PRE_ROLL       = 0.05   # 秒: 最初の単語の前に少しだけ余白を取る
POST_ROLL      = 0.10   # 秒: 最後の単語の後に余白を取る
SILENCE_DB     = -38    # dB: これ以下を冒頭の無音として除去する閾値


@dataclass
class Segment:
    start: float
    end:   float
    text:  str
    words: list = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


def transcribe(mp3_path: Path, model_size: str) -> list[Segment]:
    print(f"[whisper] loading model: {model_size}")
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    print(f"[whisper] transcribing: {mp3_path}")
    raw_segments, info = model.transcribe(
        str(mp3_path),
        language="ja",
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 500,   # 前回300→500msに延長し境界を明確化
            "speech_pad_ms": 30,              # 発話検出パディング
        },
    )
    print(f"[whisper] detected language: {info.language} (prob={info.language_probability:.2f})")

    segments = []
    for s in raw_segments:
        words = [(w.start, w.end, w.word) for w in (s.words or [])]
        segments.append(Segment(start=s.start, end=s.end, text=s.text.strip(), words=words))
        print(f"  [{s.start:.1f}→{s.end:.1f}] {s.text.strip()[:60]}")

    print(f"[whisper] raw segments: {len(segments)}")
    return segments


def merge_short(segments: list[Segment], min_dur: float) -> list[Segment]:
    """MIN_DUR 未満のセグメントを次のセグメントと結合する。"""
    merged: list[Segment] = []
    buf: Segment | None = None

    for seg in segments:
        if buf is None:
            buf = Segment(seg.start, seg.end, seg.text, list(seg.words))
        else:
            buf.end   = seg.end
            buf.text  = buf.text + seg.text
            buf.words = buf.words + seg.words

        if buf.duration >= min_dur:
            merged.append(buf)
            buf = None

    if buf is not None and buf.duration > 0:
        if merged:
            last = merged[-1]
            last.end   = buf.end
            last.text  = last.text + buf.text
            last.words = last.words + buf.words
        else:
            merged.append(buf)

    return merged


def split_long(segments: list[Segment], max_dur: float) -> list[Segment]:
    """MAX_DUR 超えのセグメントを単語境界で再分割する。"""
    result: list[Segment] = []

    for seg in segments:
        if seg.duration <= max_dur or not seg.words:
            result.append(seg)
            continue

        chunk_start = seg.start
        chunk_words: list = []
        chunk_text  = ""

        for (ws, we, wt) in seg.words:
            chunk_words.append((ws, we, wt))
            chunk_text += wt
            if (we - chunk_start) >= max_dur:
                result.append(Segment(chunk_start, we, chunk_text.strip(), list(chunk_words)))
                chunk_start = we
                chunk_words = []
                chunk_text  = ""

        if chunk_words:
            result.append(Segment(chunk_start, chunk_words[-1][1], chunk_text.strip(), chunk_words))

    return result


def get_precise_bounds(seg: Segment) -> tuple[float, float]:
    """
    単語レベルのタイムスタンプから正確な開始・終了時刻を計算する。
    単語情報がある場合は seg.start/end よりも精度が高い。
    """
    if seg.words:
        word_start = seg.words[0][0]
        word_end   = seg.words[-1][1]
        # 最初の単語の少し前から取ることで、自然な発話開始を保持
        start = max(0.0, word_start - PRE_ROLL)
        end   = word_end + POST_ROLL
    else:
        start = seg.start
        end   = seg.end
    return start, end


def extract_wav(
    mp3_path: Path,
    start: float,
    end: float,
    out_path: Path,
) -> bool:
    """
    ffmpeg で区間を切り出し、silenceremove で冒頭ノイズをトリムする。
    silenceremove により実際の発話開始から記録される。
    """
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-i", str(mp3_path),
        "-ac", "1",
        "-ar", "44100",
        "-af", (
            f"silenceremove="
            f"start_periods=1:"
            f"start_threshold={SILENCE_DB}dB:"
            f"start_duration=0.02:"
            f"detection=rms"
        ),
        str(out_path),
    ]
    r = subprocess.run(cmd, capture_output=True)
    return r.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0


def get_wav_duration(path: Path) -> float:
    """WAV ファイルの実尺（秒）を返す。"""
    try:
        with wave.open(str(path)) as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="音声分割 + 文字起こし（改良版）")
    parser.add_argument("--input-mp3",  type=Path, default=DEFAULT_MP3,
                        help="入力 MP3 ファイル")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTDIR,
                        help="WAV と metadata.csv の出力先ディレクトリ")
    args = parser.parse_args()

    mp3_path = args.input_mp3
    wavs_dir = args.output_dir

    if not mp3_path.exists():
        print(f"ERROR: {mp3_path} が見つかりません", file=sys.stderr)
        sys.exit(1)

    wavs_dir.mkdir(parents=True, exist_ok=True)
    print(f"[config] input  : {mp3_path}")
    print(f"[config] output : {wavs_dir}")
    print(f"[config] silence trim threshold: {SILENCE_DB} dB")

    # 1. 転写
    segments = transcribe(mp3_path, MODEL_SIZE)

    # 2. 短いセグメントを結合
    segments = merge_short(segments, MIN_DUR)
    print(f"[merge]  after merge_short: {len(segments)} segments")

    # 3. 長すぎるセグメントを再分割
    segments = split_long(segments, MAX_DUR)
    print(f"[split]  after split_long: {len(segments)} segments")

    # 4. WAV 切り出し + metadata.csv 書き出し
    csv_path = wavs_dir / "metadata.csv"
    skipped  = 0
    written  = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "transcription"])
        writer.writeheader()

        for i, seg in enumerate(segments):
            if len(seg.text) < MIN_TEXT:
                skipped += 1
                continue

            fname    = f"seg_{i:05d}.wav"
            out_path = wavs_dir / fname

            # 単語タイムスタンプを使って正確な境界を取得
            start, end = get_precise_bounds(seg)

            if not extract_wav(mp3_path, start, end, out_path):
                print(f"  [warn] ffmpeg failed: {fname}", file=sys.stderr)
                skipped += 1
                continue

            # silenceremove 後の実尺チェック
            actual_dur = get_wav_duration(out_path)
            if actual_dur < MIN_CLIP_DUR:
                out_path.unlink(missing_ok=True)
                skipped += 1
                continue

            writer.writerow({"file_name": fname, "transcription": seg.text})
            written += 1

            if written % 50 == 0:
                print(f"  [{written}] {fname} ({actual_dur:.1f}s): {seg.text[:50]}")

    print(f"\n[done] written={written}, skipped={skipped}")
    print(f"[done] metadata.csv -> {csv_path}")


if __name__ == "__main__":
    main()
