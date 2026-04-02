#!/usr/bin/env python3
"""
Whisper のセグメントタイムスタンプを使って音声を自然な文単位に分割し、
同時に文字起こしを行い metadata.csv を生成するスクリプト。

出力:
  data/mamimi/wavs/seg_XXXXX.wav  - 分割済み WAV
  data/mamimi/wavs/metadata.csv   - (file_name, transcription)
"""

from __future__ import annotations

import csv
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from faster_whisper import WhisperModel

# ---- 設定 ----
MP3_PATH   = Path("data/input/mamimi/tanakamamimi_clipped_full.mp3")
WAVS_DIR   = Path("data/mamimi/wavs")
MODEL_SIZE = "large-v3"          # faster-whisper 1.0.3 対応モデル
MIN_DUR    = 2.0                # 秒: これ未満は次のセグメントと結合
MAX_DUR    = 15.0               # 秒: これ超えは単語境界で再分割
MIN_TEXT   = 3                  # 文字数: これ未満は除外


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
        vad_parameters={"min_silence_duration_ms": 300},
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
            buf.text  = buf.text + ("" if buf.text.endswith("。") else "") + seg.text
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

        # 単語リストを max_dur に収まるよう分割
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


def extract_wav(mp3_path: Path, start: float, end: float, out_path: Path) -> bool:
    """ffmpeg で指定区間を WAV として切り出す。"""
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-i", str(mp3_path),
        "-ac", "1",
        "-ar", "44100",
        str(out_path),
    ]
    r = subprocess.run(cmd, capture_output=True)
    return r.returncode == 0


def main() -> None:
    if not MP3_PATH.exists():
        print(f"ERROR: {MP3_PATH} が見つかりません", file=sys.stderr)
        sys.exit(1)

    WAVS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 転写
    segments = transcribe(MP3_PATH, MODEL_SIZE)

    # 2. 短いセグメントを結合
    segments = merge_short(segments, MIN_DUR)
    print(f"[merge]  after merge_short: {len(segments)} segments")

    # 3. 長すぎるセグメントを再分割
    segments = split_long(segments, MAX_DUR)
    print(f"[split]  after split_long: {len(segments)} segments")

    # 4. WAV 切り出し + metadata.csv 書き出し
    csv_path = WAVS_DIR / "metadata.csv"
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
            out_path = WAVS_DIR / fname

            if not extract_wav(MP3_PATH, seg.start, seg.end, out_path):
                print(f"  [warn] ffmpeg failed: {fname}", file=sys.stderr)
                skipped += 1
                continue

            writer.writerow({"file_name": fname, "transcription": seg.text})
            written += 1

            if written % 50 == 0:
                print(f"  [{written}] {fname}: {seg.text[:50]}")

    print(f"\n[done] written={written}, skipped={skipped}")
    print(f"[done] metadata.csv -> {csv_path}")


if __name__ == "__main__":
    main()
