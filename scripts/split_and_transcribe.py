#!/usr/bin/env python3
"""
stable-ts による文境界ベース音声分割スクリプト（v5）

v4 からの改善点:
  - POST_ROLL を 0.50s → 1.20s に拡大（文末下降イントネーション・気息音を確実にキャプチャ）
  - areverse+silenceremove+areverse パターンで末尾実無音を自動トリム
    → 過剰な無音パディングなく、自然な終端になる
  - 末尾トリム閾値 TAIL_SILENCE_DB=-45dB を先頭トリムと分離して管理

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
from pathlib import Path

import stable_whisper

# ---- デフォルト設定 ----
DEFAULT_MP3    = Path("data/input/mamimi/tanakamamimi_clipped_full.mp3")
DEFAULT_OUTDIR = Path("data/mamimi_v5/wavs")
MODEL_SIZE     = "large-v3"

# 分割・結合パラメータ
MERGE_GAP  = 0.15   # 秒: これ未満のギャップは同一発話（息継ぎ）として結合
SPLIT_GAP  = 0.5    # 秒: これ超えのギャップは発話の切れ目として分割
MIN_DUR    = 2.0    # 秒: これ未満のセグメントは隣と結合
MAX_DUR    = 12.0   # 秒: これ超えは強制分割

# 句読点リスト（日本語・英語）
SENTENCE_END_PUNCT = [
    '。', '？', '！', '?', '!',
    ('…', ''), ('♪', ''),
]

# 抽出設定
PRE_ROLL         = 0.05   # 秒: 最初の単語の前の余白
POST_ROLL        = 1.20   # 秒: 最後の単語の後の余白（v5: 文末イントネーション・気息音対策）
SILENCE_DB       = -38    # dB: 先頭ノイズトリム閾値
TAIL_SILENCE_DB  = -45    # dB: 末尾ノイズトリム閾値（より保守的）
TAIL_SILENCE_DUR = 0.15   # 秒: 末尾トリム判定の最小無音継続時間
MIN_CLIP_DUR     = 1.5    # 秒: 抽出後の実尺がこれ未満なら除外
MIN_TEXT     = 3      # 文字数: テキストがこれ未満なら除外


def transcribe_and_segment(mp3_path: Path) -> stable_whisper.WhisperResult:
    """stable-ts で転写し、文境界ベースで再セグメント化する。"""
    print(f"[stable-ts] loading model: {MODEL_SIZE}")
    model = stable_whisper.load_model(MODEL_SIZE, device="cuda")

    print(f"[stable-ts] transcribing: {mp3_path}")
    result = model.transcribe(
        str(mp3_path),
        language="ja",
        word_timestamps=True,
        verbose=False,
    )
    raw_count = len(result.segments)
    print(f"[stable-ts] raw segments: {raw_count}")

    # ① 150ms 以下のギャップは同一発話として結合（発話内の息継ぎ対策）
    result.merge_by_gap(MERGE_GAP)
    print(f"[regroup①] after merge_by_gap({MERGE_GAP}s): {len(result.segments)} segs")

    # ② 句読点（。！？）で分割 — 文が確実に終わった箇所
    result.split_by_punctuation(SENTENCE_END_PUNCT)
    print(f"[regroup②] after split_by_punctuation: {len(result.segments)} segs")

    # ③ 500ms 超えのギャップで分割 — 句読点がない発話間隔
    result.split_by_gap(SPLIT_GAP)
    print(f"[regroup③] after split_by_gap({SPLIT_GAP}s): {len(result.segments)} segs")

    # ④ MAX_DUR を超えるセグメントを強制分割
    result.split_by_duration(MAX_DUR)
    print(f"[regroup④] after split_by_duration({MAX_DUR}s): {len(result.segments)} segs")

    return result


def get_bounds(seg: stable_whisper.result.Segment) -> tuple[float, float]:
    """単語レベルのタイムスタンプから正確な開始・終了を返す。"""
    if seg.has_words:
        words = seg.words
        start = max(0.0, words[0].start - PRE_ROLL)
        end   = words[-1].end + POST_ROLL
    else:
        start = seg.start
        end   = seg.end
    return start, end


def extract_wav(mp3_path: Path, start: float, end: float, out_path: Path) -> bool:
    """ffmpeg で区間を切り出し、冒頭ノイズを silenceremove でトリムする。"""
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-i", str(mp3_path),
        "-ac", "1",
        "-ar", "44100",
        "-af", (
            # ① 先頭無音トリム
            f"silenceremove="
            f"start_periods=1:"
            f"start_threshold={SILENCE_DB}dB:"
            f"start_duration=0.02:"
            f"detection=rms,"
            # ② 反転 → 末尾が先頭になる
            f"areverse,"
            # ③ 末尾無音トリム（元の末尾を削る）
            f"silenceremove="
            f"start_periods=1:"
            f"start_threshold={TAIL_SILENCE_DB}dB:"
            f"start_duration={TAIL_SILENCE_DUR}:"
            f"detection=rms,"
            # ④ 再反転（元の向きに戻す）
            f"areverse"
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
    parser = argparse.ArgumentParser(description="音声分割 + 文字起こし（stable-ts v3）")
    parser.add_argument("--input-mp3",  type=Path, default=DEFAULT_MP3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    mp3_path = args.input_mp3
    wavs_dir = args.output_dir

    if not mp3_path.exists():
        print(f"ERROR: {mp3_path} が見つかりません", file=sys.stderr)
        sys.exit(1)

    wavs_dir.mkdir(parents=True, exist_ok=True)
    print(f"[config] input      : {mp3_path}")
    print(f"[config] output     : {wavs_dir}")
    print(f"[config] merge_gap  : {MERGE_GAP}s  split_gap: {SPLIT_GAP}s")
    print(f"[config] min_dur    : {MIN_DUR}s  max_dur: {MAX_DUR}s")

    # 転写 + セグメント化
    result = transcribe_and_segment(mp3_path)

    # WAV 切り出し + metadata.csv 書き出し
    csv_path = wavs_dir / "metadata.csv"
    written  = 0
    skipped  = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "transcription"])
        writer.writeheader()

        for i, seg in enumerate(result.segments):
            text = seg.text.strip()
            if len(text) < MIN_TEXT:
                skipped += 1
                continue

            fname    = f"seg_{i:05d}.wav"
            out_path = wavs_dir / fname

            start, end = get_bounds(seg)

            if not extract_wav(mp3_path, start, end, out_path):
                print(f"  [warn] ffmpeg failed: {fname}", file=sys.stderr)
                skipped += 1
                continue

            actual_dur = get_wav_duration(out_path)
            if actual_dur < MIN_CLIP_DUR:
                out_path.unlink(missing_ok=True)
                skipped += 1
                continue

            writer.writerow({"file_name": fname, "transcription": text})
            written += 1

            if written % 50 == 0:
                print(f"  [{written}] {fname} ({actual_dur:.1f}s): {text[:50]}")

    print(f"\n[done] written={written}, skipped={skipped}")
    print(f"[done] metadata.csv -> {csv_path}")


if __name__ == "__main__":
    main()
