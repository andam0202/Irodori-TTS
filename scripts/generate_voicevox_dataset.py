#!/usr/bin/env python3
"""
VOICEVOX HTTP API を使い、コーパステキストから学習用データセットを一括合成するスクリプト。

入力:
  --corpus <text>   1行1文のテキストファイル

出力:
  <output-dir>/seg_XXXXX.wav   合成済み WAV（44.1kHz mono）
  <output-dir>/metadata.csv    file_name,transcription（scripts/encode_latents.py 互換）

VOICEVOX エンジン（デフォルト http://127.0.0.1:50021）が起動していることが前提。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlencode

RETRY_COUNT = 3
RETRY_WAIT_SEC = 2.0
MIN_VALID_SIZE_BYTES = 1000
SAMPLE_RATE = 44100
PRE_PHONEME_LENGTH = 0.15
POST_PHONEME_LENGTH = 0.3


def http_post(url: str, data: bytes | None, content_type: str | None = None) -> bytes:
    headers = {}
    if content_type:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def with_retry(func, *args, **kwargs):
    last_err: Exception | None = None
    for attempt in range(1, RETRY_COUNT + 1):
        try:
            return func(*args, **kwargs)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last_err = e
            print(f"  [warn] attempt {attempt}/{RETRY_COUNT} failed: {e}", file=sys.stderr)
            if attempt < RETRY_COUNT:
                time.sleep(RETRY_WAIT_SEC)
    assert last_err is not None
    raise last_err


def synthesize(host: str, style_id: int, text: str, speed_scale: float) -> bytes:
    query_url = f"{host}/audio_query?" + urlencode({"speaker": style_id, "text": text})
    query_raw = with_retry(http_post, query_url, b"")
    query = json.loads(query_raw)

    query["outputSamplingRate"] = SAMPLE_RATE
    query["outputStereo"] = False
    query["prePhonemeLength"] = PRE_PHONEME_LENGTH
    query["postPhonemeLength"] = POST_PHONEME_LENGTH
    query["speedScale"] = speed_scale

    synth_url = f"{host}/synthesis?" + urlencode({"speaker": style_id})
    wav_bytes = with_retry(
        http_post, synth_url, json.dumps(query).encode("utf-8"), "application/json"
    )
    return wav_bytes


def load_corpus(corpus_path: Path) -> list[str]:
    with open(corpus_path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def write_metadata(output_dir: Path, sentences: list[str], start_index: int) -> None:
    metadata_path = output_dir / "metadata.csv"
    with open(metadata_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "transcription"])
        writer.writeheader()
        for i, text in enumerate(sentences):
            seg_index = start_index + i
            file_name = f"seg_{seg_index:05d}.wav"
            wav_path = output_dir / file_name
            if not wav_path.exists():
                # 合成に失敗し WAV が存在しないセグメントは manifest に含めない
                continue
            writer.writerow({"file_name": file_name, "transcription": text})
    print(f"[metadata] -> {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="VOICEVOX HTTP API で学習用データセットを合成")
    parser.add_argument(
        "--host", type=str, default="http://127.0.0.1:50021", help="VOICEVOX エンジンのホスト"
    )
    parser.add_argument("--style-id", type=int, required=True, help="VOICEVOX のスタイル(話者) ID")
    parser.add_argument(
        "--corpus", type=Path, required=True, help="1行1文のコーパステキストファイル"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="出力先ディレクトリ（data/<speaker>/wavs 想定）",
    )
    parser.add_argument(
        "--start-index", type=int, default=0, help="seg 番号の開始値（デフォルト0）"
    )
    parser.add_argument("--limit", type=int, default=0, help="合成する文の上限（デフォルト0=全件）")
    parser.add_argument(
        "--speed-scale", type=float, default=1.0, help="話速スケール（デフォルト1.0）"
    )
    args = parser.parse_args()

    sentences = load_corpus(args.corpus)
    if args.limit > 0:
        sentences = sentences[: args.limit]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[config] host        : {args.host}")
    print(f"[config] style_id    : {args.style_id}")
    print(f"[config] corpus      : {args.corpus} ({len(sentences)} sentences)")
    print(f"[config] output_dir  : {output_dir}")
    print(f"[config] start_index : {args.start_index}")
    print(f"[config] speed_scale : {args.speed_scale}")

    synthesized = 0
    skipped_existing = 0
    failed = 0

    for i, text in enumerate(sentences):
        seg_index = args.start_index + i
        file_name = f"seg_{seg_index:05d}.wav"
        wav_path = output_dir / file_name

        if wav_path.exists() and wav_path.stat().st_size > MIN_VALID_SIZE_BYTES:
            skipped_existing += 1
            continue

        try:
            wav_bytes = synthesize(args.host, args.style_id, text, args.speed_scale)
        except Exception as e:
            print(f"  [error] seg_{seg_index:05d} synthesis failed: {e}", file=sys.stderr)
            failed += 1
            continue

        with open(wav_path, "wb") as f:
            f.write(wav_bytes)
        synthesized += 1

        if (i + 1) % 10 == 0:
            print(
                f"[progress] {i + 1}/{len(sentences)} (synthesized={synthesized}, skipped={skipped_existing}, failed={failed})"
            )

    print(
        f"\n[done] synthesized={synthesized}, skipped_existing={skipped_existing}, "
        f"failed={failed}, total={len(sentences)}"
    )

    write_metadata(output_dir, sentences, args.start_index)


if __name__ == "__main__":
    main()
