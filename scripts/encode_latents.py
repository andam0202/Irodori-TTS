#!/usr/bin/env python3
"""
metadata.csv の WAV ファイルを DACVAE でエンコードし、
学習用マニフェスト（JSONL）を生成するスクリプト。

prepare_manifest.py は audiofolder の data_dir 形式に非対応なため、
DACVAECodec を直接呼び出してエンコードする。

入力:
  <wavs-dir>/metadata.csv
  <wavs-dir>/seg_*.wav

出力:
  <latent-dir>/seg_XXXXX.pt  (DACVAE エンコード済みレイテント)
  <manifest-path>            (学習マニフェスト JSONL)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from irodori_tts.codec import DACVAECodec

MAX_FRAMES  = 750   # train_mamimi_lora.yaml の max_latent_steps
SPEAKER_ID  = "mamimi"
DEVICE      = "cuda"


def main() -> None:
    parser = argparse.ArgumentParser(description="DACVAE レイテント変換 + マニフェスト生成")
    parser.add_argument(
        "--wavs-dir",
        type=Path,
        default=Path("data/mamimi/wavs"),
        help="WAV ファイルと metadata.csv があるディレクトリ",
    )
    parser.add_argument(
        "--latent-dir",
        type=Path,
        default=None,
        help="レイテント保存先（省略時は wavs-dir の親ディレクトリ / latents）",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="マニフェスト JSONL の出力パス（省略時は wavs-dir の親ディレクトリ / manifest.jsonl）",
    )
    parser.add_argument(
        "--speaker-id",
        type=str,
        default=SPEAKER_ID,
        help="マニフェストに記録する speaker_id",
    )
    args = parser.parse_args()

    wavs_dir = args.wavs_dir
    latent_dir = args.latent_dir or wavs_dir.parent / "latents"
    manifest_path = args.manifest_path or wavs_dir.parent / "manifest.jsonl"

    csv_path = wavs_dir / "metadata.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} が見つかりません", file=sys.stderr)
        sys.exit(1)

    latent_dir.mkdir(parents=True, exist_ok=True)

    print(f"[config] wavs     : {wavs_dir}")
    print(f"[config] latents  : {latent_dir}")
    print(f"[config] manifest : {manifest_path}")

    print("[codec] loading DACVAE...")
    codec = DACVAECodec.load(device=DEVICE, normalize_db=-16.0)
    print("[codec] ready.")

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"[encode] {len(rows)} samples -> {latent_dir}")

    written = 0
    skipped = 0

    with open(manifest_path, "w", encoding="utf-8") as mf:
        for row in tqdm(rows, desc="encoding"):
            wav_path = wavs_dir / row["file_name"]
            if not wav_path.exists():
                skipped += 1
                continue

            try:
                latent = codec.encode_file(wav_path)  # (1, T, D)
            except Exception as e:
                print(f"  [warn] encode failed {wav_path.name}: {e}", file=sys.stderr)
                skipped += 1
                continue

            num_frames = latent.shape[1]
            if num_frames == 0 or num_frames > MAX_FRAMES:
                skipped += 1
                continue

            stem    = wav_path.stem
            pt_path = latent_dir / f"{stem}.pt"
            torch.save(latent.squeeze(0), pt_path)  # (T, D)

            # manifest_dir からの相対パス（dataset.py の _resolve_latent_path に合わせる）
            manifest_dir  = manifest_path.parent.resolve()
            latent_relpath = pt_path.resolve().relative_to(manifest_dir)

            record = {
                "text":        row["transcription"],
                "latent_path": str(latent_relpath),
                "speaker_id":  args.speaker_id,
                "num_frames":  num_frames,
            }
            mf.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"\n[done] written={written}, skipped={skipped}")
    print(f"[done] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
