#!/usr/bin/env python3
"""
metadata.csv の WAV ファイルを DACVAE でエンコードし、
学習用マニフェスト（JSONL）を生成するスクリプト。

prepare_manifest.py は audiofolder の data_dir 形式に非対応なため、
DACVAECodec を直接呼び出してエンコードする。

入力:
  data/mamimi/wavs/metadata.csv
  data/mamimi/wavs/seg_*.wav

出力:
  data/mamimi/latents/seg_XXXXX.pt  (DACVAE エンコード済みレイテント)
  data/mamimi/manifest.jsonl         (学習マニフェスト)
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# ---- 設定 ----
WAVS_DIR       = Path("data/mamimi/wavs")
LATENT_DIR     = Path("data/mamimi/latents")
MANIFEST_PATH  = Path("data/mamimi/manifest.jsonl")
SPEAKER_ID     = "mamimi"
DEVICE         = "cuda"
MAX_FRAMES     = 750  # train_500m_v2_lora.yaml の max_latent_steps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from irodori_tts.codec import DACVAECodec


def main() -> None:
    csv_path = WAVS_DIR / "metadata.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} が見つかりません", file=sys.stderr)
        sys.exit(1)

    LATENT_DIR.mkdir(parents=True, exist_ok=True)

    print("[codec] loading DACVAE...")
    codec = DACVAECodec.load(device=DEVICE, normalize_db=-16.0)
    print("[codec] ready.")

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"[encode] {len(rows)} samples -> {LATENT_DIR}")

    written = 0
    skipped = 0

    with open(MANIFEST_PATH, "w", encoding="utf-8") as mf:
        for row in tqdm(rows, desc="encoding"):
            wav_path = WAVS_DIR / row["file_name"]
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

            stem      = wav_path.stem
            pt_path   = LATENT_DIR / f"{stem}.pt"
            torch.save(latent.squeeze(0), pt_path)  # (T, D)

            # manifest_dir からの相対パス（dataset.py の _resolve_latent_path に合わせる）
            manifest_dir  = MANIFEST_PATH.parent.resolve()
            latent_relpath = pt_path.resolve().relative_to(manifest_dir)

            record = {
                "text":         row["transcription"],
                "latent_path":  str(latent_relpath),
                "speaker_id":   SPEAKER_ID,
                "num_frames":   num_frames,
            }
            mf.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"\n[done] written={written}, skipped={skipped}")
    print(f"[done] manifest -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
