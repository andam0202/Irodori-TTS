#!/usr/bin/env python3
"""
mamimi_v6 データを VoxCPM LoRA/SFT 学習用 JSONL manifest に変換する。

VoxCPM の load_audio_text_datasets は HuggingFace datasets の JSON ローダで読み込み、
デフォルトで "audio"(ファイルパス) + "text" カラムを期待する（src/voxcpm/training/data.py）。
mamimi_v6 の manifest.jsonl の latent_path("latents/seg_00005.pt") から seg 名を取り、
対応する wavs/*.wav の絶対パス + text で JSONL を出力する。

実行:
    uv run python scripts/prepare_voxcpm_mamimi.py
    # → data/mamimi_v6/voxcpm_{train,val}.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="mamimi_v6 → VoxCPM 学習用 JSONL 変換")
    ap.add_argument("--manifest", default="data/mamimi_v6/manifest.jsonl")
    ap.add_argument("--wavs-dir", default="data/mamimi_v6/wavs")
    ap.add_argument("--output-train", default="data/mamimi_v6/voxcpm_train.jsonl")
    ap.add_argument("--output-val", default="data/mamimi_v6/voxcpm_val.jsonl")
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--max-frames", type=int, default=300,
                    help="DACVAE latent frames 上限。超えるサンプルは除外（長すぎて OOM になるのを防ぐ）")
    args = ap.parse_args()

    wavs_dir = Path(args.wavs_dir).resolve()
    rows: list[dict] = []
    skipped = 0
    for line in Path(args.manifest).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        seg = Path(rec["latent_path"]).stem  # "latents/seg_00005.pt" -> "seg_00005"
        wav = wavs_dir / f"{seg}.wav"
        if not wav.exists():
            skipped += 1
            continue
        text = rec.get("text", "").strip()
        if not text:
            skipped += 1
            continue
        # num_frames は mamimi DACVAE の latent frames。極端に長いサンプルは事前除外
        # （最終的な長さフィルタは VoxCPM 側の max_batch_tokens で行われる）。
        if rec.get("num_frames", 0) > args.max_frames:
            skipped += 1
            continue
        rows.append({"audio": str(wav), "text": text})

    n_val = max(1, int(len(rows) * args.val_ratio))
    val = rows[:n_val]
    train = rows[n_val:]

    for path, data in [(args.output_train, train), (args.output_val, val)]:
        Path(path).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in data) + "\n",
            encoding="utf-8",
        )

    print(f"train: {len(train):4d} -> {args.output_train}")
    print(f"val:   {len(val):4d} -> {args.output_val}")
    print(f"skipped: {skipped}")


if __name__ == "__main__":
    main()
