#!/usr/bin/env python3
"""果穂テスト生成用の batch_infer マニフェスト(JSONL)を生成する。

projects/kaho/kaho_test_lines.txt（1行1セリフ）を読み、checkpoint/ref_wav/out_path/seed/
tail_* を付与した JSONL を stdout に出力する。

使い方:
  uv run python projects/kaho/make_test_jsonl.py \
    --checkpoint data/lora/kaho_v1/kaho_v1_best.safetensors \
    --ref-wav data/kaho/wavs/seg_00170.wav \
    --out-dir data/output/kaho_test_v1 \
    > projects/kaho/kaho_test_v1.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="推論用 safetensors（LoRA マージ済み）")
    ap.add_argument("--ref-wav", default="data/kaho/wavs/seg_00170.wav", help="参照音声")
    ap.add_argument("--out-dir", default="data/output/kaho_test_v1", help="出力ディレクトリ")
    ap.add_argument(
        "--lines",
        default=str(Path(__file__).parent / "kaho_test_lines.txt"),
        help="1行1セリフのテキストファイル",
    )
    ap.add_argument("--name", default=None, help="出力ファイル名プレフィックス（省略時 out_dir 名）")
    ap.add_argument("--caption", default=None, help="全行に付与する caption（任意）")
    ap.add_argument("--tail-fade-ms", type=int, default=120)
    ap.add_argument("--tail-pad-out-ms", type=int, default=250)
    args = ap.parse_args()

    lines = [l.strip() for l in Path(args.lines).read_text(encoding="utf-8").splitlines() if l.strip()]
    # 出力ファイル名のプレフィックス: --name 指定優先、なければ out_dir 名から _test_v1 を除去
    stem = args.name or Path(args.out_dir).name.replace("_test_v1", "")
    for i, text in enumerate(lines):
        # 行内容から決定論的に seed を導出（再現性確保）
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16) % 100000
        obj: dict = {
            "text": text,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "ref_wav": str(Path(args.ref_wav).resolve()),
            "out_path": str((Path(args.out_dir) / f"{stem}_{i:03d}.wav").resolve()),
            "seed": seed,
            "tail_fade_ms": args.tail_fade_ms,
            "tail_pad_out_ms": args.tail_pad_out_ms,
        }
        if args.caption:
            obj["caption"] = args.caption
        print(json.dumps(obj, ensure_ascii=False))


if __name__ == "__main__":
    main()
