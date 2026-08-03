#!/usr/bin/env python3
"""既存の統合済みマニフェストを別モデル世代（例 v1 → v4）向けに作り替える。

checkpoint を `data/lora/<話者>_<世代>/` の best（val loss 最小）に差し替え、
out_path の出力ルートを世代ごとのフォルダへ張り替える。台詞・キャプション・
seed・duration_scale はそのまま引き継ぐので、**同じ台本を新モデルで再生成**できる。

    uv run python projects/bluearchive_asmr/retarget_jsonl.py --model v4
    uv run python projects/bluearchive_asmr/retarget_jsonl.py --model v4 --categories nsfw nsfw_asmr
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
JSONL_DIR = REPO / "projects/bluearchive_asmr/jsonl"

SPEAKERS = ("asuna", "karin", "toki")
CATEGORIES = ("asmr_general", "nonverbal", "nsfw", "nsfw_asmr", "nsfw_fera")

# モデル世代 -> wav 出力先ルート（世代を混ぜないよう必ず分ける）
OUTDIR_BY_MODEL = {
    "v1": Path("/mnt/c/Users/mao0202/Desktop/bluearchive_asmr"),
    "v4": Path("/mnt/c/Users/mao0202/Desktop/bluearchive_asmr_v4"),
}


def resolve_checkpoint(speaker: str, model: str) -> Path:
    lora_dir = REPO / "data/lora" / f"{speaker}_{model}"
    if not lora_dir.is_dir():
        raise SystemExit(f"LoRA ディレクトリがありません: {lora_dir}")
    # マージ済み .safetensors とアダプタ dir が同名で並ぶため、ファイルを優先する
    cands = sorted(lora_dir.glob("checkpoint_best_val_loss_*.safetensors"))
    if not cands:
        cands = sorted(p for p in lora_dir.glob("checkpoint_best_val_loss_*") if p.is_dir())
    if not cands:
        raise SystemExit(f"best checkpoint が見つかりません: {lora_dir}")

    def loss_of(path: Path) -> float:
        try:
            return float(path.name.removesuffix(".safetensors").split("_")[-1])
        except ValueError:
            return float("inf")

    return min(cands, key=loss_of)


def speaker_of(out_path: str) -> str:
    stem = Path(out_path).stem
    for sp in SPEAKERS:
        if stem.startswith(f"{sp}_"):
            return sp
    raise ValueError(f"話者を判定できません: {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(OUTDIR_BY_MODEL), default="v4", help="対象のモデル世代")
    ap.add_argument("--categories", nargs="*", default=list(CATEGORIES), help="対象カテゴリ")
    ap.add_argument("--outdir", type=Path, default=None, help="wav 出力ルート（既定: 世代ごとのフォルダ）")
    ap.add_argument(
        "--speakers", nargs="*", default=list(SPEAKERS),
        help="対象話者（既定: 全員）。学習が終わった話者だけ先に焼き直すときに使う",
    )
    ap.add_argument("--suffix", default="", help="出力 jsonl のファイル名に付ける接尾辞")
    args = ap.parse_args()

    outroot = args.outdir or OUTDIR_BY_MODEL[args.model]
    checkpoints = {sp: resolve_checkpoint(sp, args.model) for sp in args.speakers}
    print(f"モデル世代: {args.model}")
    for sp, ckpt in checkpoints.items():
        print(f"  {sp}: {ckpt.name}")

    dest_dir = JSONL_DIR / args.model
    dest_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for category in args.categories:
        src = JSONL_DIR / f"{category}.jsonl"
        if not src.exists():
            print(f"  skip（元マニフェストなし）: {src.name}")
            continue
        out_records = []
        for line in src.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            sp = speaker_of(rec["out_path"])
            if sp not in checkpoints:
                continue
            old = Path(rec["out_path"])
            # <ルート>/<話者>/<バケット>/<ファイル名> の構造を保ったままルートだけ張り替える
            rec["checkpoint"] = str(checkpoints[sp])
            rec["out_path"] = str(outroot / old.parts[-3] / old.parts[-2] / old.name)
            out_records.append(rec)
        if not out_records:
            continue
        dest = dest_dir / f"{category}{args.suffix}.jsonl"
        dest.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in out_records) + "\n", encoding="utf-8"
        )
        total += len(out_records)
        print(f"{category:14s}: {len(out_records):3d} lines -> {dest.relative_to(REPO)}")
    print(f"合計 {total} lines / 出力ルート: {outroot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
