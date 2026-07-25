#!/usr/bin/env python3
"""バッチ別 jsonl を「種類ごと」に統合する。

各 make_*_jsonl.py が出力した `jsonl/batches/_batch*.jsonl` を読み、
out_path の出力先ディレクトリと kind 名から種類を判定して
`jsonl/<category>.jsonl` へまとめ直す。

種類の判定:
    <char>/asmr/      → asmr_general  （通常 ASMR）
    <char>/nonverbal/ → nonverbal     （非言語発声）
    <char>/nsfw/      → kind が `asmr_` 始まり → nsfw_asmr（ASMR明示エロ）
                        kind が `fera_` 始まり → nsfw_fera（口内音派生）
                        それ以外              → nsfw     （通常 NSFW）

同一 out_path が複数バッチに現れた場合は、後のバッチ（ファイル名順で後方）を採用する。
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path("/mnt/c/Users/mao0202/Documents/GitHub/Irodori-TTS")
JSONL_DIR = REPO / "projects/bluearchive_asmr/jsonl"
BATCH_DIR = JSONL_DIR / "batches"

SPEAKERS = ("asuna", "karin", "toki")
CATEGORIES = ("asmr_general", "nonverbal", "nsfw", "nsfw_asmr", "nsfw_fera")


def classify(out_path: str) -> str:
    parts = Path(out_path).parts
    bucket = parts[-2]
    kind = Path(out_path).stem
    for sp in SPEAKERS:
        if kind.startswith(f"{sp}_"):
            kind = kind[len(sp) + 1 :]
            break
    if bucket == "asmr":
        return "asmr_general"
    if bucket == "nonverbal":
        return "nonverbal"
    if bucket == "nsfw":
        if kind.startswith("asmr_"):
            return "nsfw_asmr"
        if kind.startswith("fera_"):
            return "nsfw_fera"
        return "nsfw"
    raise ValueError(f"未知の出力先: {out_path}")


def sort_key(record: dict) -> tuple[int, str]:
    out_path = Path(record["out_path"])
    stem = out_path.stem
    for i, sp in enumerate(SPEAKERS):
        if stem.startswith(f"{sp}_"):
            return (i, stem[len(sp) + 1 :])
    return (len(SPEAKERS), stem)


def main() -> None:
    # out_path をキーに後勝ちで集約（バッチ名順に読む）
    buckets: dict[str, dict[str, dict]] = {c: {} for c in CATEGORIES}
    sources = sorted(BATCH_DIR.glob("*.jsonl"), key=lambda p: (len(p.name), p.name))
    if not sources:
        raise SystemExit(f"バッチ jsonl が見つかりません: {BATCH_DIR}")

    for src in sources:
        for line in src.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            buckets[classify(record["out_path"])][record["out_path"]] = record

    total = 0
    for category in CATEGORIES:
        records = sorted(buckets[category].values(), key=sort_key)
        if not records:
            continue
        dest = JSONL_DIR / f"{category}.jsonl"
        dest.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
            encoding="utf-8",
        )
        total += len(records)
        print(f"{category:14s}: {len(records):3d} lines -> {dest.name}")
    print(f"合計 {total} lines（{len(sources)} バッチから統合）")


if __name__ == "__main__":
    main()
