#!/usr/bin/env python3
"""クラスタ内の「重心から遠い（＝混入の疑いが高い）」セグメントを優先的に洗い出す。

refine_speaker_clusters.py の試聴サンプルは「発話時間が長い順」に選ばれるため、
クラスタ境界付近に紛れ込んだ別話者（ゲスト等）のセグメントは長さで弾かれ、
チェックをすり抜けやすい。本スクリプトは学習用 wav 群の話者埋め込みを計算し、
クラスタ重心からの cosine 距離が遠い順にランキングして試聴用フォルダへコピーする。
全件を無作為に聴く代わりに、上位（＝最も怪しい）数十件だけ確認すれば済む。

使い方:
  uv run python scripts/rank_cluster_outliers.py \
    --wavs-dir data/tenchan/wavs --metadata data/tenchan/wavs/metadata.csv \
    --output-dir data/output/diarization/tenchan/outlier_review --top-k 80

出力:
  <output-dir>/ranking.csv           - 全件のランク・距離・書き起こし
  <output-dir>/review/rankNNN_distX.XXX_<元ファイル名>  - 上位 top-k の試聴用コピー
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from refine_speaker_clusters import extract_embeddings  # noqa: E402


def load_metadata(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    text_by_name: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text_by_name[row["file_name"]] = row.get("transcription", "")
    return text_by_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="クラスタ重心からの距離でセグメントをランキングし、混入疑いの試聴レビューを作る",
    )
    parser.add_argument("--wavs-dir", type=Path, required=True, help="学習用 wav ディレクトリ")
    parser.add_argument("--metadata", type=Path, default=None,
                         help="metadata.csv（file_name,transcription）。あれば書き起こしをCSVに含める")
    parser.add_argument("--output-dir", type=Path, required=True, help="出力ディレクトリ")
    parser.add_argument("--top-k", type=int, default=80,
                         help="試聴用にコピーする件数（重心から遠い順、デフォルト80）")
    parser.add_argument("--min-duration", type=float, default=0.3,
                         help="この秒数未満は埋め込み不安定のため除外（デフォルト0.3）")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA が利用できないため CPU に切り替えます", file=sys.stderr)
        args.device = "cpu"

    files = [(p, "cluster") for p in sorted(args.wavs_dir.glob("*.wav"))]
    if not files:
        print(f"ERROR: {args.wavs_dir} に wav がありません", file=sys.stderr)
        sys.exit(1)

    embeddings, durations, _f0, valid_indices = extract_embeddings(
        files, args.device, compute_f0=False,
    )

    centroid = embeddings.mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    cos_sim = embeddings @ centroid
    cos_dist = 1.0 - cos_sim

    order = np.argsort(-cos_dist)  # 遠い(=怪しい)順

    text_by_name = load_metadata(args.metadata)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    review_dir = args.output_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for rank, emb_idx in enumerate(order):
        file_idx = valid_indices[emb_idx]
        path, _ = files[file_idx]
        rows.append({
            "rank": rank,
            "file_name": path.name,
            "cosine_distance_to_centroid": round(float(cos_dist[emb_idx]), 4),
            "duration_sec": round(durations[emb_idx], 2),
            "transcription": text_by_name.get(path.name, ""),
        })

    csv_path = args.output_dir / "ranking.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["rank", "file_name", "cosine_distance_to_centroid",
                           "duration_sec", "transcription"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[output] ranking: {csv_path}")

    for row in rows[: args.top_k]:
        src = args.wavs_dir / row["file_name"]
        dist = row["cosine_distance_to_centroid"]
        dest = review_dir / f"rank{row['rank']:03d}_dist{dist:.3f}_{row['file_name']}"
        shutil.copy2(src, dest)

    print(f"\n{'=' * 70}")
    print(f"  全{len(rows)}件中、重心から遠い上位{min(args.top_k, len(rows))}件を試聴用にコピーしました")
    print(f"  試聴用: {review_dir}/")
    print("  ファイル名は rank(怪しい順)_dist(cosine距離)_元ファイル名 の形式です")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
