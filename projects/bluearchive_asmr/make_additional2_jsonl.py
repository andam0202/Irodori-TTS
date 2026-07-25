#!/usr/bin/env python3
"""ブルアカ3キャラ 第3バッチASMR台本生成。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path("/mnt/c/Users/mao0202/Documents/GitHub/Irodori-TTS")
DESKTOP = Path("/mnt/c/Users/mao0202/Desktop/bluearchive_asmr")

CHARS = {
    "asuna": (REPO / "data/lora/asuna_v1/checkpoint_best_val_loss_0000300_0.764766.safetensors",
              REPO / "data/asuna/wavs/seg_00005.wav"),
    "karin": (REPO / "data/lora/karin_v1/checkpoint_best_val_loss_0001200_0.606443.safetensors",
              REPO / "data/karin/wavs/seg_00059.wav"),
    "toki": (REPO / "data/lora/toki_v1/checkpoint_best_val_loss_0000300_0.642423.safetensors",
             REPO / "data/toki/wavs/seg_00225.wav"),
}

BATCH3 = {
    "asuna": [
        ("humming", "ん〜♪ んん〜♪ えへへ、ご主人様の好きなメロディだよぉ", "穏やかにゆっくり"),
        ("outing", "ご主人様、今日お出かけしたの楽しかったよぉ。また一緒に行こっか", "親しみを込めて"),
        ("cookie", "ご主人様、クッキー焼いてみたんだぁ。形はいびつだけど、味は保証するよぉ", "明るく元気に"),
        ("cooking", "ご主人様、一緒に料理しよぉ。私が頑張るから、見ててねぇ", "親しみを込めて"),
        ("morning_prep", "ご主人様、朝の準備手伝うねぇ。えへへ、おそろいのネクタイどう？", "明るく元気に"),
        ("night_report", "ご主人様、今日も一日お疲れ様。えらいえらい。明日も頑張ろうねぇ", "慈しむように優しく"),
    ],
    "karin": [
        ("humming", "ん〜♪ …任務中の鼻歌だ。気にするな", "穏やかにゆっくり"),
        ("outing", "先生、本日の行動報告だ。…悪くない一日だった", "落ち着いて真面目に"),
        ("cookie", "先生、手製のクッキーだ。…休憩に食べろ", "慈しむように優しく"),
        ("cooking", "先生、料理を手伝う。…任せろ", "親しみを込めて"),
        ("morning_prep", "先生、朝の準備だ。身支度を済ませろ", "落ち着いて真面目に"),
        ("night_report", "先生、本日の任務完了を報告する。…よくやったな", "慈しむように優しく"),
    ],
    "toki": [
        ("humming", "ん〜♪ …データによると、リラックス効果があります", "穏やかにゆっくり"),
        ("outing", "先生、本日の外出記録です。…有意義な時間でした", "落ち着いて真面目に"),
        ("cookie", "先生、手製クッキーを提供します。…栄養バランスは最適です", "慈しむように優しく"),
        ("cooking", "先生、調理を支援します。手順に従ってください", "親しみを込めて"),
        ("morning_prep", "先生、朝の準備をサポートします。身支度を推奨します", "落ち着いて真面面に"),
        ("night_report", "先生、本日の活動報告です。…完璧な一日でした", "慈しむように優しく"),
    ],
}


def seed_of(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16) % 100000


def main() -> None:
    outdir = REPO / "projects/bluearchive_asmr/jsonl"
    outdir.mkdir(parents=True, exist_ok=True)
    all_lines = []
    for sp, (ckpt, ref) in CHARS.items():
        for kind, text, caption in BATCH3[sp]:
            out = DESKTOP / sp / "asmr" / f"{sp}_{kind}.wav"
            all_lines.append({
                "text": text, "caption": caption,
                "checkpoint": str(ckpt), "ref_wav": str(ref),
                "out_path": str(out), "seed": seed_of(text + sp + kind + "b3"),
                "duration_scale": 1.3, "tail_fade_ms": 120, "tail_pad_out_ms": 250,
            })
    (outdir / "_additional2_all.jsonl").write_text(
        "\n".join(json.dumps(o, ensure_ascii=False) for o in all_lines) + "\n", encoding="utf-8")
    print(f"batch3: {len(all_lines)} lines -> _additional2_all.jsonl")


if __name__ == "__main__":
    main()
