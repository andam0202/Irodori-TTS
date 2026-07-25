#!/usr/bin/env python3
"""ブルアカ3キャラ 第4バッチASMR台本生成（biboran/英語/誕生日/寝言/ハグ/占い）。"""
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

BATCH4 = {
    "asuna": [
        ("biboran", "ご主人様、お肌のお手入れしましょうかぁ。ぷにぷにさせてねぇ", "慈しむように優しく"),
        ("english", "あい・らぶ・ゆー、ご主人様！えへへ、言い方合ってる？", "明るく元気に"),
        ("birthday", "ご主人様、お誕生日おめでとう！アスナからのお祝いだよぉ", "慈しむように優しく"),
        ("nemuri", "ん…ご主人様…だぁいすき…", "毛布ごしのようにこもった穏やかな声"),
        ("hug", "ご主人様、ぎゅーってしていい？えへへ、暖かい…", "慈しむように優しく"),
        ("fortune", "ご主人様の今日の運勢は…大吉だよぉ！えらいえらい", "明るく元気に"),
    ],
    "karin": [
        ("biboran", "先生、肌の手入れだ。動くな", "耳元でこもった柔らかい声"),
        ("english", "I love you. …何でもない", "慈しむように優しく囁く"),
        ("birthday", "先生、誕生日だ。…祝わせろ", "慈しむように優しく"),
        ("nemuri", "ん…先生…傍に…", "毛布ごしのようにこもった穏やかな声"),
        ("hug", "先生…少しだけ、このままでいいか", "慈しむように優しく"),
        ("fortune", "先生の運勢…悪くない。…安心しろ", "落ち着いて真面目に"),
    ],
    "toki": [
        ("biboran", "先生、スキンケアを提案します。最適な手法で実行します", "耳元でこもった柔らかい声"),
        ("english", "I love you. …翻訳不要です", "慈しむように優しく囁く"),
        ("birthday", "先生、誕生日を祝います。おめでとうございます", "慈しむように優しく"),
        ("nemuri", "ん…先生…専属…", "毛布ごしのようにこもった穏やかな声"),
        ("hug", "先生…抱擁を要求します。…合理的です", "慈しむように優しく"),
        ("fortune", "先生の運勢を分析。…良好です。いぇい", "落ち着いて真面目に"),
    ],
}


def seed_of(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16) % 100000


def main() -> None:
    outdir = REPO / "projects/bluearchive_asmr/jsonl"
    outdir.mkdir(parents=True, exist_ok=True)
    all_lines = []
    for sp, (ckpt, ref) in CHARS.items():
        for kind, text, caption in BATCH4[sp]:
            out = DESKTOP / sp / "asmr" / f"{sp}_{kind}.wav"
            all_lines.append({
                "text": text, "caption": caption,
                "checkpoint": str(ckpt), "ref_wav": str(ref),
                "out_path": str(out), "seed": seed_of(text + sp + kind + "b4"),
                "duration_scale": 1.3, "tail_fade_ms": 120, "tail_pad_out_ms": 250,
            })
    (outdir / "_batch4_all.jsonl").write_text(
        "\n".join(json.dumps(o, ensure_ascii=False) for o in all_lines) + "\n", encoding="utf-8")
    print(f"batch4: {len(all_lines)} lines -> _batch4_all.jsonl")


if __name__ == "__main__":
    main()
