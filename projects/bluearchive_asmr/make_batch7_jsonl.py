#!/usr/bin/env python3
"""ブルアカ3キャラ 第7バッチASMR台本生成（冬/春/雨/歯磨き/爪切り/靴下）。"""
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

BATCH7 = {
    "asuna": [
        ("winter", "ご主人様、寒いねぇ。一緒にこたつで暖まろうよぉ", "慈しむように優しく"),
        ("spring", "ご主人様、春だねぇ！お花見行きたいなぁ。えへへ", "明るく元気に"),
        ("rainy", "ご主人様、雨だよ。おうちでのんびりしよぉ", "毛布ごしのようにこもった穏やかな声"),
        ("teeth", "ご主人様、歯磨きしましょうかぁ。あーんしてねぇ", "慈しむように優しく"),
        ("nail", "ご主人様、爪切りましょぉ。じっとしててねぇ", "耳元でこもった柔らかい声"),
        ("socks", "ご主人様、靴下脱がせてあげるねぇ。お疲れ様", "慈しむように優しく"),
    ],
    "karin": [
        ("winter", "先生、冬だ。…暖を取るか", "落ち着いて真面面に"),
        ("spring", "先生、春。…任務の再編成だ", "落ち着いて真面面に"),
        ("rainy", "先生、雨。室内訓練に切り替える", "落ち着いて真面面に"),
        ("teeth", "先生、歯磨きだ。口を開けろ", "耳元でこもった柔らかい声"),
        ("nail", "先生、爪を切る。動くな", "耳元でこもった柔らかい声"),
        ("socks", "先生、靴下だ。脱がせる", "慈しむように優しく"),
    ],
    "toki": [
        ("winter", "先生、冬季を検知。暖房を最適化します", "落ち着いて真面面に"),
        ("spring", "先生、春季。任務計画を更新します", "落ち着いて真面面に"),
        ("rainy", "先生、降雨。屋内活動を推奨します", "落ち着いて真面面に"),
        ("teeth", "先生、歯磨きを支援します。開けてください", "耳元でこもった柔らかい声"),
        ("nail", "先生、爪切りを提案します。動かないでください", "耳元でこもった柔らかい声"),
        ("socks", "先生、靴下を脱がせます。お疲れ様です", "慈しむように優しく"),
    ],
}


def seed_of(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16) % 100000


def main() -> None:
    outdir = REPO / "projects/bluearchive_asmr/jsonl"
    outdir.mkdir(parents=True, exist_ok=True)
    all_lines = []
    for sp, (ckpt, ref) in CHARS.items():
        for kind, text, caption in BATCH7[sp]:
            out = DESKTOP / sp / "asmr" / f"{sp}_{kind}.wav"
            all_lines.append({
                "text": text, "caption": caption,
                "checkpoint": str(ckpt), "ref_wav": str(ref),
                "out_path": str(out), "seed": seed_of(text + sp + kind + "b7"),
                "duration_scale": 1.3, "tail_fade_ms": 120, "tail_pad_out_ms": 250,
            })
    (outdir / "_batch7_all.jsonl").write_text(
        "\n".join(json.dumps(o, ensure_ascii=False) for o in all_lines) + "\n", encoding="utf-8")
    print(f"batch7: {len(all_lines)} lines -> _batch7_all.jsonl")


if __name__ == "__main__":
    main()
