#!/usr/bin/env python3
"""ブルアカ3キャラ 第9バッチASMR台本生成（おやつ/お茶/ストレッチ/歌/お風呂/パジャマ）。"""
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

BATCH9 = {
    "asuna": [
        ("snack", "ご主人様、おやつ持ってきたよぉ。えへへ、一緒に食べよぉ", "親しみを込めて"),
        ("tea", "ご主人様、お茶淹れたよぉ。はい、どうぞぉ", "慈しむように優しく"),
        ("stretch", "ご主人様、一緒にストレッチしよぉ。体ほぐそうねぇ", "明るく元気に"),
        ("song", "ご主人様、歌うねぇ。えへへ、聴いてて", "穏やかにゆっくり"),
        ("bath_enter", "ご主人様、お風呂沸いたよぉ。先に入ってねぇ", "親しみを込めて"),
        ("pajama", "ご主人様、パジャマに着替えよぉ。えへへ、おそろいだよ", "明るく元気に"),
    ],
    "karin": [
        ("snack", "先生、おやつだ。食え", "親しみを込めて"),
        ("tea", "先生、茶を淹れた。飲め", "慈しむように優しく"),
        ("stretch", "先生、ストレッチだ。付き合え", "落ち着いて真面面に"),
        ("song", "先生…歌う。聞け", "穏やかにゆっくり"),
        ("bath_enter", "先生、風呂だ。先に入れ", "親しみを込めて"),
        ("pajama", "先生、寝巻きだ。着替えろ", "落ち着いて真面面に"),
    ],
    "toki": [
        ("snack", "先生、間食を提供します。摂取してください", "親しみを込めて"),
        ("tea", "先生、お茶を淹れました。最適温度です", "慈しむように優しく"),
        ("stretch", "先生、ストレッチを提案します。実行してください", "落ち着いて真面面に"),
        ("song", "先生、歌唱を開始します。…情感パラメータ上昇", "穏やかにゆっくり"),
        ("bath_enter", "先生、入浴を推奨します。温度は最適です", "親しみを込めて"),
        ("pajama", "先生、寝間着への着替えを推奨します", "落ち着いて真面面に"),
    ],
}


def seed_of(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16) % 100000


def main() -> None:
    outdir = REPO / "projects/bluearchive_asmr/jsonl"
    outdir.mkdir(parents=True, exist_ok=True)
    all_lines = []
    for sp, (ckpt, ref) in CHARS.items():
        for kind, text, caption in BATCH9[sp]:
            out = DESKTOP / sp / "asmr" / f"{sp}_{kind}.wav"
            all_lines.append({
                "text": text, "caption": caption,
                "checkpoint": str(ckpt), "ref_wav": str(ref),
                "out_path": str(out), "seed": seed_of(text + sp + kind + "b9"),
                "duration_scale": 1.3, "tail_fade_ms": 120, "tail_pad_out_ms": 250,
            })
    (outdir / "_batch9_all.jsonl").write_text(
        "\n".join(json.dumps(o, ensure_ascii=False) for o in all_lines) + "\n", encoding="utf-8")
    print(f"batch9: {len(all_lines)} lines -> _batch9_all.jsonl")


if __name__ == "__main__":
    main()
