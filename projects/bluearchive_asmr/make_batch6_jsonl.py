#!/usr/bin/env python3
"""ブルアカ3キャラ 第6バッチASMR台本生成（夏/褒め/謝罪/お昼/おやすみ2/おにぎり）。"""
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

BATCH6 = {
    "asuna": [
        ("summer", "ご主人様、夏だねぇ！海行きたいなぁ。えへへ、水着選ぼうか", "明るく元気に"),
        ("praise", "ご主人様、えらいえらい！いつも頑張ってて尊敬しちゃう", "慈しむように優しく"),
        ("apologize", "ご主人様、ごめんねぇ。また失敗しちゃった…許して？", "慈しむように優しく"),
        ("lunch", "ご主人様、お昼にしよぉ。一緒にお弁当食べようね", "親しみを込めて"),
        ("goodnight2", "ご主人様、もう寝ようねぇ。明日も一緒だよ。おやすみぃ", "毛布ごしのようにこもった穏やかな声"),
        ("onigiri", "ご主人様、おにぎり握ったんだぁ。はい、あーん", "慈しむように優しく"),
    ],
    "karin": [
        ("summer", "先生、夏だ。…任務の服装、検討する", "落ち着いて真面面に"),
        ("praise", "先生、よくやった。…評価する", "慈しむように優しく"),
        ("apologize", "先生、すまない。…不手際だった", "慈しむように優しく"),
        ("lunch", "先生、昼食だ。一緒に取るか", "親しみを込めて"),
        ("goodnight2", "先生、もう休め。…また明日", "毛布ごしのようにこもった穏やかな声"),
        ("onigiri", "先生、手製のおにぎりだ。食え", "慈しむように優しく"),
    ],
    "toki": [
        ("summer", "先生、夏季を検知。任務計画を最適化します", "落ち着いて真面面に"),
        ("praise", "先生、仕事を称賛します。完璧です", "慈しむように優しく"),
        ("apologize", "先生、不手際を報告。謝罪します", "慈しむように優しく"),
        ("lunch", "先生、昼食を推奨します。摂取してください", "親しみを込めて"),
        ("goodnight2", "先生、睡眠を推奨します。おやすみなさい", "毛布ごしのようにこもった穏やかな声"),
        ("onigiri", "先生、おにぎりを提供します。あーん", "慈しむように優しく"),
    ],
}


def seed_of(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16) % 100000


def main() -> None:
    outdir = REPO / "projects/bluearchive_asmr/jsonl"
    outdir.mkdir(parents=True, exist_ok=True)
    all_lines = []
    for sp, (ckpt, ref) in CHARS.items():
        for kind, text, caption in BATCH6[sp]:
            out = DESKTOP / sp / "asmr" / f"{sp}_{kind}.wav"
            all_lines.append({
                "text": text, "caption": caption,
                "checkpoint": str(ckpt), "ref_wav": str(ref),
                "out_path": str(out), "seed": seed_of(text + sp + kind + "b6"),
                "duration_scale": 1.3, "tail_fade_ms": 120, "tail_pad_out_ms": 250,
            })
    (outdir / "_batch6_all.jsonl").write_text(
        "\n".join(json.dumps(o, ensure_ascii=False) for o in all_lines) + "\n", encoding="utf-8")
    print(f"batch6: {len(all_lines)} lines -> _batch6_all.jsonl")


if __name__ == "__main__":
    main()
