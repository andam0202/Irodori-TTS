#!/usr/bin/env python3
"""ブルアカ3キャラ(asuna/karin/toki)の健全ASMR + 非言語ボイス用 batch_infer マニフェスト生成。

tenchan ASMR 確定レシピ(D方式)を踏襲:
  - caption: 通常の愛情caption(こもった柔らかい声/慈しむように)。ウィスパー/吐息captionはNG(母音濁り)
  - text: 母音を伸ばす(だぁいすきだよぉ)
  - duration_scale: 1.3
  - ref_wav: 各キャラの遅めの参照
  - tail_fade_ms: 120, tail_pad_out_ms: 250

使い方:
  uv run python projects/bluearchive_asmr/make_asmr_jsonl.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path("/mnt/c/Users/mao0202/Documents/GitHub/Irodori-TTS")
DESKTOP = Path("/mnt/c/Users/mao0202/Desktop/bluearchive_asmr")

# 各キャラ: checkpoint, ref_wav
CHARS = {
    "asuna": {
        "ckpt": REPO / "data/lora/asuna_v1/checkpoint_best_val_loss_0000300_0.764766.safetensors",
        "ref": REPO / "data/asuna/wavs/seg_00005.wav",
    },
    "karin": {
        "ckpt": REPO / "data/lora/karin_v1/checkpoint_best_val_loss_0001200_0.606443.safetensors",
        "ref": REPO / "data/karin/wavs/seg_00059.wav",
    },
    "toki": {
        "ckpt": REPO / "data/lora/toki_v1/checkpoint_best_val_loss_0000300_0.642423.safetensors",
        "ref": REPO / "data/toki/wavs/seg_00225.wav",
    },
}

# 健全ASMR: (種類key, text, caption)
ASMR = {
    "asuna": [
        ("earclean", "ご主人様、耳かきしましょうかぁ。じっとしててくださいねぇ", "耳元でこもった柔らかい声"),
        ("whisper", "ご主人様…だぁいすきだよぉ。ずっとそばにいるからね", "慈しむように優しく囁く"),
        ("sleep", "もう夜遅いよぉ。ご主人様、一緒にねんねしようね。おやすみなさぁい", "毛布ごしのようにこもった穏やかな声"),
        ("yoshiyoshi", "よしよぉし、ご主人様、今日もお疲れ様。えらいえらい", "慈しむように優しく"),
        ("morning", "ご主人様、朝ですよぉ！起きて起きて！元気な一日の始まりだぁ", "明るく元気に"),
        ("chat", "えへへ、今日も楽しかったねぇ。明日もいっぱい遊ぼうね", "親しみを込めて"),
        ("study", "ご主人様、一緒におべんきょうしよぉ。私が教えてあげるから", "丁寧に優しく"),
        ("heal", "ご主人様、お疲れ？じゃぁ私がなでなでしてあげるねぇ", "包み込むように優しく"),
    ],
    "karin": [
        ("earclean", "先生、耳の掃除だ。動くなよ", "耳元でこもった柔らかい声"),
        ("whisper", "先生…傍にいてくれて、助かる", "慈しむように優しく囁く"),
        ("sleep", "先生、もう休め。私が見張っている", "毛布ごしのようにこもった穏やかな声"),
        ("yoshiyoshi", "任務完了だ。先生、よくやったな", "慈しむように優しく"),
        ("morning", "先生、朝だ。起床しろ。任務の時間だ", "落ち着いて真面目に"),
        ("chat", "…静かだな。先生、このままでいい", "静かに穏やかに"),
        ("study", "先生、資料を整理した。確認してくれ", "丁寧に落ち着いて"),
        ("heal", "先生、無理はするな。…心配、してるんだ", "包み込むように優しく"),
    ],
    "toki": [
        ("earclean", "先生、耳かきを提案します。最適な角度で実行します", "耳元でこもった柔らかい声"),
        ("whisper", "先生…あなたのそばにいること、合理的です", "慈しむように優しく囁く"),
        ("sleep", "先生、睡眠を推奨します。おやすみなさい", "毛布ごしのようにこもった穏やかな声"),
        ("yoshiyoshi", "先生、お疲れ様です。完璧な仕事でした", "慈しむように優しく"),
        ("morning", "先生、定刻です。起床してください", "落ち着いて真面目に"),
        ("chat", "先生、静寂は悪くありません。…いぇい", "静かに穏やかに"),
        ("study", "先生、データを分析しました。確認を", "丁寧に落ち着いて"),
        ("heal", "先生、休憩を。…あなたの健康は、私の管理対象です", "包み込むように優しく"),
    ],
}

# 非言語ボイス: (種類key, text)  ※captionなし(ウィスパー/吐息captionは母音を濁すため)
NONVERBAL = {
    "asuna": [
        ("sleep_breath", "すぅ…はぁ…すぴー…"),
        ("sigh", "ふぅ…"),
        ("surprise", "はっ！"),
        ("laugh", "えへへ"),
        ("hum", "ん〜♪"),
        ("thinking", "うーん…"),
    ],
    "karin": [
        ("sleep_breath", "すぅ…はぁ…すぴー…"),
        ("sigh", "ふぅ…"),
        ("surprise", "はっ！"),
        ("laugh", "ふっ"),
        ("hum", "ん〜♪"),
        ("thinking", "うーん…"),
    ],
    "toki": [
        ("sleep_breath", "すぅ…はぁ…すぴー…"),
        ("sigh", "ふぅ…"),
        ("surprise", "はっ！"),
        ("laugh", "…ふっ"),
        ("hum", "ん〜♪"),
        ("thinking", "うーん…"),
    ],
}


def seed_of(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16) % 100000


def main() -> None:
    outdir = REPO / "projects/bluearchive_asmr/jsonl"
    outdir.mkdir(parents=True, exist_ok=True)
    for sp, info in CHARS.items():
        lines = []
        # ASMR
        for kind, text, caption in ASMR[sp]:
            out = DESKTOP / sp / "asmr" / f"{sp}_{kind}.wav"
            lines.append({
                "text": text, "caption": caption,
                "checkpoint": str(info["ckpt"]), "ref_wav": str(info["ref"]),
                "out_path": str(out), "seed": seed_of(text + sp + kind),
                "duration_scale": 1.3, "tail_fade_ms": 120, "tail_pad_out_ms": 250,
            })
        # 非言語
        for kind, text in NONVERBAL[sp]:
            out = DESKTOP / sp / "nonverbal" / f"{sp}_{kind}.wav"
            lines.append({
                "text": text,
                "checkpoint": str(info["ckpt"]), "ref_wav": str(info["ref"]),
                "out_path": str(out), "seed": seed_of(text + sp + kind + "nv"),
                "tail_fade_ms": 120, "tail_pad_out_ms": 250,
            })
        p = outdir / f"{sp}_asmr.jsonl"
        p.write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in lines) + "\n", encoding="utf-8")
        print(f"{sp}: {len(lines)} lines -> {p}")


if __name__ == "__main__":
    main()
