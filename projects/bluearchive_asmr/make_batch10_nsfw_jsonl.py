#!/usr/bin/env python3
"""ブルアカ3キャラ 第10バッチ: NSFW ASMR 第1弾（喘ぎ/うめき/吐息/誘惑）。"""
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

# (kind, text, caption)
BATCH10 = {
    "asuna": [
        ("aegi_soft",
         "んっ……あぁっ♡……ご主人様ぁ……だめぇ、そこぉ……声、出ちゃうよぉ……んぅっ♡",
         "甘えた高い声の少女が、快感で小さく甘く喘いでいる。息が弾み、語尾が甘く伸びて溶ける、耳元でこもった柔らかい響き。"),
        ("aegi_gaman",
         "んんっ……！……だめ、我慢……できな……っ、ふぅっ……んっ、んっ……♡",
         "声を必死に押し殺そうとして漏れる、くぐもったうめき混じりの喘ぎ。息を詰め、途切れ途切れに震える。"),
        ("toiki_ear",
         "はぁ……っ、ふぅ……ご主人様の耳、あったかいねぇ……はぁ……っ♡",
         "耳元にごく近い距離で吐く、湿った甘い吐息。ほとんど声にならない息だけの囁き、ゆっくり。"),
        ("sasoi",
         "ねぇ、ご主人様ぁ……今夜は、わたしだけ見ててぇ……♡……もっと、近くきてぇ……",
         "甘えた少女が、しなだれかかるように誘っている。とろけた甘い声で、ゆっくりねだるように囁く。"),
        ("umeki",
         "うぅ……んっ……はぁっ……もう、変になっちゃうよぉ……んぅっ……♡",
         "切なさに耐えかねて漏れる低いうめき声。喉の奥でくぐもり、息が荒く、語尾が震える。"),
        ("iki_agari",
         "はぁっ、はぁっ……ご主人様ぁ……まだ、だめぇ……はぁっ……♡",
         "息が完全に上がった状態の荒い呼吸。言葉の合間に激しい息継ぎが入り、声がかすれる。"),
    ],
    "karin": [
        ("aegi_soft",
         "……んっ、あっ……先生……そこ、は……っ……ふぅっ……♡",
         "クールで低めの声の少女が、不意の快感に短く喘ぐ。普段の硬い口調が崩れ、息が乱れる。"),
        ("aegi_gaman",
         "……くっ……声、出さない……んんっ……！……はぁっ……",
         "歯を食いしばって声を殺そうとするが漏れてしまう、押し殺したうめき。低く硬い声が震える。"),
        ("toiki_ear",
         "……はぁ……っ、ふぅ……先生。近い、ぞ……はぁ……っ♡",
         "耳のすぐそばで吐く、低く湿った吐息。声量を落とし、息だけがマイクに触れるような近さ。"),
        ("sasoi",
         "……先生。今夜は、私だけを見ろ……♡……こっちに、来い……",
         "クールな少女が命令口調のまま誘っている。低くゆっくり、独占欲の滲む甘さを含む。"),
        ("umeki",
         "……うっ……んんっ……くそ、体が……熱い……っ♡",
         "低く喉を鳴らすようなうめき声。悔しさと快感が混ざり、息が荒く途切れる。"),
        ("iki_agari",
         "はぁっ、はぁっ……先生……もう、限界……だ……っ",
         "息が上がりきった荒い呼吸。低い声がかすれ、言葉が息に飲まれて途切れる。"),
    ],
    "toki": [
        ("aegi_soft",
         "んっ……先生、体温上昇……あっ♡……センサー、誤作動……んぅっ……",
         "淡々とした報告口調の少女が、快感で言葉を保てず甘く喘ぐ。機械的な平坦さが崩れ、息が弾む。"),
        ("aegi_gaman",
         "……っ、抑制……失敗。んんっ……！……声帯、制御……不能です……",
         "無表情に自己制御を試みるが失敗して漏れる、押し殺したうめき。抑揚を欠いた声が震える。"),
        ("toiki_ear",
         "はぁ……っ、ふぅ……先生、呼吸が乱れて……はぁ……っ♡",
         "耳元での湿った吐息。淡々とした声のまま息だけが荒く、ごく近い距離のこもった響き。"),
        ("sasoi",
         "先生。今夜の任務は、私の相手です……♡……接近を、許可します……",
         "感情を抑えた報告口調のまま誘っている。平坦だがわずかに熱を帯び、ゆっくり間を置く。"),
        ("umeki",
         "うぅ……っ、んっ……エラー……思考、まとまりません……っ♡",
         "低くくぐもったうめき声。冷静さを失い、言葉が途切れて息に変わる。"),
        ("iki_agari",
         "はぁっ、はぁっ……先生……冷却、間に合いません……っ♡",
         "息が上がりきった荒い呼吸。淡々とした声がかすれ、息継ぎに飲まれる。"),
    ],
}


def seed_of(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16) % 100000


def main() -> None:
    outdir = REPO / "projects/bluearchive_asmr/jsonl"
    outdir.mkdir(parents=True, exist_ok=True)
    all_lines = []
    for sp, (ckpt, ref) in CHARS.items():
        for kind, text, caption in BATCH10[sp]:
            out = DESKTOP / sp / "nsfw" / f"{sp}_{kind}.wav"
            all_lines.append({
                "text": text, "caption": caption,
                "checkpoint": str(ckpt), "ref_wav": str(ref),
                "out_path": str(out), "seed": seed_of(text + sp + kind + "b10"),
                "duration_scale": 1.3, "tail_fade_ms": 120, "tail_pad_out_ms": 250,
            })
    (outdir / "_batch10_nsfw_all.jsonl").write_text(
        "\n".join(json.dumps(o, ensure_ascii=False) for o in all_lines) + "\n", encoding="utf-8")
    print(f"batch10(nsfw): {len(all_lines)} lines -> _batch10_nsfw_all.jsonl")


if __name__ == "__main__":
    main()
