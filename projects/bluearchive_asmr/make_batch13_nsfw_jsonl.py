#!/usr/bin/env python3
"""ブルアカ3キャラ 第13バッチ: 口内音派生フェラチオ音声 + ASMR明示版。

キャプションは tenchan プロジェクトの実績書式に合わせ「20代の女性」と成人明示する。
水音の連続はテンポが速くなりやすいので duration_scale を 1.45 に上げている。
"""
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
BATCH13 = {
    "asuna": [
        ("fera_start",
         "れろぉ……♡……ん、ちゅ……先っぽから、ゆっくりねぇ……ちゅぷ……れろ、れろ……",
         "20代の女性が、口淫を焦らすようにゆっくり舌を這わせている。湿った舌の水音とリップ音が主体で、合間に甘えた囁きが混ざる。至近距離のこもった響き。"),
        ("fera_jupo",
         "じゅぷ……じゅぽ、じゅぽ……♡……んっ、んぅ……じゅる、じゅぽっ……",
         "20代の女性の口淫の水音。口が塞がった状態のくぐもった発声で、粘つく吸引音と鼻から抜ける荒い息が連続する。台詞はほぼ含まない。"),
        ("fera_nodooku",
         "んぐっ……♡……んぉ、じゅる……っ……んーっ、んぐ、んぐ……っ",
         "20代の女性が喉の奥まで含んで苦しげに呻いている。喉が詰まったくぐもった声と粘つく水音、鼻息が荒く途切れる。"),
        ("fera_iki",
         "ぷはっ……♡……はぁっ、はぁっ……ご主人様、気持ちいい？……えへへ……じゅる……",
         "20代の女性が口を離して息継ぎし、荒い呼吸のまま甘く問いかける。息が完全に上がり、語尾に湿った水音が戻る。"),
        ("fera_finish",
         "じゅぽ、じゅぽっ……♡……んっ、んーっ……！……んくっ……ん、ごくっ……",
         "20代の女性の口淫が終盤で加速する。吸引の水音が速く激しくなり、くぐもった呻きと嚥下音で終わる。"),
        ("asmr_fera_binaural",
         "じゅぷ……♡……ん、ちゅ……れろ、じゅる……っ……ん、ふぅ……っ",
         "ASMR音声。バイノーラル至近距離で録られた20代の女性の口淫の水音。台詞をほぼ含まず、湿った吸引音・舌の音・鼻呼吸だけを左右の耳に聴かせる。"),
    ],
    "karin": [
        ("fera_start",
         "……れろ……♡……ん、ちゅ……動くなよ。……ちゅぷ……れろ、れろ……",
         "20代の女性が、口淫を焦らすようにゆっくり舌を這わせている。低く抑えた声で短く命じ、湿った舌の水音とリップ音が主体。至近距離のこもった響き。"),
        ("fera_jupo",
         "じゅぷ……じゅぽ、じゅぽ……♡……んっ、んぅ……じゅる、じゅぽっ……",
         "20代の女性の口淫の水音。口が塞がった状態のくぐもった低い発声で、粘つく吸引音と鼻から抜ける荒い息が連続する。台詞はほぼ含まない。"),
        ("fera_nodooku",
         "んぐっ……♡……んぉ、じゅる……っ……んーっ、んぐ、んぐ……っ",
         "20代の女性が喉の奥まで含んで苦しげに呻いている。喉が詰まったくぐもった低い声と粘つく水音、鼻息が荒く途切れる。"),
        ("fera_iki",
         "……ぷはっ……♡……はぁっ、はぁっ……どうだ、先生……くっ……じゅる……",
         "20代の女性が口を離して息継ぎし、荒い呼吸のまま低い声で問う。息が完全に上がり、語尾に湿った水音が戻る。"),
        ("fera_finish",
         "じゅぽ、じゅぽっ……♡……んっ、んーっ……！……んくっ……ん、ごくっ……",
         "20代の女性の口淫が終盤で加速する。吸引の水音が速く激しくなり、くぐもった低い呻きと嚥下音で終わる。"),
        ("asmr_fera_binaural",
         "じゅぷ……♡……ん、ちゅ……れろ、じゅる……っ……ん、ふぅ……っ",
         "ASMR音声。バイノーラル至近距離で録られた20代の女性の口淫の水音。台詞をほぼ含まず、湿った吸引音・舌の音・低い鼻呼吸だけを左右の耳に聴かせる。"),
    ],
    "toki": [
        ("fera_start",
         "れろ……♡……ん、ちゅ……口腔での奉仕を、開始します……ちゅぷ……れろ……",
         "20代の女性が、淡々とした報告口調のまま舌を這わせている。湿った舌の水音とリップ音が主体で、抑揚を抑えた声が混ざる。至近距離のこもった響き。"),
        ("fera_jupo",
         "じゅぷ……じゅぽ、じゅぽ……♡……んっ、んぅ……じゅる、じゅぽっ……",
         "20代の女性の口淫の水音。口が塞がった状態のくぐもった発声で、粘つく吸引音と鼻から抜ける荒い息が連続する。台詞はほぼ含まない。"),
        ("fera_nodooku",
         "んぐっ……♡……んぉ、じゅる……っ……んーっ、んぐ、んぐ……っ",
         "20代の女性が喉の奥まで含んで苦しげに呻いている。喉が詰まったくぐもった声と粘つく水音、鼻息が荒く途切れる。"),
        ("fera_iki",
         "ぷはっ……♡……はぁっ、はぁっ……先生、反応は良好です……じゅる……",
         "20代の女性が口を離して息継ぎし、荒い呼吸のまま淡々と報告する。息が完全に上がり、語尾に湿った水音が戻る。"),
        ("fera_finish",
         "じゅぽ、じゅぽっ……♡……んっ、んーっ……！……んくっ……ん、ごくっ……",
         "20代の女性の口淫が終盤で加速する。吸引の水音が速く激しくなり、くぐもった呻きと嚥下音で終わる。"),
        ("asmr_fera_binaural",
         "じゅぷ……♡……ん、ちゅ……れろ、じゅる……っ……ん、ふぅ……っ",
         "ASMR音声。バイノーラル至近距離で録られた20代の女性の口淫の水音。台詞をほぼ含まず、湿った吸引音・舌の音・鼻呼吸だけを左右の耳に聴かせる。"),
    ],
}


def seed_of(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16) % 100000


def main() -> None:
    outdir = REPO / "projects/bluearchive_asmr/jsonl"
    outdir.mkdir(parents=True, exist_ok=True)
    all_lines = []
    for sp, (ckpt, ref) in CHARS.items():
        for kind, text, caption in BATCH13[sp]:
            out = DESKTOP / sp / "nsfw" / f"{sp}_{kind}.wav"
            all_lines.append({
                "text": text, "caption": caption,
                "checkpoint": str(ckpt), "ref_wav": str(ref),
                "out_path": str(out), "seed": seed_of(text + sp + kind + "b13"),
                "duration_scale": 1.45, "tail_fade_ms": 120, "tail_pad_out_ms": 250,
            })
    (outdir / "_batch13_nsfw_all.jsonl").write_text(
        "\n".join(json.dumps(o, ensure_ascii=False) for o in all_lines) + "\n", encoding="utf-8")
    print(f"batch13(nsfw/fera): {len(all_lines)} lines -> _batch13_nsfw_all.jsonl")


if __name__ == "__main__":
    main()
