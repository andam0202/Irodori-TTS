#!/usr/bin/env python3
"""ブルアカ3キャラ 追加ASMR種類の batch_infer マニフェスト生成（第2バッチ以降）。
make_asmr_jsonl.py と同じ CHARS/REPO/DESKTOP を参照。Tips遵守(通常愛情caption・duration_scale 1.3)。
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

ADDITIONAL = {
    "asuna": [
        ("shampoo", "ご主人様、シャンプーしましょうかぁ。目、閉じててねぇ", "耳元でこもった柔らかい声"),
        ("backwash", "ご主人様、背中流しますねぇ。力加減へたっぴだけど、許してね", "慈しむように優しく"),
        ("count", "いーち、にーい、さーん、しーい。えへへ、数え歌好きなんだぁ", "穏やかにゆっくり"),
        ("reading", "むかしむかし、あるところに…。ご主人様、ねんねするまで読むねぇ", "毛布ごしのようにこもった穏やかな声"),
        ("radio", "ぶるあかラジオ、始まりまーす！今日もアスナがお届けだよぉ", "明るく元気に"),
        ("earmomi", "ご主人様、耳、もみもみしてあげるねぇ。気持ちいい？", "慈しむように優しく"),
    ],
    "karin": [
        ("shampoo", "先生、頭を洗う。動くな", "耳元でこもった柔らかい声"),
        ("backwash", "先生、背中だ。任せろ", "慈しむように優しく"),
        ("count", "一、二、三、四…。…任務の確認だ", "落ち着いて真面目に"),
        ("reading", "昔話を読む。…静かに聞け", "毛布ごしのようにこもった穏やかな声"),
        ("radio", "…ラジオ、始まる。先生、聞け", "落ち着いて真面目に"),
        ("earmomi", "先生、耳を揉む。…硬いな", "慈しむように優しく"),
    ],
    "toki": [
        ("shampoo", "先生、シャンプーを提案します。目を閉じてください", "耳元でこもった柔らかい声"),
        ("backwash", "先生、背中を洗浄します。動かないでください", "慈しむように優しく"),
        ("count", "一、二、三、四…。正確にカウントします", "落ち着いて真面目に"),
        ("reading", "先生、絵本を読みます。おやすみなさい", "毛布ごしのようにこもった穏やかな声"),
        ("radio", "先生、ラジオ形式の配信を開始します。…いぇい", "落ち着いて真面目に"),
        ("earmomi", "先生、耳のマッサージを提案します", "慈しむように優しく"),
    ],
}


def seed_of(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16) % 100000


def main() -> None:
    outdir = REPO / "projects/bluearchive_asmr/jsonl"
    outdir.mkdir(parents=True, exist_ok=True)
    all_lines = []
    for sp, (ckpt, ref) in CHARS.items():
        lines = []
        for kind, text, caption in ADDITIONAL[sp]:
            out = DESKTOP / sp / "asmr" / f"{sp}_{kind}.wav"
            obj = {
                "text": text, "caption": caption,
                "checkpoint": str(ckpt), "ref_wav": str(ref),
                "out_path": str(out), "seed": seed_of(text + sp + kind + "add"),
                "duration_scale": 1.3, "tail_fade_ms": 120, "tail_pad_out_ms": 250,
            }
            lines.append(obj)
            all_lines.append(obj)
        p = outdir / f"{sp}_additional.jsonl"
        p.write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in lines) + "\n", encoding="utf-8")
        print(f"{sp}: {len(lines)} lines -> {p}")
    (outdir / "_additional_all.jsonl").write_text(
        "\n".join(json.dumps(o, ensure_ascii=False) for o in all_lines) + "\n", encoding="utf-8")
    print(f"combined: {len(all_lines)} lines -> _additional_all.jsonl")


if __name__ == "__main__":
    main()
