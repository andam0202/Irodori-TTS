#!/usr/bin/env python3
"""ブルアカ3キャラ 第5バッチASMR台本生成（耳ふー/計算/天気/手紙/夢/目覚まし2）。"""
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

BATCH5 = {
    "asuna": [
        ("earblow", "ご主人様、耳、ふーふーしてあげるねぇ。あったかい？", "慈しむように優しく"),
        ("math", "ご主人様、一緒に計算しよぉ。えへへ、私、算数ちょっと苦手かも", "親しみを込めて"),
        ("weather", "ご主人様、今日の天気は晴れだよぉ！お出かけ日和だね", "明るく元気に"),
        ("letter", "ご主人様へ、手紙書いたんだぁ。えへへ、聞いてねぇ", "慈しむように優しく"),
        ("dream", "ご主人様、今日見た夢ね、ご主人様と一緒に…えへへ", "毛布ごしのようにこもった穏やかな声"),
        ("wakeup2", "ご主人様、そろそろ起きる時間だよぉ。朝だぁ！", "明るく元気に"),
    ],
    "karin": [
        ("earblow", "先生、耳にふーっと息を吹く。…くすぐったいか", "慈しむように優しく"),
        ("math", "先生、計算だ。私が検算する", "落ち着いて真面目に"),
        ("weather", "先生、本日の天気。晴れだ。任務に支障なし", "落ち着いて真面面に"),
        ("letter", "先生…手紙だ。読んでくれ", "慈しむように優しく"),
        ("dream", "先生…夢を見た。お前が出てきた", "毛布ごしのようにこもった穏やかな声"),
        ("wakeup2", "先生、二度寝は許さん。起きろ", "落ち着いて真面目に"),
    ],
    "toki": [
        ("earblow", "先生、耳への送風を提案します。ふー。温かいですか？", "慈しむように優しく"),
        ("math", "先生、計算を支援します。正解を提示します", "落ち着いて真面面に"),
        ("weather", "先生、天気予報です。晴れ。外出に最適です", "落ち着いて真面面に"),
        ("letter", "先生、手紙を読み上げます。…感情パラメータ上昇", "慈しむように優しく"),
        ("dream", "先生、夢のログです。あなたが登場しました", "毛布ごしのようにこもった穏やかな声"),
        ("wakeup2", "先生、再起床を推奨します。定刻を過ぎました", "落ち着いて真面面に"),
    ],
}


def seed_of(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16) % 100000


def main() -> None:
    outdir = REPO / "projects/bluearchive_asmr/jsonl"
    outdir.mkdir(parents=True, exist_ok=True)
    all_lines = []
    for sp, (ckpt, ref) in CHARS.items():
        for kind, text, caption in BATCH5[sp]:
            out = DESKTOP / sp / "asmr" / f"{sp}_{kind}.wav"
            all_lines.append({
                "text": text, "caption": caption,
                "checkpoint": str(ckpt), "ref_wav": str(ref),
                "out_path": str(out), "seed": seed_of(text + sp + kind + "b5"),
                "duration_scale": 1.3, "tail_fade_ms": 120, "tail_pad_out_ms": 250,
            })
    (outdir / "_batch5_all.jsonl").write_text(
        "\n".join(json.dumps(o, ensure_ascii=False) for o in all_lines) + "\n", encoding="utf-8")
    print(f"batch5: {len(all_lines)} lines -> _batch5_all.jsonl")


if __name__ == "__main__":
    main()
