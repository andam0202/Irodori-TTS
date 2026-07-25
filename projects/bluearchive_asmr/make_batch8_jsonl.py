#!/usr/bin/env python3
"""ブルアカ3キャラ 第8バッチASMR台本生成（秋/夜更かし/二度寝/肩もみ/手紙返信/夢分析）。"""
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

BATCH8 = {
    "asuna": [
        ("autumn", "ご主人様、秋だねぇ！紅葉見に行きたいなぁ。えへへ", "明るく元気に"),
        ("nightowl", "ご主人様、まだ起きてるの？夜更かしはだめだよぉ。一緒に寝よぉ", "慈しむように優しく"),
        ("sleep2", "ご主人様、もうちょっと寝ていい？えへへ、二度寝しちゃおう", "毛布ごしのようにこもった穏やかな声"),
        ("shoulder", "ご主人様、肩もみしてあげるねぇ。固いよぉ", "慈しむように優しく"),
        ("reply", "ご主人様、手紙のお返事書いたんだぁ。えへへ", "親しみを込めて"),
        ("dream2", "ご主人様、変な夢見ちゃった…。分析してほしいなぁ", "毛布ごしのようにこもった穏やかな声"),
    ],
    "karin": [
        ("autumn", "先生、秋。…任務の整理だ", "落ち着いて真面面に"),
        ("nightowl", "先生、夜更かしはするな。…体を壊す", "慈しむように優しく"),
        ("sleep2", "先生、二度寝か。…許す", "毛布ごしのようにこもった穏やかな声"),
        ("shoulder", "先生、肩をもむ。硬いな", "慈しむように優しく"),
        ("reply", "先生、手紙の返信だ。読め", "親しみを込めて"),
        ("dream2", "先生、夢を見た。…分析は不要だ", "毛布ごしのようにこもった穏やかな声"),
    ],
    "toki": [
        ("autumn", "先生、秋季。任務計画を更新します", "落ち着いて真面面に"),
        ("nightowl", "先生、夜更かしを検知。睡眠を推奨します", "慈しむように優しく"),
        ("sleep2", "先生、二度寝を許可します。…合理的です", "毛布ごしのようにこもった穏やかな声"),
        ("shoulder", "先生、肩のマッサージを提案します。硬直しています", "慈しむように優しく"),
        ("reply", "先生、手紙の返信を作成しました。確認を", "親しみを込めて"),
        ("dream2", "先生、夢のデータを分析。…あなたが主題でした", "毛布ごしのようにこもった穏やかな声"),
    ],
}


def seed_of(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16) % 100000


def main() -> None:
    outdir = REPO / "projects/bluearchive_asmr/jsonl"
    outdir.mkdir(parents=True, exist_ok=True)
    all_lines = []
    for sp, (ckpt, ref) in CHARS.items():
        for kind, text, caption in BATCH8[sp]:
            out = DESKTOP / sp / "asmr" / f"{sp}_{kind}.wav"
            all_lines.append({
                "text": text, "caption": caption,
                "checkpoint": str(ckpt), "ref_wav": str(ref),
                "out_path": str(out), "seed": seed_of(text + sp + kind + "b8"),
                "duration_scale": 1.3, "tail_fade_ms": 120, "tail_pad_out_ms": 250,
            })
    (outdir / "_batch8_all.jsonl").write_text(
        "\n".join(json.dumps(o, ensure_ascii=False) for o in all_lines) + "\n", encoding="utf-8")
    print(f"batch8: {len(all_lines)} lines -> _batch8_all.jsonl")


if __name__ == "__main__":
    main()
