#!/usr/bin/env python3
"""ぎんこ（腹上トランプのチュートリアル役）の喘ぎ CV 候補を 4 話者ぶん量産する。

`projects/asuna_moan/make_mass_jsonl.py` のレシピをそのまま 4 話者へ広げたもの。
擬音だけのテキスト（母音を伸ばす）＋段ごとの固定キャプション＋`duration_scale` 1.3。
句の作り方・段の並び（m0 = ほとんど声にならない吐息 … m4 = 絶頂寸前、o = 絶頂）は
VaM_Updater の `docs/引き継ぎ_性行為の音.md` §1-c に従う。

    uv run python projects/ginko_moan_cast/make_cast_jsonl.py
    uv run python scripts/batch_infer.py --manifest projects/ginko_moan_cast/jsonl/cast.jsonl
    uv run python projects/asuna_moan/split_probe.py \
        --src outputs/ginko_moan_cast/<speaker>/raw --dst outputs/ginko_moan_cast/<speaker>/clips

**チェックポイントは各プロジェクトの現行世代**（sayaka だけ v4.1-Small ベース）。
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/ginko_moan_cast"
LORA = REPO / "data/lora"
DURATION_SCALE = 1.3

#: 話者ごとの（表示名, チェックポイント, 参照音声, キャプションの声の説明）。
SPEAKERS = {
    "sayaka": dict(
        label="美樹さやか（まどマギ／マギレコ）",
        ckpt=LORA / "sayaka_v1/checkpoint_best_val_loss_0000300_0.749929.safetensors",
        refs=["data/sayaka/wavs/seg_00056.wav"],
        voice="勢いのある快活な高めの声の10代後半の少女",
    ),
    "tenchan": dict(
        label="てんちゃん（おるすばん 双子ヒロイン）",
        ckpt=LORA / "tenchan_v2/tenchan_v2_best.safetensors",
        refs=["data/tenchan_v2/wavs/seg_00023.wav"],
        voice="あどけない高めの声の少女",
    ),
    "diana": dict(
        label="Diana（PRAGMATA）",
        ckpt=LORA / "diana_v4/diana_v4_best.safetensors",
        refs=["data/diana_v4/wavs/seg_00023.wav"],
        voice="天真爛漫で明るい高めの声の少女",
    ),
    "toki": dict(
        label="飛鳥馬トキ（ブルーアーカイブ）",
        ckpt=LORA / "toki_v4/checkpoint_best_val_loss_0000900_0.601365.safetensors",
        refs=["data/toki/wavs/seg_00225.wav"],
        voice="淡々とした平坦な声の20代の女性",
    ),
}

CAP = {
    "m0": "{voice}が小さく甘い声で息を漏らす。言葉は無く、短い吐息と『んっ』だけが静かに続く。",
    "m1": "{voice}が控えめに喘ぐ。息が上がり始め、甘く鼻にかかった短い声が続く。言葉は無い。",
    "m2": "{voice}が気持ちよさそうに喘ぐ。息が荒く、声が少し大きく、一定のリズムで続く。言葉は無い。",
    "m3": "{voice}が強く喘ぐ。声が高く大きくなり、息が乱れて切れ切れになる。言葉は無い。",
    "m4": "{voice}が激しく喘いで声を上げる。大きく高い声が連続し、泣きそうに震える。言葉は無い。",
    "o": "{voice}が絶頂で長く声を上げ、震えながら息を吐き切る。言葉は無い。",
}
PARTS = {
    "m0": ["んっ", "はぁ", "ふぅ", "んん", "はぁぁ", "ふ", "んぅ", "ふぁ", "ん", "はぁ…ん"],
    "m1": ["んっ", "はぁ", "あっ", "んん", "あぁ", "んぅ", "はぁ、んっ", "あ…んっ", "ふぅ、あ", "んっ、はぁ"],
    "m2": ["あっ、あっ", "んんっ", "はぁ、あぁっ", "んっ、あっ", "あぁ", "はぁっ", "あっ、んっ", "んっ、あぁ", "あっ、あっ、あっ", "はぁ、はぁっ"],
    "m3": ["あっ、あっ、あぁっ", "んんっ、あっ", "はぁっ、あぁっ", "あぁっ……！", "あっ、あっ", "んっ、あぁっ……！", "あぁぁっ", "はぁっ、あっ、あっ", "あっ、あぁっ", "んんっ……！"],
    "m4": ["あぁぁっ……！", "あっ、あっ、あっ", "んんんっ", "あぁぁっ", "はぁっ、あっ、あっ", "あぁぁぁっ……！", "あっ、あっ、あぁぁっ……！", "んんっ……！", "あぁっ、あっ", "あっ、あっ、あっ、あっ"],
}
O_TEXTS = [
    "あっ、あっ、あっ……んんっ……あぁぁぁっ……！……んんんんっ……！……はぁ……はぁ……",
    "あぁっ、あっ、あっ……！……あぁぁぁぁっ……！……んっ、んんっ……はぁぁ……ふぅ……",
    "あっ、あっ、あっ、あっ……あぁぁぁっ……！……んんっ……！……はぁ、はぁ……んっ……",
]
#: 段ごとの句の数（弱い段ほど多く：1 句から取れる本数が少ない）。
#: asuna（294 本）より小さく、切り出し後に各段 8 本以上を狙う。
COUNT = {"m0": 12, "m1": 12, "m2": 10, "m3": 8, "m4": 6}


def phrase(rng: random.Random, level: str) -> str:
    n = rng.randint(4, 6)
    return "……".join(rng.choice(PARTS[level]) for _ in range(n)) + "……"


def main() -> None:
    out = []
    for speaker, cfg in SPEAKERS.items():
        ckpt = Path(cfg["ckpt"])
        if not ckpt.exists():
            raise SystemExit(f"チェックポイントが無い: {ckpt}")
        refs = [REPO / r for r in cfg["refs"]]
        for r in refs:
            if not r.exists():
                raise SystemExit(f"参照音声が無い: {r}")
        rng = random.Random(int(hashlib.sha1(speaker.encode()).hexdigest()[:8], 16))
        rows, seen = [], set()
        for level, count in COUNT.items():
            made = 0
            while made < count:
                text = phrase(rng, level)
                if text in seen:
                    continue
                seen.add(text)
                rows.append((f"{level}_{made:03d}", level, text))
                made += 1
        rows += [(f"o_{i:03d}", "o", t) for i, t in enumerate(O_TEXTS)]
        for i, (key, level, text) in enumerate(rows):
            seed = int(hashlib.sha1(f"cast:{speaker}:{key}".encode()).hexdigest()[:8], 16) % 100000
            out.append({
                "text": text,
                "caption": CAP[level].format(voice=cfg["voice"]),
                "checkpoint": str(ckpt),
                "ref_wav": str(refs[i % len(refs)]),
                "out_path": str(OUT / speaker / "raw" / f"{key}.wav"),
                "seed": seed,
                "tail_fade_ms": 120,
                "tail_pad_out_ms": 250,
                "duration_scale": DURATION_SCALE,
            })
    dst = REPO / "projects/ginko_moan_cast/jsonl/cast.jsonl"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    print(len(out), "lines /", len(SPEAKERS), "speakers ->", dst)


if __name__ == "__main__":
    main()
