#!/usr/bin/env python3
"""生成した喘ぎの句を切り出し、VAMMoan と同じ形で素材置き場へ並べる。

`projects/asuna_moan/split_probe.py` の切り出し（無音の谷・0.5〜3.5 秒・RMS −20 dBFS）を
話者ごとに回し、`m0-01.wav` … `m4-NN.wav` を VaM_Updater の
`projects/vam-animation/runs/_audio_lib/vammoan/<話者>/` へ置く。
**絶頂（o）は切り出さない**（句のまま 1 本。`docs/引き継ぎ_性行為の音.md` §1-c）。

    uv run python projects/ginko_moan_cast/build_packs.py
    # そのあと VaM_Updater 側で
    cd tools && uv run python godot/voice_pack.py --voice sayaka
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "projects/asuna_moan"))
from split_probe import RATE, load, save, segments  # noqa: E402

SRC = REPO / "outputs/ginko_moan_cast"
LIB = Path("/mnt/c/Users/mao0202/Desktop/VaM_Updater/projects/vam-animation/runs/_audio_lib/vammoan")
SPEAKERS = ["sayaka", "tenchan", "diana", "toki"]
TARGET_RMS = 0.1  # −20 dBFS


def norm(y: np.ndarray) -> np.ndarray:
    fade = int(RATE * 0.015)
    y = y.copy()
    y[:fade] *= np.linspace(0, 1, fade)
    y[-fade:] *= np.linspace(1, 0, fade)
    rms = float(np.sqrt(np.mean(y ** 2))) or 1e-6
    y = y * (TARGET_RMS / rms)
    peak = float(np.abs(y).max())
    return y * (0.95 / peak) if peak > 0.95 else y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speakers", nargs="*", default=SPEAKERS)
    ap.add_argument("--lib", type=Path, default=LIB)
    a = ap.parse_args()
    summary = {}
    for sp in a.speakers:
        raw = SRC / sp / "raw"
        if not raw.is_dir():
            print(f"※ {sp}: {raw} が無い", file=sys.stderr)
            continue
        dst = a.lib / sp
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)
        counters: dict[str, int] = {}
        meta = []
        for src in sorted(raw.glob("*.wav")):
            level = src.stem.split("_")[0]
            x = load(src)
            # 絶頂は句のまま 1 本（切ると声が途切れて使い物にならない）
            chunks = [(0, len(x))] if level == "o" else segments(x)
            for i0, i1 in chunks:
                y = norm(x[i0:i1])
                counters[level] = counters.get(level, 0) + 1
                name = f"{level}-{counters[level]:02d}.wav"
                save(dst / name, y)
                meta.append({"file": name, "source": src.name,
                             "seconds": round((i1 - i0) / RATE, 2)})
        (dst / "clips.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
        summary[sp] = dict(sorted(counters.items()))
        print(sp, summary[sp], "->", dst)
    print("SUMMARY", json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
