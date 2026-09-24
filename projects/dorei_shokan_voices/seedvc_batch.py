"""Seed-VC の一括変換と、resemblyzer による話者類似度（Seed-VC 環境で動かす）。

``tools/seed-vc/inference.py`` は 1 ファイルごとにモデルを読み直すので、``load_models`` を
1 回だけ呼んで結果を使い回し、``inference.main`` を行ごとに呼ぶ。resemblyzer は Seed-VC の venv に
入っている（``projects/moan_vc`` で声の近さの測定に使ったのと同じ）。

    bash scripts/seedvc.sh python <abs>/seedvc_batch.py convert --jobs <jobs.json>
    bash scripts/seedvc.sh python <abs>/seedvc_batch.py score --jobs <score.json> --out <scores.json>

convert の jobs: [{"source": wav, "target": wav, "out": wav}, ...]（パスは絶対）
score の jobs:   [{"wav": wav, "ref": wav, "key": "..."}, ...] → {"key": cos, ...}
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path


def convert(jobs: list[dict], a: argparse.Namespace) -> None:
    sys.path.insert(0, os.getcwd())  # scripts/seedvc.sh が tools/seed-vc に cd してから呼ぶ
    import inference  # noqa: PLC0415

    base = argparse.Namespace(
        diffusion_steps=a.diffusion_steps,
        length_adjust=1.0,
        inference_cfg_rate=a.cfg_rate,
        f0_condition=a.f0_condition,
        auto_f0_adjust=a.f0_condition,  # F0 モデルのときは演者の音域を参照へ合わせる
        semi_tone_shift=0,
        checkpoint=None,
        config=None,
        fp16=True,
    )
    t0 = time.monotonic()
    loaded = inference.load_models(base)
    inference.load_models = lambda _args: loaded  # 2 回目以降は読み直さない
    print(f"[seedvc] models loaded ({time.monotonic() - t0:.1f}s)", flush=True)
    for i, j in enumerate(jobs, 1):
        t = time.monotonic()
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                **vars(base), source=j["source"], target=j["target"], output=tmp
            )
            inference.main(args)
            produced = next(Path(tmp).glob("*.wav"))
            Path(j["out"]).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(produced), j["out"])
        print(
            f"[seedvc] {i}/{len(jobs)} {Path(j['out']).name} ({time.monotonic() - t:.1f}s)",
            flush=True,
        )


def score(jobs: list[dict], out: Path) -> None:
    import numpy as np  # noqa: PLC0415
    from resemblyzer import VoiceEncoder, preprocess_wav  # noqa: PLC0415

    enc = VoiceEncoder("cpu", verbose=False)
    cache: dict[str, np.ndarray] = {}

    def emb(path: str) -> np.ndarray:
        if path not in cache:
            cache[path] = enc.embed_utterance(preprocess_wav(Path(path)))
        return cache[path]

    res = {}
    for j in jobs:
        a, b = emb(j["wav"]), emb(j["ref"])
        res[j["key"]] = round(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))), 4)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[score] {len(res)} -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["convert", "score"])
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out")
    ap.add_argument("--diffusion-steps", type=int, default=30)
    ap.add_argument("--cfg-rate", type=float, default=0.7)
    ap.add_argument("--f0-condition", action="store_true", help="F0 条件付き 44.1kHz モデルを使う")
    a = ap.parse_args()
    jobs = json.loads(Path(a.jobs).read_text(encoding="utf-8"))
    if a.mode == "convert":
        convert(jobs, a)
    else:
        score(jobs, Path(a.out))


if __name__ == "__main__":
    main()
