#!/usr/bin/env python3
"""
VoxCPM2 + mamimi LoRA 適用推論。ゼロショット（03/05）と同じ条件で生成し、学習効果を比較する。

実行（学習完了後）:
    bash scripts/voxcpm.sh python projects/mamimi/infer_voxcpm_lora.py \
        --lora-ckpt data/output/voxcpm_lora/checkpoints/step_0000300

出力: data/output/voxcpm_test/11_lora/*.wav
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import soundfile as sf
from voxcpm.core import VoxCPM
from voxcpm.model.voxcpm import LoRAConfig

# ゼロショット検証（03/05）と同一条件
REF_A_WAV = "data/mamimi_v6/wavs/seg_00005.wav"
REF_A_TEXT = "私服で人前に出ることも結構あるから買っておこうかなって思ってるんだけど"
TEXT_CLONE = "こんにちは、プロデューサー。今日も一緒に頑張りましょうね。"
TEXT_BASE = "プロデューサー、今日は私のために時間を作ってくれてありがとう。大事な話があるんだ。"


def generate(model, out_path: Path, *, sr: int, **kwargs) -> None:
    t0 = time.time()
    wav = model.generate(cfg_value=2.0, inference_timesteps=10, normalize=True, denoise=False, **kwargs)
    dt = time.time() - t0
    dur = len(wav) / sr
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), wav, sr)
    print(f"  [RTF {dt/dur:.2f} | {dur:.1f}s] {out_path.name}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="mamimi LoRA 適用推論（ゼロショット比較用）")
    ap.add_argument("--lora-ckpt", required=True, help="LoRA チェックポイント dir（lora_config.json 含む）")
    ap.add_argument("--base-model", default="tools/voxcpm/models/VoxCPM2")
    ap.add_argument("--output-dir", default="data/output/voxcpm_test/11_lora")
    args = ap.parse_args()

    ckpt_dir = Path(args.lora_ckpt)
    lora_info = json.loads((ckpt_dir / "lora_config.json").read_text(encoding="utf-8"))
    lora_cfg = LoRAConfig(**lora_info["lora_config"])
    print(f"[init] LoRA ckpt: {ckpt_dir} (r={lora_cfg.r}, alpha={lora_cfg.alpha})", flush=True)

    model = VoxCPM.from_pretrained(
        hf_model_id=args.base_model,
        load_denoiser=False,
        optimize=True,
        lora_config=lora_cfg,
        lora_weights_path=str(ckpt_dir),
    )
    sr = model.tts_model.sample_rate
    print(f"[init] sample_rate={sr}", flush=True)

    out_dir = Path(args.output_dir)

    # 1. LoRA + ゼロショットクローン（03 と比較）
    print("[gen] 1: LoRA + クローン（参照のみ）", flush=True)
    generate(model, out_dir / "lora_clone_A.wav", sr=sr, text=TEXT_CLONE, reference_wav_path=REF_A_WAV)

    # 2. LoRA + Ultimate Cloning（05 と比較）
    print("[gen] 2: LoRA + Ultimate Cloning", flush=True)
    generate(model, out_dir / "lora_ultimate_A.wav", sr=sr, text=TEXT_CLONE,
             prompt_wav_path=REF_A_WAV, prompt_text=REF_A_TEXT, reference_wav_path=REF_A_WAV)

    # 3. LoRA 単体（参照なし）— 学習した mamimi 声質がそのまま出るか
    print("[gen] 3: LoRA 単体（参照なし）", flush=True)
    generate(model, out_dir / "lora_noref_base.wav", sr=sr, text=TEXT_BASE)

    print(f"\n[done] 11_lora -> {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
