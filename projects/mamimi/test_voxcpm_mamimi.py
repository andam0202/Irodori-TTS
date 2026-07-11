#!/usr/bin/env python3
"""
mamimi（田中まみみ）音声で VoxCPM2 の全機能を網羅検証するスクリプト。

VoxCPM2 (OpenBMB, 2B, 30言語, tokenizer-free) の推論機能を mamimi の参照音声で検証し、
機能別ディレクトリに WAV を出力する。台詞・control は既存 mamimi テストスクリプト
(run_test_mamimi_v6.sh / run_nsfw_mamimi_v6.sh) から流用し、Irodori 側結果と直接比較可能。

実行:
    bash scripts/voxcpm.sh test                       # 全機能
    bash scripts/voxcpm.sh test --only 03,05          # 特定機能のみ
    bash scripts/voxcpm.sh test --denoise             # ZipEnhancer denoise 有効化

出力: data/output/voxcpm_test/{00_reference..10_nsfw}/*.wav + report.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from voxcpm import VoxCPM

# ─── mamimi 参照音声（manifest.jsonl の transcript を使用）───────────────────
# VoxCPM のゼロショットクローンは 3-10秒の参照推奨。seg_00005 は既存 mamimi_v6
# スクリプトと同一（比較一貫性）。長短3種で参照長の品質影響を比較する。
REFS = {
    "A_long": {  # 9.93s
        "wav": "data/mamimi_v6/wavs/seg_00005.wav",
        "text": "私服で人前に出ることも結構あるから買っておこうかなって思ってるんだけど",
    },
    "B_short": {  # 3.41s
        "wav": "data/mamimi_v6/wavs/seg_00338.wav",
        "text": "この辺のセット見てきますね",
    },
    "C_mid": {  # 7.46s
        "wav": "data/mamimi_v6/wavs/seg_00017.wav",
        "text": "違いますよ、プロデューサー私が話してたのはこっち",
    },
}

# ─── mamimi 台詞（既存 run_test_mamimi_v6.sh から流用）──────────────────────
TEXT_BASE = "プロデューサー、今日は私のために時間を作ってくれてありがとう。大事な話があるんだ。"
TEXT_SHORT = "別に普通ですよ。そんなに大したことじゃないし、気にしないでください。"
TEXT_CLONE = "こんにちは、プロデューサー。今日も一緒に頑張りましょうね。"

# ─── 感情 control ─ VoxCPM は英中推奨だが日本語キャラのため日本語 control の効きも検証
EMOTIONS_JA = [
    ("01_内気_緊張", "少し内気な10代の女の子。緊張しているが、勇気を振り絞って大切なことを伝えようとしている。声は小さめで、途切れ途切れになりがち。"),
    ("02_明るい_元気", "明るく元気な10代の女の子。親しい相手に楽しそうに、少し早口で話している。声は高めでハキハキとしている。"),
    ("03_怒り_感情的", "激しい怒りを感じている女の子。相手を責め立てるような強い口調で、感情的に声を荒らげている。"),
    ("04_冷静_淡々", "落ち着いた冷静な女の子。感情を抑えて、静かに淡々と話している。声は低めで穏やか。"),
    ("05_泣き声_悲痛", "深く傷つき、今にも泣き出しそうな女の子。声が震えており、悲痛なトーンで弱々しく話す。涙声が混じっている。"),
    ("06_呆れ_冷笑", "完全に呆れ返っている女の子。感情の起伏が乏しく、冷たいトーンで静かに突き放すように話す。"),
]
EMOTIONS_EN = [
    ("07_shy_tense_EN", "A shy teenage girl, slightly nervous, gathering courage to say something important. Soft, hesitant, breathy voice."),
    ("08_cheerful_EN", "A cheerful energetic teenage girl talking happily to a close friend. Bright, brisk, fast-paced voice."),
]

# ─── Voice Design 用 description（参照音声不要）────────────────────────────
VOICE_DESIGNS = [
    ("mamimi風_甘え_JA", "少し内気な10代の女の子。吐息混じりの甘い声で、親しい相手にゆっくり話しかける。"),
    ("低音_大人女性_JA", "20代前半の落ち着いた女性。低めの声で、ゆっくり丁寧に話す。"),
    ("mamimi_like_EN", "A young Japanese woman in her late teens, slightly shy, sweet breathy voice, speaking softly to a close friend."),
    ("deep_calm_woman_EN", "A calm woman in her early twenties with a deep, gentle voice, speaking slowly and politely."),
]

# ─── クロスリンガル台詞（mamimi の声で他言語）──────────────────────────────
TEXT_EN = "Hello everyone! Thank you so much for coming today. I'm really happy to see you all here."
TEXT_ZH = "大家好，非常感谢大家今天的到来，能见到你们我真的很高兴。"

# ─── NSFW 台詞（既存 run_nsfw_mamimi_v6.sh から流用）────────────────────────
NSFW = [
    ("01_甘い吐息", "はぁ、もう、勘弁してよ。", "甘えるような吐息混じりの声で、恥ずかしそうに小さく呟いている。距離感が近い。"),
    ("02_小さな喘ぎ", "んっ、だめ、声出ちゃう。", "声を押し殺そうとしている女の子。小さな喘ぎ声が漏れてしまい、恥ずかしがっている。"),
    ("03_耳元囁き", "ねぇ、好きだよ、プロデューサー。", "耳元で囁く女の子。非常に近い距離で、甘く吐息混じりに告白する。声は小さくふわふわ。"),
    ("04_誘惑", "ねぇ、今日はまだ帰らないでよ。", "甘えて引き止める女の子。少し上目遣いで、声に甘えと誘いが混じっている。"),
    ("05_事後の余韻", "はぁ、よかった。このままでいい、ずっと。", "事後の余韻に浸る女の子。満足げな吐息混じりの声。甘くふわふわとした雰囲気。"),
]

DEFAULT_OUT = "data/output/voxcpm_test"
HF_MODEL = "openbmb/VoxCPM2"


def build_final_text(text: str, control: str | None) -> str:
    """VoxCPM の (control)text 形式を構築（app.py / cli.py と同じロジック）。"""
    control = (control or "").strip()
    return f"({control}){text}" if control else text


def generate(model, out_path: Path, *, sr: int, text, reference_wav_path=None,
             prompt_wav_path=None, prompt_text=None, cfg_value=2.0,
             inference_timesteps=10, normalize=True, denoise=False, max_len=900) -> dict:
    """1件生成して RTF/長さを返す。"""
    t0 = time.time()
    wav = model.generate(
        text=text,
        reference_wav_path=str(reference_wav_path) if reference_wav_path else None,
        prompt_wav_path=str(prompt_wav_path) if prompt_wav_path else None,
        prompt_text=prompt_text,
        cfg_value=cfg_value,
        inference_timesteps=inference_timesteps,
        normalize=normalize,
        denoise=denoise,
        max_len=max_len,
    )
    dt = time.time() - t0
    dur = len(wav) / sr
    rtf = dt / dur if dur > 0 else float("inf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), wav, sr)
    print(f"  [RTF {rtf:.2f} | {dur:.1f}s | gen {dt:.1f}s] {out_path.name}", flush=True)
    return {"rtf": round(rtf, 3), "duration": round(dur, 2), "gen_time": round(dt, 2)}


def prepare_reference(out_dir: Path, sr_in: int = 16000) -> dict[str, Path]:
    """参照音声を 16kHz mono にリサンプルして 00_reference/ へ。"""
    ref_dir = out_dir / "00_reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    ref_paths: dict[str, Path] = {}
    lines = []
    for key, info in REFS.items():
        src = Path(info["wav"])
        wav_16k, _ = librosa.load(str(src), sr=sr_in, mono=True)
        dst = ref_dir / f"{key}_16k.wav"
        sf.write(str(dst), wav_16k, sr_in)
        shutil.copy2(str(src), str(ref_dir / f"{key}_original.wav"))
        ref_paths[key] = dst
        lines.append(f"{key} ({src.name}, 16kHz): {info['text']}")
    (ref_dir / "transcript.txt").write_text("\n".join(lines), encoding="utf-8")
    return ref_paths


# ─── 各機能 ─────────────────────────────────────────────────────────────────
def run_01_tts_default(model, sr, out_dir, report, denoise):
    print("[01] TTS default (参照音声なし・デフォルトボイス)", flush=True)
    d = out_dir / "01_tts_default"
    for text, name in [(TEXT_BASE, "base"), (TEXT_SHORT, "short")]:
        report[f"01/{name}"] = generate(model, d / f"tts_default_{name}.wav", sr=sr, text=text, denoise=denoise)


def run_02_voice_design(model, sr, out_dir, report, denoise):
    print("[02] Voice Design (テキスト記述からボイス作成・参照不要)", flush=True)
    d = out_dir / "02_voice_design"
    for name, desc in VOICE_DESIGNS:
        text = build_final_text(TEXT_BASE, desc)
        report[f"02/{name}"] = generate(model, d / f"vdesign_{name}.wav", sr=sr, text=text, denoise=denoise)


def run_03_clone(model, sr, out_dir, report, ref_paths, denoise):
    print("[03] Controllable Cloning (ゼロショット・参照長比較)", flush=True)
    d = out_dir / "03_clone_control"
    for key, ref in ref_paths.items():
        report[f"03/{key}"] = generate(model, d / f"clone_{key}.wav", sr=sr,
                                       text=TEXT_CLONE, reference_wav_path=ref, denoise=denoise)


def run_04_clone_style(model, sr, out_dir, report, ref_paths, denoise):
    print("[04] Controllable Cloning + style (感情制御)", flush=True)
    d = out_dir / "04_clone_style"
    ref = ref_paths["A_long"]
    for name, ctrl in EMOTIONS_JA + EMOTIONS_EN:
        text = build_final_text(TEXT_BASE, ctrl)
        report[f"04/{name}"] = generate(model, d / f"clone_style_{name}.wav", sr=sr,
                                        text=text, reference_wav_path=ref, denoise=denoise)


def run_05_ultimate(model, sr, out_dir, report, ref_paths, denoise):
    print("[05] Ultimate Cloning (prompt_wav+prompt_text+reference_wav)", flush=True)
    d = out_dir / "05_ultimate_clone"
    for key, ref in ref_paths.items():
        prompt_text = REFS[key]["text"]
        report[f"05/{key}"] = generate(model, d / f"ultimate_{key}.wav", sr=sr, text=TEXT_CLONE,
                                       prompt_wav_path=ref, prompt_text=prompt_text,
                                       reference_wav_path=ref, denoise=denoise)


def run_06_streaming(model, sr, out_dir, report, denoise):
    print("[06] Streaming API (generate_streaming)", flush=True)
    d = out_dir / "06_streaming"
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    chunks = list(model.generate_streaming(text=TEXT_BASE))
    dt = time.time() - t0
    wav = np.concatenate(chunks)
    dur = len(wav) / sr
    sf.write(str(d / "streaming_base.wav"), wav, sr)
    report["06/streaming"] = {
        "rtf": round(dt / dur, 3), "duration": round(dur, 2),
        "gen_time": round(dt, 2), "n_chunks": len(chunks),
    }
    print(f"  [RTF {dt/dur:.2f} | {dur:.1f}s | {len(chunks)} chunks] streaming_base.wav", flush=True)


def run_07_params(model, sr, out_dir, report, ref_paths, denoise):
    print("[07] パラメータグリッド (cfg × inference_timesteps)", flush=True)
    d = out_dir / "07_params"
    ref = ref_paths["A_long"]
    for cfg in [1.5, 2.0, 2.5, 3.0]:
        for steps in [5, 10, 20, 40]:
            name = f"cfg{cfg}_steps{steps}"
            report[f"07/{name}"] = generate(model, d / f"{name}.wav", sr=sr, text=TEXT_CLONE,
                                            reference_wav_path=ref, cfg_value=cfg,
                                            inference_timesteps=steps, denoise=denoise)


def run_08_crosslingual(model, sr, out_dir, report, ref_paths, denoise):
    print("[08] クロスリンガル (mamimi クローンで英語/中国語)", flush=True)
    d = out_dir / "08_crosslingual"
    ref = ref_paths["A_long"]
    report["08/en"] = generate(model, d / "crosslingual_en.wav", sr=sr,
                               text=TEXT_EN, reference_wav_path=ref, denoise=denoise)
    report["08/zh"] = generate(model, d / "crosslingual_zh.wav", sr=sr,
                               text=TEXT_ZH, reference_wav_path=ref, denoise=denoise)


def run_10_nsfw(model, sr, out_dir, report, ref_paths, denoise):
    print("[10] NSFW 表現 (既存スクリプト流用・息遣い/喘ぎ等)", flush=True)
    d = out_dir / "10_nsfw"
    ref = ref_paths["A_long"]
    for name, text, ctrl in NSFW:
        final = build_final_text(text, ctrl)
        report[f"10/{name}"] = generate(model, d / f"nsfw_{name}.wav", sr=sr,
                                        text=final, reference_wav_path=ref, denoise=denoise)


def main() -> None:
    ap = argparse.ArgumentParser(description="mamimi 音声で VoxCPM2 の全機能を検証")
    ap.add_argument("--output-dir", default=DEFAULT_OUT)
    ap.add_argument("--model", default=HF_MODEL, help="HF model id またはローカルパス")
    ap.add_argument("--only", default="", help="カンマ区切りで機能指定 (01,02,05,...)")
    ap.add_argument("--denoise", action="store_true", help="ZipEnhancer denoise 有効化（load_denoiser=True）")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    print(f"[init] model={args.model} load_denoiser={args.denoise}", flush=True)
    model = VoxCPM.from_pretrained(args.model, load_denoiser=args.denoise, optimize=True)
    sr = model.tts_model.sample_rate
    print(f"[init] output sample_rate={sr}", flush=True)

    ref_paths = prepare_reference(out_dir)

    funcs = {
        "01": lambda: run_01_tts_default(model, sr, out_dir, report, args.denoise),
        "02": lambda: run_02_voice_design(model, sr, out_dir, report, args.denoise),
        "03": lambda: run_03_clone(model, sr, out_dir, report, ref_paths, args.denoise),
        "04": lambda: run_04_clone_style(model, sr, out_dir, report, ref_paths, args.denoise),
        "05": lambda: run_05_ultimate(model, sr, out_dir, report, ref_paths, args.denoise),
        "06": lambda: run_06_streaming(model, sr, out_dir, report, args.denoise),
        "07": lambda: run_07_params(model, sr, out_dir, report, ref_paths, args.denoise),
        "08": lambda: run_08_crosslingual(model, sr, out_dir, report, ref_paths, args.denoise),
        "10": lambda: run_10_nsfw(model, sr, out_dir, report, ref_paths, args.denoise),
    }
    keys = [k.strip() for k in args.only.split(",") if k.strip()] if args.only else list(funcs.keys())

    for k in keys:
        if k not in funcs:
            print(f"[skip] unknown function: {k}", flush=True)
            continue
        funcs[k]()

    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {len(report)} files -> {out_dir}/", flush=True)
    print(f"[done] report -> {out_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
