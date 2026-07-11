"""Qwen3-TTS / VoxCPM2 テスト生成スクリプトの共通ランナー.

scripts/test_qwen3tts_*.py / test_voxcpm_*.py の共通部分（argparse 定義・
モデルロード・生成ループ・manifest 出力）を集約する。各スクリプトは
データ辞書（LINES / INSTRUCTS / VOICE_BASE 等）の定義と build_specs への
組み立てだけを持つ。

CLI ではなく共有モジュール（アンダースコア接頭辞）。利用側は
`sys.path.insert(0, str(Path(__file__).resolve().parent))` を置いてから
`import _tts_testgen as testgen` する（scripts/ 直下兄弟 import の確立パターン）。
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GenSpec:
    """1ファイル分の生成指示."""

    filename: str
    language: str
    type_key: str
    text: str
    instruct: str
    seed: int
    log: str  # 生成時に `[filename] {log}` と表示する文字列


def build_specs(
    lines: dict[str, list[tuple[str, str]]],
    lang_code: dict[str, str],
    instruct_for: Callable[[str, str], str],
    *,
    languages: Iterable[str],
    types: Iterable[str],
    seeds: Sequence[int],
) -> list[GenSpec]:
    """言語 × (タイプ, テキスト) × シードの標準ループ形状を GenSpec 列に展開する."""
    types = set(types)
    specs: list[GenSpec] = []
    for language in languages:
        for typ, text in lines[language]:
            if typ not in types:
                continue
            instruct = instruct_for(language, typ)
            for seed in seeds:
                specs.append(
                    GenSpec(
                        filename=f"{lang_code[language]}_{typ}_s{seed}.wav",
                        language=language,
                        type_key=typ,
                        text=text,
                        instruct=instruct,
                        seed=seed,
                        log=f"{text[:40]}...",
                    )
                )
    return specs


def make_parser(
    description: str,
    *,
    output_dir: Path,
    model_dir: Path,
    languages: Sequence[str],
    types: Sequence[str],
    seeds: Sequence[int],
    engine: str,
) -> argparse.ArgumentParser:
    """テスト生成スクリプト共通の ArgumentParser を作る（engine: "qwen" | "voxcpm"）."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output-dir", default=str(output_dir))
    parser.add_argument("--model-dir", default=str(model_dir))
    parser.add_argument("--languages", nargs="+", default=list(languages), choices=list(languages))
    parser.add_argument("--types", nargs="+", default=list(types), choices=list(types))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(seeds))
    if engine == "qwen":
        parser.add_argument("--temperature", type=float, default=0.9)
        parser.add_argument("--top-p", type=float, default=0.9)
    elif engine == "voxcpm":
        parser.add_argument("--cfg", type=float, default=2.0)
        parser.add_argument("--timesteps", type=int, default=10)
    else:
        raise ValueError(f"unknown engine: {engine!r}")
    parser.add_argument(
        "--dry-run", action="store_true", help="生成せず GenSpec 一覧を表示して終了"
    )
    return parser


def _print_dry_run(specs: list[GenSpec]) -> None:
    for s in specs:
        print(f"[dry-run] {s.filename} seed={s.seed} instruct={s.instruct[:60]}...")
    print(f"[dry-run] total: {len(specs)} files")


def _write_manifest(out_dir: Path, manifest: list[dict], manifest_name: str) -> None:
    (out_dir / manifest_name).write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"done: {len(manifest)} files -> {out_dir}")


def run_qwen_design(
    specs: list[GenSpec],
    *,
    model_dir: str,
    output_dir: str,
    temperature: float,
    top_p: float,
    repetition_penalty: float = 1.05,
    type_field: str = "type",
    manifest_name: str = "manifest.json",
    dry_run: bool = False,
) -> None:
    """Qwen3-TTS VoiceDesign で specs を一括生成し manifest を書き出す."""
    if dry_run:
        _print_dry_run(specs)
        return

    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {model_dir} ...")
    model = Qwen3TTSModel.from_pretrained(
        model_dir,
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )

    manifest = []
    for s in specs:
        torch.manual_seed(s.seed)
        print(f"[{s.filename}] {s.log}")
        wavs, sr = model.generate_voice_design(
            text=s.text,
            instruct=s.instruct,
            language=s.language,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        sf.write(str(out_dir / s.filename), wavs[0], sr)
        manifest.append(
            {"file": s.filename, "language": s.language, type_field: s.type_key,
             "seed": s.seed, "text": s.text, "instruct": s.instruct, "sr": sr}
        )

    _write_manifest(out_dir, manifest, manifest_name)


def run_voxcpm(
    specs: list[GenSpec],
    *,
    model_dir: str,
    output_dir: str,
    cfg: float,
    timesteps: int,
    manifest_name: str = "manifest.json",
    dry_run: bool = False,
) -> None:
    """VoxCPM2 Voice Design（`(description)text` 埋め込み方式）で specs を一括生成する.

    GenSpec.instruct を description として使う（manifest でも "description"）。
    """
    if dry_run:
        _print_dry_run(specs)
        return

    import soundfile as sf
    import torch
    from voxcpm import VoxCPM

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {model_dir} ...")
    model = VoxCPM.from_pretrained(model_dir, load_denoiser=False)
    sample_rate = model.tts_model.sample_rate
    print(f"sample_rate={sample_rate}")

    manifest = []
    for s in specs:
        torch.manual_seed(s.seed)
        full_text = f"({s.instruct}){s.text}"
        print(f"[{s.filename}] {s.log}")
        wav = model.generate(
            text=full_text,
            cfg_value=cfg,
            inference_timesteps=timesteps,
            normalize=True,
            denoise=False,
            retry_badcase=True,
        )
        sf.write(str(out_dir / s.filename), wav, sample_rate)
        manifest.append(
            {"file": s.filename, "language": s.language, "type": s.type_key, "seed": s.seed,
             "text": s.text, "description": s.instruct, "sr": sample_rate}
        )

    _write_manifest(out_dir, manifest, manifest_name)
