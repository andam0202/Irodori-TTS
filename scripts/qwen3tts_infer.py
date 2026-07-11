"""Qwen3-TTS 汎用推論 CLI（VoiceDesign / ボイスクローン）.

bash scripts/qwen3tts.sh design --text "..." --instruct "..." --output out.wav
bash scripts/qwen3tts.sh clone --text "..." --ref-audio ref.wav --ref-text "..." --output out.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "tools" / "qwen3-tts" / "models"
DEFAULT_DESIGN_MODEL = MODELS_DIR / "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
DEFAULT_BASE_MODEL = MODELS_DIR / "Qwen3-TTS-12Hz-1.7B-Base"

# Qwen3-TTS の language 引数に渡せる値（Auto は言語自動判定）
LANGUAGES = [
    "Auto", "Chinese", "English", "Japanese", "Korean", "German",
    "French", "Russian", "Portuguese", "Spanish", "Italian",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-TTS inference (design / clone)")
    parser.add_argument("--mode", choices=["design", "clone"], required=True)
    parser.add_argument("--text", required=True, help="生成するテキスト")
    parser.add_argument("--language", default="Auto", choices=LANGUAGES)
    parser.add_argument("--output", required=True, help="出力 wav パス")
    parser.add_argument("--model-dir", default=None, help="モデルディレクトリ（省略時はモードに応じた既定）")
    # design 用
    parser.add_argument("--instruct", default="", help="ボイス・スタイルの自然言語記述（design 用）")
    # clone 用
    parser.add_argument("--ref-audio", default=None, help="参照音声 wav（clone 用、3秒以上推奨）")
    parser.add_argument("--ref-text", default=None, help="参照音声の書き起こし（clone 用）")
    parser.add_argument("--x-vector-only", action="store_true", help="話者埋め込みのみ使用（ref-text 不要、品質低下）")
    # サンプリング
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    args = parser.parse_args()

    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    model_dir = args.model_dir or (DEFAULT_DESIGN_MODEL if args.mode == "design" else DEFAULT_BASE_MODEL)
    model = Qwen3TTSModel.from_pretrained(
        str(model_dir),
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )

    gen_kwargs: dict = {
        "do_sample": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
    }
    if args.max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = args.max_new_tokens

    if args.mode == "design":
        wavs, sr = model.generate_voice_design(
            text=args.text,
            instruct=args.instruct,
            language=args.language,
            **gen_kwargs,
        )
    else:
        if not args.ref_audio:
            parser.error("--ref-audio は clone モードで必須です")
        if not args.x_vector_only and not args.ref_text:
            parser.error("--ref-text が無い場合は --x-vector-only を指定してください")
        wavs, sr = model.generate_voice_clone(
            text=args.text,
            language=args.language,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            x_vector_only_mode=args.x_vector_only,
            **gen_kwargs,
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), wavs[0], sr)
    print(f"wrote {out} ({len(wavs[0]) / sr:.2f}s @ {sr}Hz)")


if __name__ == "__main__":
    main()
