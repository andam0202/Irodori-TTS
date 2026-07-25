"""Qwen3-TTS 日本語 中年男性ボイス（命令 / 性行為）検証.

ゲーム制作用。中年男性（約45歳）の日本語ボイスを Qwen3-TTS で生成（Irodori ではなく）。
4タイプ（**成人男性限定**、未成年は扱わない）:
  command       : 強い口調の命令（軍人/ボス/権力者）
  command_threat: 脅しを含む命令
  sex_moan      : 性行為中の喘ぎ声（NSFW・成人男性）
  sex_climax    : クライマックスの声（NSFW・成人男性）

bash scripts/qwen3tts.sh python projects/multilingual/test_qwen3tts_man_ja.py --dry-run
bash scripts/qwen3tts.sh python projects/multilingual/test_qwen3tts_man_ja.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _tts_testgen as testgen  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_DIR / "tools" / "qwen3-tts" / "models" / "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
OUTPUT_DIR = PROJECT_DIR / "data" / "output" / "qwen3tts_man_ja"

LANG_CODE = {"Japanese": "ja"}

VOICE_BASE: dict[str, str] = {
    "Japanese": "A middle-aged man around 45 years old",
}

TYPE_DESC: dict[str, str] = {
    "command": "barking a sharp command, a deep powerful commanding voice, loud and intimidating, like a military officer or crime boss. Speaking forcefully with no warmth.",
    "command_threat": "delivering a threatening command, a deep low dangerous voice, cold and intimidating, with barely contained aggression. Like a ruthless warlord.",
    "sex_moan": "moaning in pleasure during intimacy, a deep breathless husky voice, low groans and heavy breathing, clearly aroused. Adult mature male, NON-violent, consensual intimacy.",
    "sex_climax": "at the peak of pleasure during intimacy, intense deep groaning, breathless and overwhelmed, a low rough voice crying out. Adult mature male at climax, NON-violent, consensual intimacy.",
}

LINES: dict[str, list[tuple[str, str]]] = {
    "Japanese": [
        ("command", "そこを動くな！俺の言う通りにしろ、今すぐだ！"),
        ("command_threat", "ひざまずけ！また逆らったら、後悔するぞ！"),
        ("sex_moan", "んっ……あぁ……いい……そこだ……あ……止めるな……はぁ……"),
        ("sex_climax", "あっ——！俺っ——！はぁ……あぁ……！"),
    ],
}


def main() -> None:
    parser = testgen.make_parser(
        "Qwen3-TTS Japanese middle-aged man (command / sex) test",
        output_dir=OUTPUT_DIR,
        model_dir=MODEL_DIR,
        languages=list(LINES.keys()),
        types=list(TYPE_DESC.keys()),
        seeds=[0],
        engine="qwen",
    )
    args = parser.parse_args()
    specs = testgen.build_specs(
        LINES,
        LANG_CODE,
        lambda language, typ: f"{VOICE_BASE[language]}, {TYPE_DESC[typ]}",
        languages=args.languages,
        types=args.types,
        seeds=args.seeds,
    )
    testgen.run_qwen_design(
        specs,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        temperature=args.temperature,
        top_p=args.top_p,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
