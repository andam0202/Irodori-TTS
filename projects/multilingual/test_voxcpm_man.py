"""VoxCPM2 中年男性ボイス（命令 / 性行為）Swahili/Arabic 生成.

Qwen3-TTS 非対応の Swahili(東アフリカ)・Arabic(北アフリカ) の中年男性（約45歳）
ボイスを VoxCPM2 Voice Design で生成。**成人男性限定**（未成年は扱わない）。
4タイプ（command / command_threat / sex_moan / sex_climax）。

bash scripts/voxcpm.sh python projects/multilingual/test_voxcpm_man.py
bash scripts/voxcpm.sh python projects/multilingual/test_voxcpm_man.py --languages Swahili --types command
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _tts_testgen as testgen  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_DIR / "tools" / "voxcpm" / "models" / "VoxCPM2"
OUTPUT_DIR = PROJECT_DIR / "data" / "output" / "voxcpm_man"

LANG_CODE = {"Swahili": "sw", "Arabic": "ar"}

VOICE_BASE: dict[str, str] = {
    "Swahili": "A middle-aged man around 45 years old",
    "Arabic": "A middle-aged man around 45 years old",
}

TYPE_DESC: dict[str, str] = {
    "command": "barking a sharp command, a deep powerful commanding voice, loud and intimidating, like a military officer or crime boss. Speaking forcefully with no warmth.",
    "command_threat": "delivering a threatening command, a deep low dangerous voice, cold and intimidating, with barely contained aggression. Like a ruthless warlord.",
    "sex_moan": "moaning in pleasure during intimacy, a deep breathless husky voice, low groans and heavy breathing, clearly aroused. Adult mature male, NON-violent, consensual intimacy.",
    "sex_climax": "at the peak of pleasure during intimacy, intense deep groaning, breathless and overwhelmed, a low rough voice crying out. Adult mature male at climax, NON-violent, consensual intimacy.",
}

LINES: dict[str, list[tuple[str, str]]] = {
    "Swahili": [
        ("command", "Simama pale! Usisogeze! Fanya hasa ninavyosema, sasa hivi!"),
        ("command_threat", "Vitia magotini! Nipinge tena na utajuta makubwa!"),
        ("sex_moan", "Mmm... naam... hiyo ndiyo... ah... usimamishe... hmm..."),
        ("sex_climax", "Ah—! Mimi—! Hah... naam... ahh...!"),
    ],
    "Arabic": [
        ("command", "توقف هناك! لا تتحرك! افعل بالضبط ما أقوله، الآن!"),
        ("command_threat", "على ركبتيك! عارضني مرة أخرى وستندم!"),
        ("sex_moan", "مم... نعم... هكذا... آه... لا تتوقف... همم..."),
        ("sex_climax", "آه—! أنا—! هاه... نعم... آاه—!"),
    ],
}


def main() -> None:
    parser = testgen.make_parser(
        "VoxCPM2 middle-aged man (command / sex) Swahili/Arabic generation",
        output_dir=OUTPUT_DIR,
        model_dir=MODEL_DIR,
        languages=list(LINES.keys()),
        types=list(TYPE_DESC.keys()),
        seeds=[0, 1],
        engine="voxcpm",
    )
    args = parser.parse_args()
    specs = testgen.build_specs(
        LINES,
        LANG_CODE,
        lambda language, typ: f"{VOICE_BASE[language]} {TYPE_DESC[typ]}",
        languages=args.languages,
        types=args.types,
        seeds=args.seeds,
    )
    testgen.run_voxcpm(
        specs,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        cfg=args.cfg,
        timesteps=args.timesteps,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
