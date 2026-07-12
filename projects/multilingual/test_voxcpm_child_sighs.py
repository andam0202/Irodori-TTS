"""VoxCPM2 子供ボイス「吐息」Swahili/Arabic 生成.

Qwen3-TTS 非対応の Swahili/Arabic の子ども(8歳女児)の **非性的な吐息** 5種を生成。
5タイプ: relief / tired_sigh / nervous_breath / sleepy / wistful_sigh。NON-sexual。

bash scripts/voxcpm.sh python projects/multilingual/test_voxcpm_child_sighs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _tts_testgen as testgen  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_DIR / "tools" / "voxcpm" / "models" / "VoxCPM2"
OUTPUT_DIR = PROJECT_DIR / "data" / "output" / "voxcpm_child_sighs"

LANG_CODE = {"Swahili": "sw", "Arabic": "ar"}

VOICE_BASE: dict[str, str] = {
    "Swahili": "A little 8-year-old girl",
    "Arabic": "A little 8-year-old girl",
}

TYPE_DESC: dict[str, str] = {
    "relief": "letting out a big relieved sigh after tension, exhaling long and slow, a soft breath of relief. NON-sexual, ordinary relief.",
    "tired_sigh": "a heavy tired sigh after a long day, exhaling deeply with exhaustion, slightly deflated. NON-sexual, ordinary fatigue.",
    "nervous_breath": "taking short nervous breaths before something scary or important, shaky shallow breathing, anxious. NON-sexual, ordinary nervousness.",
    "sleepy": "a drowsy sleepy sigh, yawning softly, breathing slow and heavy with sleepiness. NON-sexual, ordinary drowsiness.",
    "wistful_sigh": "a wistful melancholy sigh, exhaling slowly while looking down, a small sad longing breath. NON-sexual, ordinary wistfulness.",
}

LINES: dict[str, list[tuple[str, str]]] = {
    "Swahili": [
        ("relief", "Fuu... ilikuwa karibu sana... ninafurahi imekwisha... fuu..."),
        ("tired_sigh", "Haaah... nimechoka sana... siku ndefu mno... haaah..."),
        ("nervous_breath", "Hh... hh... sawa... pumua kwa kina... ninaweza... hh..."),
        ("sleepy", "Hahmm... usingizi unanijia... hahmm... ninajisikia vizuri..."),
        ("wistful_sigh", "Huuu... lafu yote yangekuwa tofauti... huuu... sawa tu..."),
    ],
    "Arabic": [
        ("relief", "فوف... كان قريبا جدا... أنا سعيدة أن انتهى... فوو..."),
        ("tired_sigh", "هاااه... أنا متعبة جدا... يا له من يوم طويل... هاااه..."),
        ("nervous_breath", "هه... هه... حسنا... تنفس بعمق... أستطيع فعلها... هه..."),
        ("sleepy", "هاهمم... أصابني النعاس... هاهمم... ما أدفأه..."),
        ("wistful_sigh", "هووو... ليت الأمور كانت مختلفة... هووو... لا يهم..."),
    ],
}


def main() -> None:
    parser = testgen.make_parser(
        "VoxCPM2 child (8yo) NON-sexual sigh Swahili/Arabic generation",
        output_dir=OUTPUT_DIR,
        model_dir=MODEL_DIR,
        languages=list(LINES.keys()),
        types=list(TYPE_DESC.keys()),
        seeds=[0],
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
