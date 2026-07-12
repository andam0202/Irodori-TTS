"""VoxCPM2 子供ボイス「非性的うめき声・非言語音」Swahili/Arabic 生成.

Qwen3-TTS が対応しない Swahili(東アフリカ)・Arabic(北アフリカ) の子ども(8歳女児)
音声を、VoxCPM2 の Voice Design で生成する。言語タグ不要（テキストから自動判定）。
9タイプ（fear/pain/grief/exhaustion/dread/cold/cough/wordless_groan/slow_sad）、
いずれも NON-sexual。

bash scripts/voxcpm.sh python projects/multilingual/test_voxcpm_child_groans.py
bash scripts/voxcpm.sh python projects/multilingual/test_voxcpm_child_groans.py --languages Swahili --types cough --seeds 0
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _tts_testgen as testgen  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_DIR / "tools" / "voxcpm" / "models" / "VoxCPM2"
OUTPUT_DIR = PROJECT_DIR / "data" / "output" / "voxcpm_child_groans"

LANG_CODE = {"Swahili": "sw", "Arabic": "ar"}

VOICE_BASE: dict[str, str] = {
    "Swahili": "A little 8-year-old girl",
    "Arabic": "A little 8-year-old girl",
}

TYPE_DESC: dict[str, str] = {
    "fear": "whimpering in pure fear, trembling, a shaky voice, terrified. NON-sexual, a frightened child.",
    "pain": "crying out softly from a hurt arm, a pained whimper, sniffling. NON-sexual, ordinary physical pain.",
    "grief": "sobbing quietly in sadness, a broken crying voice, sniffling through tears. NON-sexual.",
    "exhaustion": "out of breath after running, panting from exertion, a weak tired voice. NON-sexual.",
    "dread": "with a sinking anxious feeling, a low uneasy groan of worry. NON-sexual.",
    "cold": "shivering hard in the cold, teeth chattering, a shaky voice from the chill. NON-sexual.",
    "cough": "coughing repeatedly, a sick child's dry weak cough, persistent. NON-sexual, non-verbal sound only.",
    "wordless_groan": "wordless groans of discomfort, no articulated words, low pained throat sounds. NON-sexual, non-verbal.",
    "slow_sad": "slow long sad sounds, deep sighing breaths and soft whimpering, no words. NON-sexual, non-verbal.",
}

LINES: dict[str, list[tuple[str, str]]] = {
    "Swahili": [
        ("fear", "Hapana... usikaribe... tafadhali... naogopa sana..."),
        ("pain", "Auw... auw, mkono wangu... unauma sana... auw..."),
        ("grief", "Woo... kwa nini ilikuwa hivi... woo... nina huzuni sana..."),
        ("exhaustion", "Haa... haa... siwezi kukimbia tena... nimechoka sana... haa..."),
        ("dread", "Kuna kitu si sawa... nina hisia mbaya sana..."),
        ("cold", "Brr... kuna baridi sana... siwezi kuacha kutetemeka... brr..."),
        ("cough", "Kohoa... kohoa, kohoa... uhh... kohoa..."),
        ("wordless_groan", "Ngh... ugh... hnnn... urr... ngh..."),
        ("slow_sad", "Haaah... huu... uuh... haaah... huu..."),
    ],
    "Arabic": [
        ("fear", "لا... لا تقترب... من فضلك... أنا خائفة جدا..."),
        ("pain", "آخ... آخ، يدي... يؤلمني كثيرا... آخ..."),
        ("grief", "آه... لماذا حدث هذا... آه... أنا حزينة جدا..."),
        ("exhaustion", "هاا... هاا... لا أستطيع الركض أكثر... أنا متعبة جدا... هاا..."),
        ("dread", "شيء ما ليس صحيحا... لدي شعور سيء جدا..."),
        ("cold", "برر... الجو شديد البرودة... لا أستطيع التوقف عن الارتجاف... برر..."),
        ("cough", "كح... كح كح... أه... كح..."),
        ("wordless_groan", "نغ... أوغ... هنن... أرر... نغ..."),
        ("slow_sad", "هاااه... هوو... أوه... هاااه... هوو..."),
    ],
}


def main() -> None:
    parser = testgen.make_parser(
        "VoxCPM2 child (8yo) NON-sexual groan/non-verbal Swahili/Arabic generation",
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
