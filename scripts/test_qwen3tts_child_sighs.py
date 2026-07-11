"""Qwen3-TTS 子供ボイス「吐息」多言語生成（EN/RU/ZH/KO/ES/PT）.

ゲーム制作用。8歳女児の **非性的な吐息・息遣い** 5種を、言語別の声質特徴付きで生成:
  - RU / ZH : 低い声
  - ES      : ダウナー系
  - EN/KO/PT: 標準
5タイプ: relief(安堵) / tired_sigh(疲れ) / nervous_breath(緊張) / sleepy(眠気) / wistful_sigh(物憂げ)
いずれも NON-sexual。

bash scripts/qwen3tts.sh python scripts/test_qwen3tts_child_sighs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _tts_testgen as testgen  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "tools" / "qwen3-tts" / "models" / "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
OUTPUT_DIR = PROJECT_DIR / "data" / "output" / "qwen3tts_child_sighs"

LANG_CODE = {"English": "en", "Russian": "ru", "Chinese": "zh", "Korean": "ko",
             "Spanish": "es", "Portuguese": "pt"}

VOICE_BASE: dict[str, str] = {
    "English": "A little 8-year-old girl",
    "Russian": "A little 8-year-old girl with a noticeably low, deep, husky voice for a child",
    "Chinese": "A little 8-year-old girl with a low, deep voice for a child",
    "Korean": "A little 8-year-old girl",
    "Spanish": "A little 8-year-old girl with a downer, listless, low-energy, flat and gloomy personality",
    "Portuguese": "A little 8-year-old girl",
}

TYPE_DESC: dict[str, str] = {
    "relief": "letting out a big relieved sigh after tension, exhaling long and slow, a soft breath of relief. This is NON-sexual: ordinary relief.",
    "tired_sigh": "a heavy tired sigh after a long day, exhaling deeply with exhaustion, slightly deflated. This is NON-sexual: ordinary fatigue.",
    "nervous_breath": "taking short nervous breaths before something scary or important, shaky shallow breathing, anxious. This is NON-sexual: ordinary nervousness.",
    "sleepy": "a drowsy sleepy sigh, yawning softly, breathing slow and heavy with sleepiness. This is NON-sexual: ordinary drowsiness.",
    "wistful_sigh": "a wistful melancholy sigh, exhaling slowly while looking down, a small sad longing breath. This is NON-sexual: ordinary wistfulness.",
}

LINES: dict[str, list[tuple[str, str]]] = {
    "English": [
        ("relief", "Phew... that was close... I'm so glad it's over... fuu..."),
        ("tired_sigh", "Haaah... I'm so tired... what a long day... haaah..."),
        ("nervous_breath", "Hh... hh... okay... deep breath... I can do this... hh..."),
        ("sleepy", "Hahhmm... I'm getting sleepy... hahhmm... so warm..."),
        ("wistful_sigh", "Huuu... I wish things were different... huuu... never mind..."),
    ],
    "Russian": [
        ("relief", "Фух... это было близко... я так рада, что всё закончилось... фуу..."),
        ("tired_sigh", "Хааа... я так устала... какой длинный день... хааа..."),
        ("nervous_breath", "Хх... хх... ладно... глубокий вдох... я смогу... хх..."),
        ("sleepy", "Хаххм... меня клонит в сон... хаххм... так тепло..."),
        ("wistful_sigh", "Хууу... вот бы всё было иначе... хууу... неважно..."),
    ],
    "Chinese": [
        ("relief", "呼……好险……太好了，终于结束了……呼……"),
        ("tired_sigh", "哈啊……好累啊……今天好漫长……哈啊……"),
        ("nervous_breath", "呼……呼……好……深呼吸……我可以的……呼……"),
        ("sleepy", "哈嗯……好困啊……哈嗯……好暖和……"),
        ("wistful_sigh", "呼……要是能不一样就好了……呼……算了……"),
    ],
    "Korean": [
        ("relief", "후우... 위험했어... 다 끝나서 다행이야... 후우..."),
        ("tired_sigh", "하아아... 너무 피곤해... 오늘 정말 길었어... 하아아..."),
        ("nervous_breath", "후... 후... 괜찮아... 심호흡... 할 수 있어... 후..."),
        ("sleepy", "하암... 졸려오고 있어... 하암... 따뜻해..."),
        ("wistful_sigh", "후우... 다르면 좋겠어... 후우... 됐어..."),
    ],
    "Spanish": [
        ("relief", "Fiuu... eso estuvo cerca... qué alivio que terminó... fuu..."),
        ("tired_sigh", "Haaah... estoy tan cansada... qué día tan largo... haaah..."),
        ("nervous_breath", "Hh... hh... okay... respira hondo... puedo hacerlo... hh..."),
        ("sleepy", "Hahhmm... me está dando sueño... hahhmm... qué calorcito..."),
        ("wistful_sigh", "Huuu... ojalá todo fuera distinto... huuu... no importa..."),
    ],
    "Portuguese": [
        ("relief", "Fiuu... essa foi por pouco... que alívio que acabou... fuu..."),
        ("tired_sigh", "Haaah... estou tão cansada... que dia longo... haaah..."),
        ("nervous_breath", "Hh... hh... ok... respira fundo... eu consigo... hh..."),
        ("sleepy", "Hahhmm... estou com sono... hahhmm... que quentinho..."),
        ("wistful_sigh", "Huuu... quem tudo fosse diferente... huuu... deixa pra lá..."),
    ],
}


def main() -> None:
    parser = testgen.make_parser(
        "Qwen3-TTS child (8yo) NON-sexual sigh multilingual generation",
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
