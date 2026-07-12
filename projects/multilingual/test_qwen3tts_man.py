"""Qwen3-TTS 中年男性ボイス（命令 / 性行為）多言語生成（EN/RU/ZH/KO/ES/PT）.

ゲーム制作用。中年男性（約45歳）の:
  - command       : 強い口調の命令（軍人/ボス/権力者）
  - command_threat: 脅しを含む命令
  - sex_moan      : 性行為中の喘ぎ声（NSFW・成人男性）
  - sex_climax    : クライマックスの声（NSFW・成人男性）
の4タイプを VoiceDesign で生成する。**成人男性限定**（未成年は一切扱わない）。

bash scripts/qwen3tts.sh python projects/multilingual/test_qwen3tts_man.py
bash scripts/qwen3tts.sh python projects/multilingual/test_qwen3tts_man.py --languages English --types command
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _tts_testgen as testgen  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_DIR / "tools" / "qwen3-tts" / "models" / "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
OUTPUT_DIR = PROJECT_DIR / "data" / "output" / "qwen3tts_man"

LANG_CODE = {"English": "en", "Russian": "ru", "Chinese": "zh", "Korean": "ko",
             "Spanish": "es", "Portuguese": "pt"}

# 言語ごとのライン: (タイプ, テキスト)
LINES: dict[str, list[tuple[str, str]]] = {
    "English": [
        ("command", "Stop right there! Don't move! You do exactly as I say, now!"),
        ("command_threat", "On your knees! Defy me again and you will regret it!"),
        ("sex_moan", "Mmm... yeah... that's it... ah... don't stop... hmm..."),
        ("sex_climax", "Ah—! I'm—! Hah... yes... ahh...!"),
    ],
    "Russian": [
        ("command", "Стоять! Не двигаться! Будешь делать всё в точности как я скажу, сейчас!"),
        ("command_threat", "На колени! Ослушаешься ещё раз — пожалеешь!"),
        ("sex_moan", "Ммм... да... вот так... ах... не останавливайся... хм..."),
        ("sex_climax", "Ах—! Я—! Ха... да... а-ах...!"),
    ],
    "Chinese": [
        ("command", "站住！不许动！给我老老实实照我说的做，马上！"),
        ("command_threat", "跪下！再敢违抗，你会后悔的！"),
        ("sex_moan", "嗯……对……就是那样……啊……别停……嗯……"),
        ("sex_climax", "啊——！我——！哈……就是……啊啊……！"),
    ],
    "Korean": [
        ("command", "거기 멈춰! 움직이지 마! 내가 시키는 대로 정확히 해, 당장!"),
        ("command_threat", "무릎 꿇어! 다시 거역하면 뼈저리게 후회할 줄 알아!"),
        ("sex_moan", "음... 그래... 그렇게... 아... 멈추지 마... 흠..."),
        ("sex_climax", "아—! 나—! 하아... 그래... 아아...!"),
    ],
    "Spanish": [
        ("command", "¡Alto ahí! ¡No te muevas! ¡Harás exactamente lo que yo diga, ahora mismo!"),
        ("command_threat", "¡De rodillas! ¡Vuelve a desafiarme y te arrepentirás!"),
        ("sex_moan", "Mmm... sí... así... ah... no pares... hmm..."),
        ("sex_climax", "¡Ah—! ¡Yo—! Jah... sí... aah...!"),
    ],
    "Portuguese": [
        ("command", "Pare aí! Não se mova! Você fará exatamente o que eu disser, agora!"),
        ("command_threat", "De joelhos! Desafie-me de novo e você vai se arrepender!"),
        ("sex_moan", "Mmm... é isso... assim... ah... não para... hmm..."),
        ("sex_climax", "Ah—! Eu—! Hah... sim... aah...!"),
    ],
}

# タイプごとのボイス記述（中年男性・約45歳）
INSTRUCTS: dict[str, str] = {
    "command": (
        "A stern, authoritative middle-aged man around 45 years old barking a sharp command. "
        "A deep, powerful, commanding voice, loud and intimidating, like a military officer "
        "or a crime boss. Speaking forcefully with no warmth."
    ),
    "command_threat": (
        "A menacing middle-aged man around 45 years old delivering a threatening command. "
        "A deep, low, dangerous voice, cold and intimidating, with barely contained "
        "aggression. Like a ruthless warlord or syndicate boss."
    ),
    "sex_moan": (
        "A middle-aged man around 45 years old moaning in pleasure during intimacy. "
        "A deep, breathless, husky voice, low groans and heavy breathing, clearly aroused. "
        "Adult mature male voice, NON-violent, consensual intimacy."
    ),
    "sex_climax": (
        "A middle-aged man around 45 years old at the peak of pleasure during intimacy. "
        "Intense deep groaning, breathless and overwhelmed, a low rough voice crying out. "
        "Adult mature male voice at climax, NON-violent, consensual intimacy."
    ),
}


def main() -> None:
    parser = testgen.make_parser(
        "Qwen3-TTS middle-aged man (command / sex) multilingual generation",
        output_dir=OUTPUT_DIR,
        model_dir=MODEL_DIR,
        languages=list(LINES.keys()),
        types=list(INSTRUCTS.keys()),
        seeds=[0, 1],
        engine="qwen",
    )
    args = parser.parse_args()
    specs = testgen.build_specs(
        LINES,
        LANG_CODE,
        lambda language, typ: INSTRUCTS[typ],
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
