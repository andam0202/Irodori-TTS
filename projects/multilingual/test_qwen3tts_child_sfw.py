"""Qwen3-TTS 子供ボイス 多言語サンプル生成（EN/RU/ZH/KO）.

VoiceDesign モデルで4言語生成する。挨拶・はしゃぎ・好奇心・呼びかけの4タイプ。

bash scripts/qwen3tts.sh python projects/multilingual/test_qwen3tts_child_sfw.py
bash scripts/qwen3tts.sh python projects/multilingual/test_qwen3tts_child_sfw.py --languages English --seeds 0 1 2
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _tts_testgen as testgen  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_DIR / "tools" / "qwen3-tts" / "models" / "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
OUTPUT_DIR = PROJECT_DIR / "data" / "output" / "qwen3tts_child_test"

LANG_CODE = {"English": "en", "Russian": "ru", "Chinese": "zh", "Korean": "ko"}

# 言語ごとのテストライン: (タイプ, テキスト)
LINES: dict[str, list[tuple[str, str]]] = {
    "English": [
        ("greeting", "Hi! Good morning! Did you sleep well? Let's play together today, okay?"),
        ("excited", "Wow! Look, look! It's so pretty! I've never seen anything like this before!"),
        ("curious", "Hey, what's this? Can I touch it? What does it do? Tell me, tell me!"),
        ("calling", "Wait for me! Don't leave me behind! I'm coming too, wait up!"),
    ],
    "Russian": [
        ("greeting", "Привет! Доброе утро! Хорошо спалось? Давай сегодня поиграем вместе!"),
        ("excited", "Ух ты! Смотри, смотри! Как красиво! Я такого никогда не видела!"),
        ("curious", "Эй, а что это? Можно потрогать? А что оно делает? Расскажи, расскажи!"),
        ("calling", "Подожди меня! Не уходи без меня! Я тоже иду, подожди!"),
    ],
    "Chinese": [
        ("greeting", "你好！早上好！睡得好吗？今天一起玩吧，好不好？"),
        ("excited", "哇！你看你看！好漂亮啊！我从来没见过这样的东西！"),
        ("curious", "欸，这是什么呀？可以摸一摸吗？它是做什么用的？告诉我嘛，告诉我嘛！"),
        ("calling", "等等我呀！不要丢下我！我也要去，等一等！"),
    ],
    "Korean": [
        ("greeting", "안녕! 좋은 아침이야! 잘 잤어? 오늘 같이 놀자, 응?"),
        ("excited", "와! 봐봐! 너무 예쁘다! 이런 거 한 번도 본 적 없어!"),
        ("curious", "어, 이게 뭐야? 만져봐도 돼? 이거 뭐 하는 거야? 알려줘, 알려줘!"),
        ("calling", "기다려 줘! 나 두고 가지 마! 나도 갈래, 같이 가!"),
    ],
}

# タイプごとのボイス記述（VoiceDesign instruct）
INSTRUCTS: dict[str, str] = {
    "greeting": (
        "A cheerful little 8-year-old girl greeting a friend, bright, innocent, "
        "warm and friendly high-pitched child's voice."
    ),
    "excited": (
        "A little 8-year-old girl, extremely excited and amazed, high-pitched, "
        "bouncy, energetic and joyful child's voice."
    ),
    "curious": (
        "A little 8-year-old girl, curious and inquisitive, eagerly asking lots "
        "of questions, innocent and playful child's voice."
    ),
    "calling": (
        "A little 8-year-old girl calling out while running, a bit worried about "
        "being left behind, energetic and slightly breathless child's voice."
    ),
}


def main() -> None:
    parser = testgen.make_parser(
        "Qwen3-TTS child (8yo) SFW multilingual test generation",
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
