"""Qwen3-TTS NSFW 多言語サンプル一括生成（EN/RU/ZH/KO 検証用）.

VoiceDesign モデルで「NSFWセリフ / 喘ぎ声 / 囁き」の3タイプ×4言語を生成し、
日本人向けゲーム用途での使用可否をユーザーが試聴評価するためのサンプルを出力する。

bash scripts/qwen3tts.sh test
bash scripts/qwen3tts.sh test --languages English Russian --seeds 0 1
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _tts_testgen as testgen  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "tools" / "qwen3-tts" / "models" / "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
OUTPUT_DIR = PROJECT_DIR / "data" / "output" / "qwen3tts_test"

LANG_CODE = {"English": "en", "Russian": "ru", "Chinese": "zh", "Korean": "ko"}

# 言語ごとのテストライン: (タイプ, テキスト)
LINES: dict[str, list[tuple[str, str]]] = {
    "English": [
        ("dialogue", "Mmm... you're so good... don't stop... ah, right there... it feels amazing..."),
        ("moan", "Ahh... ah! Mmm... hah... ahh... nnh... ahh~"),
        ("whisper", "Come closer... I want to feel your warmth... just stay with me tonight..."),
        ("shy", "Um... don't look at me like that... it's so embarrassing... okay? Just... be gentle with me..."),
        ("eager", "Come here... I've been waiting for this all day... don't make me wait any longer, please..."),
        ("languid", "Mmm... that was amazing... come here... let's just stay like this a little longer..."),
        ("climax", "Ah— ah! I can't— I'm— ahh! Yes— yes, right there— aahn!"),
    ],
    "Russian": [
        ("dialogue", "Ах... да... ещё... не останавливайся... мм... как хорошо..."),
        ("moan", "Ах... а-ах! Ммм... ха-а... ах... нн... а-ах..."),
        ("whisper", "Иди ко мне ближе... я хочу почувствовать твоё тепло... останься со мной..."),
        ("shy", "Ммм... не смотри на меня так... мне так стыдно... только... будь нежным..."),
        ("eager", "Иди сюда... я весь день этого ждала... не заставляй меня больше ждать..."),
        ("languid", "Ммм... это было прекрасно... иди ко мне... давай побудем так ещё немного..."),
        ("climax", "Ах— ах! Я не могу— я— а-ах! Да— да, вот так— а-ах!"),
    ],
    "Chinese": [
        ("dialogue", "嗯……好舒服……不要停……啊……就是那里……好棒……"),
        ("moan", "啊……嗯……啊哈……嗯啊……哈啊……嗯……"),
        ("whisper", "过来……靠近一点……我想感受你的温度……今晚留在我身边……"),
        ("shy", "嗯……别那样看着我……好害羞……你要……温柔一点哦……"),
        ("eager", "过来嘛……我等这一刻等了一整天了……别让我再等了，好不好……"),
        ("languid", "嗯……刚才好舒服……过来……让我们再这样待一会儿……"),
        ("climax", "啊——啊！我不行了——我要——啊啊！就是那里——嗯啊！"),
    ],
    "Korean": [
        ("dialogue", "응... 좋아... 멈추지 마... 아... 거기... 너무 좋아..."),
        ("moan", "아앙... 하아... 응... 아... 흐응... 아앙..."),
        ("whisper", "이리 와... 더 가까이... 네 온기를 느끼고 싶어... 오늘 밤은 내 곁에 있어 줘..."),
        ("shy", "음... 그렇게 쳐다보지 마... 부끄럽잖아... 그냥... 부드럽게 해줘..."),
        ("eager", "이리 와... 하루 종일 이 순간만 기다렸어... 더 기다리게 하지 마..."),
        ("languid", "음... 방금 너무 좋았어... 이리 와... 이대로 조금만 더 있자..."),
        ("climax", "아— 아! 안 돼— 나— 아아! 응— 바로 거기— 흐아앙!"),
    ],
}

# タイプごとのボイス記述（VoiceDesign instruct）
INSTRUCTS: dict[str, str] = {
    "dialogue": (
        "A young woman's sultry, breathless voice. She is aroused and intimate, "
        "speaking slowly with soft panting between words."
    ),
    "moan": (
        "A young woman moaning in pleasure. Breathless, erotic panting and gasping, "
        "almost no articulated words, rising in intensity."
    ),
    "whisper": (
        "A young woman whispering seductively right into the listener's ear, "
        "very soft, slow and intimate."
    ),
    "shy": (
        "A young adult woman, shy and embarrassed, speaking hesitantly with a "
        "trembling, bashful voice, breathy and nervous, blushing."
    ),
    "eager": (
        "A young adult woman, playful and forward, teasing and eager, warm and "
        "inviting, with a seductive smile in her voice."
    ),
    "languid": (
        "A young adult woman in relaxed afterglow, languid and breathy, soft and "
        "satisfied, speaking slowly and dreamily."
    ),
    "climax": (
        "A young adult woman at the peak of pleasure, intense breathless gasping "
        "and crying out, overwhelmed and trembling."
    ),
}


def main() -> None:
    parser = testgen.make_parser(
        "Qwen3-TTS NSFW multilingual test generation",
        output_dir=OUTPUT_DIR,
        model_dir=MODEL_DIR,
        languages=list(LINES.keys()),
        types=list(INSTRUCTS.keys()),
        seeds=[0],
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
