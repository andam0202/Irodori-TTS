"""Qwen3-TTS 子供ボイスの「個性」検証（同言語・同台詞で性格違いを作る）.

同じ8歳女児でも instruct（性格・声質記述）を変えると別キャラになることを確認する。
同一言語・同一台詞に対して複数ペルソナ × 複数シードを生成し、
「instruct による性格差」と「seed による個体差」の両方を聴き比べられるようにする。

bash scripts/qwen3tts.sh python scripts/test_qwen3tts_child_persona.py --language English
bash scripts/qwen3tts.sh python scripts/test_qwen3tts_child_persona.py --language Korean --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _tts_testgen as testgen  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "tools" / "qwen3-tts" / "models" / "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
OUTPUT_DIR = PROJECT_DIR / "data" / "output" / "qwen3tts_child_persona"

LANG_CODE = {"English": "en", "Russian": "ru", "Chinese": "zh", "Korean": "ko"}

# 同一台詞（挨拶＋自己紹介っぽい内容）を言語ごとに用意
LINE: dict[str, str] = {
    "English": "Hi there! I'm so happy to meet you! Let's be friends, okay? Come on, let's go play!",
    "Russian": "Привет! Я так рада тебя видеть! Давай дружить, хорошо? Пойдём скорее играть!",
    "Chinese": "你好呀！见到你我好开心！我们做朋友吧，好不好？走啦走啦，一起去玩！",
    "Korean": "안녕! 만나서 정말 반가워! 우리 친구 하자, 응? 어서 같이 놀러 가자!",
}

# 同じ「8歳女児」でも性格・声質を描き分けた5ペルソナ
PERSONAS: dict[str, str] = {
    "genki": (
        "A cheerful, energetic 8-year-old tomboy girl, loud and lively, high-pitched, "
        "speaking fast with big excitement, full of energy."
    ),
    "shy": (
        "A shy, timid 8-year-old girl, soft-spoken and quiet, hesitant and gentle, "
        "a small delicate high voice, a little nervous."
    ),
    "cool": (
        "A precocious, calm 8-year-old girl, speaking in a composed, matter-of-fact way, "
        "a clear cool voice, mature for her age but still childlike."
    ),
    "ojou": (
        "A pampered little rich 8-year-old girl, slightly haughty but cute, elegant and "
        "sing-song, a bright princess-like high voice, self-assured."
    ),
    "dreamy": (
        "A dreamy, airheaded 8-year-old girl, slow and gentle, relaxed and soft, "
        "a warm fluffy high voice, speaking languidly as if half daydreaming."
    ),
    # --- 陰気系ペルソナ（ホラー/シリアス寄りの子供NPC向け） ---
    "gloomy": (
        "A gloomy, withdrawn 8-year-old girl, speaking in a low, listless, "
        "depressed mumble, quiet and joyless, dragging her words tiredly."
    ),
    "creepy": (
        "An eerie, unsettling 8-year-old girl, speaking in a flat, hushed, "
        "sing-song whisper with no warmth, calm but deeply unnerving, like a ghost child."
    ),
    "sulky": (
        "A sulky, resentful 8-year-old girl, pouting and grumbling, muttering "
        "under her breath in a sullen, displeased tone."
    ),
    "melancholic": (
        "A lonely, melancholic 8-year-old girl on the verge of tears, a fragile "
        "trembling high voice, sad and quiet, speaking slowly with heavy sighs."
    ),
    "empty": (
        "An emotionless 8-year-old girl, speaking in a hollow, monotone voice with "
        "no inflection at all, detached and vacant, unsettlingly blank."
    ),
    # --- 低い声の8歳児（成人域まで下げず、あくまで子供のまま低め） ---
    "low_husky": (
        "An 8-year-old girl with an unusually low, husky, raspy voice for a child, "
        "deep and throaty but still clearly a little kid, calm and quiet."
    ),
    "low_calm": (
        "A little 8-year-old girl speaking in a low, deep, mellow tone, soft and "
        "composed, a dark low-pitched child's voice, gentle and grounded."
    ),
    "low_boyish": (
        "An 8-year-old tomboy with a low, boyish voice, husky and a bit gruff for "
        "a child, blunt and cool, low-pitched but still childlike."
    ),
    # --- 「低めだが、はっきり女の子」— boyish/男児化を避けた版 ---
    "low_girl": (
        "A little 8-year-old GIRL with a low, mellow voice, distinctly feminine and "
        "sweet, clearly a young girl (never a boy), soft and calm, gently low-pitched."
    ),
    "husky_girl": (
        "A cute little 8-year-old GIRL with a slightly husky, breathy voice, still "
        "clearly feminine and girlish, warm and soft, a little lower than usual but "
        "unmistakably a young girl, not a boy."
    ),
    "calm_girl_low": (
        "A calm, composed 8-year-old GIRL speaking in a soft low register, feminine "
        "and delicate, a gentle mature-sounding little girl's voice, definitely a girl."
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-TTS child persona variety test")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    parser.add_argument("--language", default="English", choices=list(LINE.keys()))
    parser.add_argument("--personas", nargs="+", default=list(PERSONAS.keys()), choices=list(PERSONAS.keys()))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--dry-run", action="store_true", help="生成せず GenSpec 一覧を表示して終了"
    )
    args = parser.parse_args()

    lang_code = LANG_CODE[args.language]
    text = LINE[args.language]
    specs = [
        testgen.GenSpec(
            filename=f"{lang_code}_{persona}_s{seed}.wav",
            language=args.language,
            type_key=persona,
            text=text,
            instruct=PERSONAS[persona],
            seed=seed,
            log=persona,
        )
        for persona in args.personas
        for seed in args.seeds
    ]
    testgen.run_qwen_design(
        specs,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        temperature=args.temperature,
        top_p=args.top_p,
        type_field="persona",
        manifest_name=f"manifest_{lang_code}.json",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
