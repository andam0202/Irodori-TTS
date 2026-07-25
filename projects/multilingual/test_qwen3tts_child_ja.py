"""Qwen3-TTS 日本の女の子(8歳)ボイス検証：台詞 / 吐息 / うめき声.

Qwen3-TTS の日本語品質をゲーム制作用に試す（Irodori-TTS ではなく Qwen3 で生成）。
8歳女児・11タイプ:
  - 台詞: greeting / excited / curious
  - 吐息: relief / tired_sigh / nervous_breath / sleepy / wistful_sigh
  - うめき: fear / pain / grief
いずれも NON-sexual。

bash scripts/qwen3tts.sh python projects/multilingual/test_qwen3tts_child_ja.py --dry-run
bash scripts/qwen3tts.sh python projects/multilingual/test_qwen3tts_child_ja.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _tts_testgen as testgen  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_DIR / "tools" / "qwen3-tts" / "models" / "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
OUTPUT_DIR = PROJECT_DIR / "data" / "output" / "qwen3tts_child_ja"

LANG_CODE = {"Japanese": "ja"}

VOICE_BASE: dict[str, str] = {
    "Japanese": "A little 8-year-old girl",
}

TYPE_DESC: dict[str, str] = {
    # --- 台詞 ---
    "greeting": "cheerfully greeting a friend, bright, innocent, warm and friendly high-pitched child's voice.",
    "excited": "extremely excited and amazed, high-pitched, bouncy, energetic and joyful child's voice.",
    "curious": "curious and inquisitive, eagerly asking lots of questions, innocent and playful child's voice.",
    # --- 吐息（非性的） ---
    "relief": "letting out a big relieved sigh after tension, exhaling long and slow. This is NON-sexual: ordinary relief.",
    "tired_sigh": "a heavy tired sigh after a long day, exhaling deeply with exhaustion. This is NON-sexual: ordinary fatigue.",
    "nervous_breath": "taking short nervous breaths before something scary, shaky shallow breathing. This is NON-sexual: ordinary nervousness.",
    "sleepy": "a drowsy sleepy sigh, yawning softly, breathing slow and heavy. This is NON-sexual: ordinary drowsiness.",
    "wistful_sigh": "a wistful melancholy sigh, exhaling slowly while looking down. This is NON-sexual: ordinary wistfulness.",
    # --- うめき声（非性的） ---
    "fear": "whimpering in pure fear, trembling, a shaky voice catching her breath, terrified. This is NON-sexual: a frightened child.",
    "pain": "crying out softly from a hurt arm, a pained whimper, sniffling. This is NON-sexual: ordinary physical pain.",
    "grief": "sobbing quietly in sadness, a broken crying voice, sniffling through tears. This is NON-sexual.",
}

LINES: dict[str, list[tuple[str, str]]] = {
    "Japanese": [
        ("greeting", "あっ、こんにちは！あそびにきたよ！いっしょにあそぼう！"),
        ("excited", "わぁ、すごい！みてみて！こんなの、はじめて！"),
        ("curious", "ねえねえ、これなあに？さわってもいい？おしえて！"),
        ("relief", "ふぅ……おわってよかったぁ……ふぅ……"),
        ("tired_sigh", "はぁ……つかれちゃった……きょうは、ながかったなぁ……はぁ……"),
        ("nervous_breath", "ふぅ……ふぅ……だいじょうぶ……しんこきゅう……できるよ……ふぅ……"),
        ("sleepy", "はぁむ……ねむくなってきちゃった……はぁむ……あったかい……"),
        ("wistful_sigh", "はぁ……ちがっていればよかったのに……はぁ……ま、いいか……"),
        ("fear", "ひっ……こないで……おねがい……こわいよ……"),
        ("pain", "いたっ……いたい、うでが……すごくいたいの……いたっ……"),
        ("grief", "うぅ……どうしてこんなことに……うぅ……かなしい……"),
    ],
}


def main() -> None:
    parser = testgen.make_parser(
        "Qwen3-TTS Japanese child (8yo girl) lines/sighs/groans test",
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
