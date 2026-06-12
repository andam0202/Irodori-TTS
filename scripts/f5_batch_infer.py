"""F5-TTS バッチ推論（モデルを1回だけロードして複数文を生成）。

英語版 Diana のゼロショット音声クローン用。tools/f5-tts 隔離環境で実行する:
    bash scripts/f5tts.sh python scripts/f5_batch_infer.py \
        --ref-audio <ref.wav> --ref-text "..." --out-dir <dir> [--tag a]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from f5_tts.api import F5TTS

# Diana（少女型アンドロイド・SFW）の英語テスト文。感情・世界観のバリエーション。
SENTENCES: list[tuple[str, str]] = [
    # グループ1: ベースライン
    ("g1_greeting", "Hello! I'm so happy to see you again. Let's explore the moon together."),
    ("g1_calm", "The data analysis is complete. Everything appears to be functioning normally."),
    # グループ2: 感情バリエーション
    ("g2_worried", "Hugh, please be careful. The enemy is much closer than we thought."),
    ("g2_happy", "We did it! I knew we could do this together. Thank you for trusting me."),
    ("g2_sad", "Sometimes I wonder if I'm truly alive, or just pretending to be."),
    # グループ3: 世界観・アンドロイドらしさ
    ("g3_intro", "I am a Pragmata, designed to assist you on this mission. Please rely on me."),
    ("g3_curious", "Look at all those stars. Seeing them with you feels different somehow."),
    # グループ4: 長め
    ("g4_long", "If we can reach the core before the storm hits, we might still have a chance. "
                "I'll calculate the safest route for us."),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-audio", required=True)
    ap.add_argument("--ref-text", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="a", help="出力ファイル名の接尾辞（参照クリップ識別用）")
    ap.add_argument("--model", default="F5TTS_v1_Base")
    ap.add_argument("--ckpt-file", default="", help="fine-tune済みチェックポイント(.pt/.safetensors)。省略時はベース")
    ap.add_argument("--vocab-file", default="", help="vocab.txt（fine-tune時のもの）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nfe-step", type=int, default=32)
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tts = F5TTS(model=args.model, ckpt_file=args.ckpt_file, vocab_file=args.vocab_file)
    for name, text in SENTENCES:
        wav_path = out / f"{name}_ref{args.tag}.wav"
        tts.infer(
            ref_file=args.ref_audio,
            ref_text=args.ref_text,
            gen_text=text,
            file_wave=str(wav_path),
            seed=args.seed,
            nfe_step=args.nfe_step,
            speed=args.speed,
        )
        print(f"[done] {wav_path.name}: {text}")


if __name__ == "__main__":
    main()
