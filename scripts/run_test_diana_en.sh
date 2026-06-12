#!/bin/bash
# ══════════════════════════════════════════════════════════
#  英語版 Diana — F5-TTS ゼロショット音声クローン テスト生成
#
#  Irodori-TTS は日本語特化のため、英語は F5-TTS を使う（プロジェクト方針）。
#  Diana(PRAGMATA, 少女型アンドロイド) の英語音声から抽出したクリーンな参照
#  クリップを用い、学習なしのゼロショットで複数文を生成する。
#
#  前提:
#   - 参照クリップは scripts/find_diana_clip.py で BS-RoFormer 分離後の
#     ボーカルから F0(高音=Diana) で抽出・Whisper で書き起こし済み
#   - F5環境は tools/f5-tts/（torch 2.11 + torchcodec 0.11.1+cu128 + nvidia-npp）
#
#  使い方: bash scripts/run_test_diana_en.sh
# ══════════════════════════════════════════════════════════
set -e
cd "$(dirname "$0")/.."

REF_DIR="data/output/diana_en/ref"
OUT_DIR="data/output/diana_en_test"

# 参照クリップ A: クリーン・4.2秒・F0 299Hz（★ユーザー採用・デフォルト推奨）
REF_A="${REF_DIR}/diana_ref_a.wav"
REF_A_TEXT="You may forget who I am, but I will always remember you."
# 参照クリップ B: アンドロイド自己紹介・9.4秒・F0 319Hz（比較用）
REF_B="${REF_DIR}/diana_ref_b.wav"
REF_B_TEXT="I am DI-03367, a state-of-the-art pragmatic created here at the Cradle. I possess basic life-saving protocols."

echo "===== ref_a（クリーン4.2秒）で生成 ====="
PYTHONUNBUFFERED=1 bash scripts/f5tts.sh python scripts/f5_batch_infer.py \
  --ref-audio "$REF_A" --ref-text "$REF_A_TEXT" --out-dir "$OUT_DIR" --tag a

echo "===== ref_b（自己紹介9.4秒）で生成（比較用）====="
PYTHONUNBUFFERED=1 bash scripts/f5tts.sh python scripts/f5_batch_infer.py \
  --ref-audio "$REF_B" --ref-text "$REF_B_TEXT" --out-dir "$OUT_DIR" --tag b

echo "===== 完了: ${OUT_DIR} ====="
ls -1 "${OUT_DIR}"/*.wav
