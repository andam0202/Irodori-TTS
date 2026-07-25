#!/bin/bash
# ============================================================================
# tenchan（超てんちゃん本人）口調テスト生成 — v1 vs v2 比較
# てんしラジオ本人のキャッチフレーズ・口調を再現できているか、
# また v2（ゲスト声疑いセグメント50件除外）で音質・声質が改善したかを
# 同一テキスト・同一参照音声で v1/v2 双方生成して聴き比べる。
# 日付: 2026-07-02
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CKPT_V1="${PROJECT_DIR}/data/lora/tenchan_v1/tenchan_v1_best.safetensors"
CKPT_V2="${PROJECT_DIR}/data/lora/tenchan_v2/tenchan_v2_best.safetensors"
REF_WAV="${PROJECT_DIR}/data/tenchan_v2/wavs/seg_00012.wav"
OUTPUT_DIR="${PROJECT_DIR}/data/output/tenchan_persona_test"
SEED=42

mkdir -p "$OUTPUT_DIR"

CAPTION="インターネットラジオの女性パーソナリティ。自分を『インターネットエンジェル』と自称し、\
明るくテンション高く早口で喋る。馴れ馴れしく親しみやすいノリで、ときどき毒舌や自虐、\
メタ的なツッコミを挟む。カジュアルな話し言葉。"

run_both() {
    local text="$1" name="$2"
    for v in v1 v2; do
        local ckpt_var="CKPT_${v^^}"
        local ckpt="${!ckpt_var}"
        echo "[generate] ${v}_${name}"
        uv run python "${PROJECT_DIR}/infer.py" \
            --checkpoint "$ckpt" \
            --text "$text" \
            --caption "$CAPTION" \
            --ref-wav "$REF_WAV" \
            --output-wav "${OUTPUT_DIR}/${v}_${name}.wav" \
            --seed "$SEED" \
            --tail-fade-ms 120 \
            --tail-pad-out-ms 250 \
            2>&1 | tail -1
    done
}

echo "===== 超てんちゃん 口調テスト（v1/v2比較）====="

# 1: 定番の名乗り
run_both "ジェルバンワー！ラジオパーソナリティエンジェル、頂点ちゃんだよ！" "01_名乗り"

# 2: 話を仕切り直す口癖
run_both "気を取り直して、今日も張り切ってお手紙読んでいくよー" "02_仕切り直し"

# 3: メタ的な自虐・ツッコミ
run_both "えー、それ本当に言ってる？　……まあ、ネットの民ならしょうがないか" "03_自虐ツッコミ"

# 4: 相方（ピエンネコ）への軽い毒舌
run_both "ピエンネコ、そういうとこだぞ" "04_相方への毒舌"

# 5: 実際の台詞そのまま（学習データ内テキストの再現度チェック）
run_both "スポンサーも募集してるぞ！お金くれたらね、お金は大切にしましょう！" "05_実台詞再現"

# 6: 締めの挨拶
run_both "今日はここまで。また来週、インターネットで会おうね" "06_締め挨拶"

echo "===== 全テスト完了 ====="
echo "出力先: ${OUTPUT_DIR}/ （v1_*.wav と v2_*.wav を聴き比べてください）"
ls -1 "${OUTPUT_DIR}/"*.wav | wc -l | xargs echo "生成ファイル数:"
