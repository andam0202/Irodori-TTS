#!/bin/bash
# ══════════════════════════════════════════════════════════
#  摩美々 音声生成スクリプト
#
#  使い方:
#    bash scripts/generate_mamimi.sh "テキスト"
#    bash scripts/generate_mamimi.sh "テキスト" output.wav
#
#  オプション:
#    -r <ref.wav>   参照音声を変更（デフォルト: seg_00338.wav）
#    -n             参照音声なしで生成（--no-ref モード）
#    -s <seed>      シードを固定（デフォルト: ランダム）
#    -c <cfg>       CFG スケール（デフォルト: 3.0）
#    -h             ヘルプを表示
# ══════════════════════════════════════════════════════════

set -e

CHECKPOINT="data/lora/mamimi_v5_lora/mamimi_v5_best.safetensors"
REF_WAV="data/mamimi_v5/wavs/seg_00225.wav"
OUTPUT_DIR="outputs/mamimi_samples"
NO_REF=false
SEED_ARG=""
CFG_TEXT="3.0"
CFG_SPEAKER="5.0"

# ── オプション解析 ──────────────────────────────────────
while getopts "r:ns:c:h" opt; do
    case "$opt" in
        r) REF_WAV="$OPTARG" ;;
        n) NO_REF=true ;;
        s) SEED_ARG="--seed $OPTARG" ;;
        c) CFG_TEXT="$OPTARG" ;;
        h)
            sed -n '2,15p' "$0" | sed 's/^#  \?//'
            exit 0
            ;;
        *) exit 1 ;;
    esac
done
shift $((OPTIND - 1))

# ── テキスト引数チェック ────────────────────────────────
if [ -z "$1" ]; then
    echo "エラー: テキストを指定してください。"
    echo "使い方: bash scripts/generate_mamimi.sh \"テキスト\""
    echo "       bash scripts/generate_mamimi.sh -h  (ヘルプ)"
    exit 1
fi

TEXT="$1"

# ── 出力ファイル名 ──────────────────────────────────────
if [ -n "$2" ]; then
    # 拡張子がなければ .wav を付加
    if [[ "$2" != *.wav ]]; then
        OUTPUT_WAV="${OUTPUT_DIR}/${2}.wav"
    else
        OUTPUT_WAV="${OUTPUT_DIR}/${2}"
    fi
else
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_WAV="${OUTPUT_DIR}/mamimi_${TIMESTAMP}.wav"
fi

# ── チェックポイント確認 ────────────────────────────────
if [ ! -f "$CHECKPOINT" ]; then
    echo "エラー: チェックポイントが見つかりません: $CHECKPOINT"
    echo "先に LoRA 学習とチェックポイント変換を完了してください。"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ── 実行情報の表示 ──────────────────────────────────────
echo "════════════════════════════════════════"
echo " 摩美々 音声生成"
echo "════════════════════════════════════════"
echo "テキスト  : $TEXT"
echo "出力先    : $OUTPUT_WAV"
if $NO_REF; then
    echo "参照音声  : なし（--no-ref）"
else
    echo "参照音声  : $REF_WAV"
fi
echo "────────────────────────────────────────"

# ── infer.py 実行 ───────────────────────────────────────
if $NO_REF; then
    uv run python infer.py \
        --checkpoint "$CHECKPOINT" \
        --text "$TEXT" \
        --no-ref \
        --cfg-scale-text "$CFG_TEXT" \
        --output-wav "$OUTPUT_WAV" \
        $SEED_ARG
else
    uv run python infer.py \
        --checkpoint "$CHECKPOINT" \
        --text "$TEXT" \
        --ref-wav "$REF_WAV" \
        --cfg-scale-text "$CFG_TEXT" \
        --cfg-scale-speaker "$CFG_SPEAKER" \
        --output-wav "$OUTPUT_WAV" \
        $SEED_ARG
fi

echo "────────────────────────────────────────"
echo "完了: $OUTPUT_WAV"

# WSL 環境では Windows のエクスプローラーで開く
if grep -qi microsoft /proc/version 2>/dev/null; then
    WIN_PATH=$(wslpath -w "$(realpath "$OUTPUT_WAV")" 2>/dev/null || true)
    if [ -n "$WIN_PATH" ]; then
        echo "Windows パス: $WIN_PATH"
    fi
fi
