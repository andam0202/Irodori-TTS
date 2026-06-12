#!/bin/bash
# ══════════════════════════════════════════════════════════
#  F5-TTS ラッパースクリプト（英語TTS用・隔離環境呼び出し）
#
#  Irodori-TTS は日本語特化のため、英語は F5-TTS を使う。
#  環境は tools/f5-tts/ に uv で隔離されている（torch cu128）。
#
#  使い方:
#    bash scripts/f5tts.sh infer --ref_audio ref.wav --ref_text "..." --gen_text "..." -o out_dir
#    bash scripts/f5tts.sh finetune <finetune-cli args...>
#    bash scripts/f5tts.sh prepare <wavs+metadata.csvのdir> <出力dir>   # データセット変換
#    bash scripts/f5tts.sh python <args...>                             # 環境内のpython直接実行
# ══════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
F5_PROJECT="${PROJECT_DIR}/tools/f5-tts"

# torchcodec の core ライブラリ(libtorchcodec_core6.so 等)は NPP/cuDNN 等の nvidia 共有ライブラリに
# runpath 無しでリンクするため、放置するとシステムの古い NPP(libnppicc.so.12 12.0.x)を拾い
# `undefined symbol: nppiNV12ToRGB_...` で落ちる。venv 内の nvidia/*/lib を最優先にして回避する。
NV_LIB_DIRS="$(ls -d "${F5_PROJECT}"/.venv/lib/python*/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')"
if [ -n "${NV_LIB_DIRS}" ]; then
    export LD_LIBRARY_PATH="${NV_LIB_DIRS}${LD_LIBRARY_PATH:-}"
fi

CMD="$1"
shift || true

case "$CMD" in
    infer)
        exec uv run --project "$F5_PROJECT" f5-tts_infer-cli "$@"
        ;;
    finetune)
        exec uv run --project "$F5_PROJECT" f5-tts_finetune-cli "$@"
        ;;
    prepare)
        # metadata.csv（audio_file|text 形式）+ wavs → F5-TTS arrow データセット
        exec uv run --project "$F5_PROJECT" python -m f5_tts.train.datasets.prepare_csv_wavs "$@"
        ;;
    python)
        exec uv run --project "$F5_PROJECT" python "$@"
        ;;
    *)
        echo "使い方: bash scripts/f5tts.sh {infer|finetune|prepare|python} <args...>"
        echo "  infer    : f5-tts_infer-cli を実行"
        echo "  finetune : f5-tts_finetune-cli を実行"
        echo "  prepare  : CSV+WAV を F5-TTS データセット形式に変換"
        echo "  python   : F5-TTS 環境の python を直接実行"
        exit 1
        ;;
esac
