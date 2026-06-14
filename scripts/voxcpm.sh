#!/bin/bash
# ══════════════════════════════════════════════════════════
#  VoxCPM2 ラッパースクリプト（多言語TTS / ボイスデザイン / クローン）
#
#  Irodori-TTS の対抗馬である VoxCPM2（OpenBMB, 2B, 30言語, tokenizer-free）を
#  呼び出す。環境は tools/voxcpm/ に uv で隔離（VoxCPM リポジトリを editable
#  install、torch cu128）。f5-tts と同じ LD_LIBRARY_PATH 回避策付き。
#
#  使い方:
#    bash scripts/voxcpm.sh test [args...]                              # mamimi 全機能検証スクリプト
#    bash scripts/voxcpm.sh design --text "..." --output out.wav        # Voice Design（参照音声不要）
#    bash scripts/voxcpm.sh clone --text "..." --reference-audio r.wav --output out.wav   # クローン
#    bash scripts/voxcpm.sh clone --text "..." --prompt-audio r.wav --prompt-text "..." --reference-audio r.wav --output out.wav  # Ultimate Clone
#    bash scripts/voxcpm.sh train --config_path VoxCPM/conf/voxcpm_v2/voxcpm_finetune_lora.yaml  # LoRA学習
#    bash scripts/voxcpm.sh app [--port 8808]                           # Web デモ
#    bash scripts/voxcpm.sh python <args...>                            # 環境内のpython直接実行
# ══════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VOXCPM_PROJECT="${PROJECT_DIR}/tools/voxcpm"
VOXCPM_REPO="${VOXCPM_PROJECT}/VoxCPM"

# torchcodec の core ライブラリは NPP/cuDNN 等の nvidia 共有ライブラリに runpath 無しで
# リンクするため、放置するとシステムの古い NPP(libnppicc.so.12)を拾い undefined symbol で
# 落ちる。venv 内の nvidia/*/lib を最優先にして回避（f5tts.sh と同じ対策）。
NV_LIB_DIRS="$(ls -d "${VOXCPM_PROJECT}"/.venv/lib/python*/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')"
if [ -n "${NV_LIB_DIRS}" ]; then
    export LD_LIBRARY_PATH="${NV_LIB_DIRS}${LD_LIBRARY_PATH:-}"
fi

CMD="$1"
shift || true

case "$CMD" in
    test)
        # mamimi 音声で全機能を網羅検証し data/output/voxcpm_test/ へ出力
        exec uv run --project "$VOXCPM_PROJECT" python "${SCRIPT_DIR}/test_voxcpm_mamimi.py" "$@"
        ;;
    design)
        exec uv run --project "$VOXCPM_PROJECT" voxcpm design "$@"
        ;;
    clone)
        exec uv run --project "$VOXCPM_PROJECT" voxcpm clone "$@"
        ;;
    batch)
        exec uv run --project "$VOXCPM_PROJECT" voxcpm batch "$@"
        ;;
    train)
        # LoRA / SFT ファインチューン。VoxCPM の train スクリプトを環境内で実行。
        # プロジェクトルートを cwd にして相対パスを通すため --directory で起動。
        exec uv run --project "$VOXCPM_PROJECT" --directory "$VOXCPM_REPO" \
            python scripts/train_voxcpm_finetune.py "$@"
        ;;
    app)
        exec uv run --project "$VOXCPM_PROJECT" --directory "$VOXCPM_REPO" \
            python app.py "$@"
        ;;
    python)
        exec uv run --project "$VOXCPM_PROJECT" python "$@"
        ;;
    *)
        echo "使い方: bash scripts/voxcpm.sh {test|design|clone|batch|train|app|python} <args...>"
        echo "  test    : scripts/test_voxcpm_mamimi.py を実行（mamimi 全機能検証）"
        echo "  design  : voxcpm design CLI（Voice Design）"
        echo "  clone   : voxcpm clone CLI（クローン / Ultimate Clone）"
        echo "  batch   : voxcpm batch CLI（一括処理）"
        echo "  train   : LoRA / SFT ファインチューン（train_voxcpm_finetune.py）"
        echo "  app     : Gradio Web デモ（app.py）"
        echo "  python  : VoxCPM 環境の python を直接実行"
        exit 1
        ;;
esac
