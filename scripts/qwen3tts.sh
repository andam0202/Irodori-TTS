#!/bin/bash
# ══════════════════════════════════════════════════════════
#  Qwen3-TTS ラッパースクリプト（多言語TTS: 英/露/中/韓/日 ほか10言語）
#
#  Alibaba Qwen3-TTS（Apache-2.0）を呼び出す。環境は tools/qwen3-tts/ に uv 隔離
#  （qwen-tts + torch 2.11.0+cu128、Blackwell 対応）。モデル重みは
#  tools/qwen3-tts/models/ に手動DL済み（aria2c、HF_HUB_OFFLINE で実行）。
#    - Qwen3-TTS-12Hz-1.7B-Base        : 3秒参照音声からのボイスクローン
#    - Qwen3-TTS-12Hz-1.7B-VoiceDesign : テキスト記述からのボイス作成
#
#  使い方:
#    bash scripts/qwen3tts.sh design --text "..." --instruct "voice description" --language English --output out.wav
#    bash scripts/qwen3tts.sh clone  --text "..." --ref-audio ref.wav --ref-text "..." --language Russian --output out.wav
#    bash scripts/qwen3tts.sh test [args...]     # NSFW多言語サンプル一括生成（EN/RU/ZH/KO）
#    bash scripts/qwen3tts.sh python <args...>   # 環境内の python 直接実行
# ══════════════════════════════════════════════════════════

set -e

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
QWEN_PROJECT="${PROJECT_DIR}/tools/qwen3-tts"
irodori_export_nv_lib_path "$QWEN_PROJECT"

# モデルはローカルDL済みのため HF へのネットワークアクセスを遮断
export HF_HUB_OFFLINE=1

CMD="$1"
shift || true

case "$CMD" in
    design|clone)
        exec uv run --project "$QWEN_PROJECT" python "${SCRIPT_DIR}/qwen3tts_infer.py" --mode "$CMD" "$@"
        ;;
    test)
        exec uv run --project "$QWEN_PROJECT" python "${SCRIPT_DIR}/test_qwen3tts_nsfw.py" "$@"
        ;;
    python)
        exec uv run --project "$QWEN_PROJECT" python "$@"
        ;;
    *)
        echo "使い方: bash scripts/qwen3tts.sh {design|clone|test|python} <args...>"
        echo "  design : VoiceDesign（テキスト記述からボイス作成、参照音声不要）"
        echo "  clone  : Base モデルで3秒ボイスクローン（--ref-audio/--ref-text）"
        echo "  test   : scripts/test_qwen3tts_nsfw.py（EN/RU/ZH/KO NSFWサンプル一括生成）"
        echo "  python : qwen3-tts 環境の python を直接実行"
        exit 1
        ;;
esac
