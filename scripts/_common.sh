#!/bin/bash
# ══════════════════════════════════════════════════════════
#  ラッパースクリプト共通ヘルパー（source して使う）
#
#    source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
#
#  source すると SCRIPT_DIR（scripts/）と PROJECT_DIR（リポジトリルート）が
#  設定される。隔離ツール環境を使うラッパーは続けて
#  irodori_export_nv_lib_path <tool_project_dir> を呼ぶこと。
# ══════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# torchcodec の core ライブラリ(libtorchcodec_core6.so 等)は NPP/cuDNN 等の
# nvidia 共有ライブラリに runpath 無しでリンクするため、放置するとシステムの
# 古い NPP(libnppicc.so.12 12.0.x)を拾い `undefined symbol: nppiNV12ToRGB_...`
# で落ちる。venv 内の nvidia/*/lib を LD_LIBRARY_PATH の最優先にして回避する。
irodori_export_nv_lib_path() {
    local tool_project="$1"
    local nv_lib_dirs
    nv_lib_dirs="$(ls -d "${tool_project}"/.venv/lib/python*/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')"
    if [ -n "${nv_lib_dirs}" ]; then
        export LD_LIBRARY_PATH="${nv_lib_dirs}${LD_LIBRARY_PATH:-}"
    fi
}
