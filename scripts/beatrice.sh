#!/bin/bash
# ============================================================================
# beatrice.sh — Beatrice v2 声質変換トレーナー (fierce-cats/beatrice-trainer)
#   tools/beatrice-trainer/ の uv 隔離環境を呼ぶラッパー。
#   日本語 TTS(Irodori) とは独立した「リアルタイム声質変換(VC)」モデルを作る。
#
# 使い方:
#   bash scripts/beatrice.sh verify                          # torch/cuda 動作確認
#   bash scripts/beatrice.sh train [<data_dir> [<out_dir>]]  # 学習
#   bash scripts/beatrice.sh python <args...>                # 環境内 python 実行
#
# データ形式: <data_dir>/<speaker_name>/*.wav (モノラル, ネスト可)
#   既定 data_dir = data/beatrice/mamimi_v6 (話者 mamimi, 1099 wav)
# 出力: <out_dir>/paraphernalia_*  → 公式 Beatrice2 VST / VCClient で利用可
# 前提: GPU 約9GB 必要。mamimi LoRA 学習と同時には回せない(VRAM 競合)。
#       RTX5070(Blackwell) のため tools/beatrice-trainer は cu128 で sync すること。
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BT_DIR="${PROJECT_DIR}/tools/beatrice-trainer"

cmd="${1:-train}"
shift || true

case "$cmd" in
  train)
    DATA_DIR="${1:-${PROJECT_DIR}/data/beatrice/mamimi_v6}"
    OUT_DIR="${2:-${PROJECT_DIR}/data/beatrice/out_mamimi_v6}"
    mkdir -p "$OUT_DIR"
    echo "[beatrice] train data=${DATA_DIR} out=${OUT_DIR}"
    cd "$BT_DIR"
    uv run python -m beatrice_trainer -d "$DATA_DIR" -o "$OUT_DIR"
    ;;
  python)
    cd "$BT_DIR"
    uv run python "$@"
    ;;
  verify)
    cd "$BT_DIR"
    uv run python -c "import torch,torchaudio,pyworld;print('torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available())"
    ;;
  *)
    echo "unknown subcommand: ${cmd}" >&2
    echo "usage: beatrice.sh {verify|train|python} ..." >&2
    exit 1
    ;;
esac
