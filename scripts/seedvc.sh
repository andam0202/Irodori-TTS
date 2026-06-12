#!/bin/bash
# ══════════════════════════════════════════════════════════
#  Seed-VC ラッパースクリプト（ゼロショット音声変換・隔離環境呼び出し）
#
#  RVC がブレス/ため息などの非言語発声を既存音素に潰してしまう問題の代替。
#  Seed-VC は拡散ベースのゼロショットVCで、参照話者クリップ1本あれば学習不要。
#  環境は tools/seed-vc/ に uv で隔離（torch cu128, RTX 5070 Ti 対応）。
#
#  使い方:
#    bash scripts/seedvc.sh vc     --source <変換元.wav> --target <Diana参照.wav> --output <出力dir> [追加引数]
#    bash scripts/seedvc.sh vc-v2  --source <変換元.wav> --target <Diana参照.wav> --output <出力dir> [追加引数]
#    bash scripts/seedvc.sh python <args...>   # 環境内のpython直接実行
#
#  代表オプション（inference.py）:
#    --diffusion-steps 25（30〜50で高品質）/ --length-adjust 1.0
#    --inference-cfg-rate 0.7 / --fp16 True
#    --checkpoint <自前モデル> --config <自前config>（省略時はHFから自動DL）
# ══════════════════════════════════════════════════════════
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SEEDVC_DIR="${PROJECT_DIR}/tools/seed-vc"

CMD="$1"; shift || true

# seed-vc は pyproject.toml を持たず .venv に uv pip install で構築している。
# `uv run --project` は pyproject が無いと .venv を使わないため、venv の python を直接呼ぶ。
PY="${SEEDVC_DIR}/.venv/bin/python"

case "$CMD" in
    vc)
        cd "$SEEDVC_DIR" && exec "$PY" inference.py "$@"
        ;;
    vc-v2)
        cd "$SEEDVC_DIR" && exec "$PY" inference_v2.py "$@"
        ;;
    python)
        cd "$SEEDVC_DIR" && exec "$PY" "$@"
        ;;
    *)
        echo "使い方: bash scripts/seedvc.sh {vc|vc-v2|python} <args...>"
        echo "  vc    : inference.py（v1, whisper-small-wavenet 等）でゼロショット変換"
        echo "  vc-v2 : inference_v2.py（v2, hubert-bsqvae。話者性の抑制が最も強い）"
        echo "  python: Seed-VC 環境の python を直接実行"
        exit 1
        ;;
esac
