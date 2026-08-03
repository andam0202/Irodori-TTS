#!/usr/bin/env bash
# 手動編集した manual/lines.txt から音声を生成する。
#
#   bash projects/bluearchive_asmr/generate_manual.sh              # 生成（既定: v4）
#   bash projects/bluearchive_asmr/generate_manual.sh --model v1   # 旧世代で生成
#   bash projects/bluearchive_asmr/generate_manual.sh --dry-run    # 確認のみ（生成しない）
#   bash projects/bluearchive_asmr/generate_manual.sh --list-presets
#
# 追加引数はそのまま make_manual_jsonl.py へ渡される（--lines で別ファイルを指定可）。
#
# ⚠ 出力先はここで固定しない。--model（v1/v4）に応じて make_manual_jsonl.py が
#   世代ごとのフォルダを選ぶ。ここで --outdir を渡すと世代分離が壊れる。
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

MANIFEST="projects/bluearchive_asmr/jsonl/manual.jsonl"

# 既定は「1フォルダに全部フラットに並べる」。--no-flat / --outdir で上書き可。
DEFAULTS=(--flat)

# --dry-run / --list-presets は jsonl を書かないので、そこで終了する
for arg in "$@"; do
    case "$arg" in
        --dry-run|--list-presets)
            uv run python projects/bluearchive_asmr/make_manual_jsonl.py "${DEFAULTS[@]}" "$@"
            exit 0
            ;;
    esac
done

uv run python projects/bluearchive_asmr/make_manual_jsonl.py "${DEFAULTS[@]}" "$@"

echo
echo "=== 生成開始（話者切替時のみモデルロード約40秒、以降は約2秒/行）"
uv run python scripts/batch_infer.py --manifest "$MANIFEST"

# 出力先はマニフェストの実際の out_path から取る（世代を取り違えないため）
OUTDIR="$(python3 -c "
import json, pathlib, sys
first = json.loads(open('$MANIFEST', encoding='utf-8').readline())
print(pathlib.Path(first['out_path']).parent)
")"
echo
echo "=== 出力先: $OUTDIR"
ls -1 "$OUTDIR"
