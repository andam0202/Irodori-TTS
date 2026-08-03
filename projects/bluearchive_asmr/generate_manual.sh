#!/usr/bin/env bash
# 手動編集した manual/lines.txt から音声を生成する。
#
#   bash projects/bluearchive_asmr/generate_manual.sh            # 生成
#   bash projects/bluearchive_asmr/generate_manual.sh --dry-run  # 確認のみ（生成しない）
#   bash projects/bluearchive_asmr/generate_manual.sh --list-presets
#
# 追加引数はそのまま make_manual_jsonl.py へ渡される（--lines で別ファイルを指定可）。
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

MANIFEST="projects/bluearchive_asmr/jsonl/manual.jsonl"
OUTDIR="/mnt/c/Users/mao0202/Desktop/bluearchive_manual"

# 既定は「デスクトップの1フォルダに全部フラットに並べる」。
# 追加引数は後ろに付くので、--outdir や --no-flat を渡せば上書きできる。
DEFAULTS=(--flat --outdir "$OUTDIR")

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
echo "=== 生成開始（話者切替時のみモデルロード約27秒、以降は約1.5〜2秒/行）"
uv run python scripts/batch_infer.py --manifest "$MANIFEST"

echo
echo "=== 出力先: $OUTDIR"
ls -1 "$OUTDIR"
