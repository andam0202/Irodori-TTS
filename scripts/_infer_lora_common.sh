#!/bin/bash
# ══════════════════════════════════════════════════════════
#  LoRA テストランナー共通ヘルパー（source して使う）
#
#  呼び出し側は _common.sh を source（PROJECT_DIR 設定）した後、
#    CHECKPOINT : LoRA チェックポイント (.safetensors)
#    REF_WAV    : 参照音声
#    OUTPUT_DIR : 出力ディレクトリ（source 時に mkdir される）
#    SEED       : シード（省略時 42）
#    TAIL_ARGS  : 語尾処理等の追加フラグ文字列（任意、例:
#                 "--tail-fade-ms 120 --tail-pad-out-ms 250"）
#  を設定してから本ファイルを source する。セリフ群は各ランナーに書く。
#
#  提供する関数:
#    run       <text> <name>                    : 参照音声のみで生成
#    run_cap   <text> <caption> <name>          : キャプション付き生成
#    run_noref <text> <caption> <name>          : 参照音声なし（--no-ref）
#    run_param <text> <caption> <name> <args…>  : 追加推論パラメータ付き
#    infer_lora <args…> [-- <seed後args…>]      : 低レベル（専用ヘルパー向け）
# ══════════════════════════════════════════════════════════

if [ -z "${CHECKPOINT:-}" ] || [ -z "${REF_WAV:-}" ] || [ -z "${OUTPUT_DIR:-}" ]; then
    echo "error: _infer_lora_common.sh を source する前に CHECKPOINT/REF_WAV/OUTPUT_DIR を設定すること" >&2
    exit 1
fi
SEED="${SEED:-42}"

mkdir -p "$OUTPUT_DIR"

# infer.py を共通パラメータ付きで起動する。`--` より後ろの引数は
# --seed/TAIL_ARGS のさらに後ろに付く（実験ごとの上書きパラメータ用）。
infer_lora() {
    local pre=() post=() seen_sep=0 a
    for a in "$@"; do
        if [ "$a" = "--" ] && [ "$seen_sep" -eq 0 ]; then
            seen_sep=1
            continue
        fi
        if [ "$seen_sep" -eq 0 ]; then pre+=("$a"); else post+=("$a"); fi
    done
    # shellcheck disable=SC2086  # TAIL_ARGS は意図的に単語分割する
    uv run python "${PROJECT_DIR}/infer.py" \
        --checkpoint "$CHECKPOINT" \
        "${pre[@]}" \
        --seed "$SEED" \
        ${TAIL_ARGS:-} \
        "${post[@]}" \
        2>&1 | tail -1
}

run() {
    local text="$1" name="$2"
    echo "[generate] ${name}"
    infer_lora \
        --text "$text" \
        --ref-wav "$REF_WAV" \
        --output-wav "${OUTPUT_DIR}/${name}.wav"
}

run_cap() {
    local text="$1" caption="$2" name="$3"
    echo "[generate] ${name} (caption: ${caption:0:30}...)"
    infer_lora \
        --text "$text" \
        --caption "$caption" \
        --ref-wav "$REF_WAV" \
        --output-wav "${OUTPUT_DIR}/${name}.wav"
}

run_noref() {
    local text="$1" caption="$2" name="$3"
    echo "[generate] ${name} (no-ref, caption: ${caption:0:30}...)"
    infer_lora \
        --text "$text" \
        --caption "$caption" \
        --no-ref \
        --output-wav "${OUTPUT_DIR}/${name}.wav"
}

run_param() {
    local text="$1"; shift
    local caption="$1"; shift
    local name="$1"; shift
    echo "[generate] ${name} (extra args: $*)"
    infer_lora \
        --text "$text" \
        --caption "$caption" \
        --ref-wav "$REF_WAV" \
        --output-wav "${OUTPUT_DIR}/${name}.wav" \
        -- "$@"
}
