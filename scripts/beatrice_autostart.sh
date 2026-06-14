#!/bin/bash
# ============================================================================
# beatrice_autostart.sh — VRAM が空いたら Beatrice 学習を自動起動する監視ループ
#   mamimi_v6 LoRA 学習中は空きVRAMが足りない(約3GB)ため、Beatrice(約9GB)を
#   同時起動すると OOM で mamimi を巻き込む。これを防ぐため、空きVRAMが閾値を
#   超える(=mamimi 学習が終わってVRAM解放される)まで待ってから train を起動する。
#   待機中は GPU に一切触れない(nvidia-smi の問い合わせのみ)ので mamimi に影響なし。
#
# 使い方: bash scripts/beatrice_autostart.sh   (バックグラウンド実行を推奨)
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 多重起動ガード: 何段ラップ起動されても実際に動く監視は1つだけにする
# (これがないと VRAM 解放時に Beatrice 学習が複数同時起動して OOM になる)
LOCKFILE="/tmp/beatrice_autostart.lock"
exec 9>"$LOCKFILE"
if ! flock -n 9; then
  echo "[$(date '+%F %T')] 既に監視が起動中のため終了"
  exit 0
fi

THRESHOLD_MIB=10000   # Beatrice 約9GB + マージン。これ以上空いたら起動
POLL_SEC=120          # 監視間隔(秒)
LOG="${PROJECT_DIR}/data/beatrice/autostart.log"
mkdir -p "${PROJECT_DIR}/data/beatrice"

echo "[$(date '+%F %T')] autostart 監視開始 (閾値 ${THRESHOLD_MIB}MiB, 間隔 ${POLL_SEC}s)" | tee -a "$LOG"

while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [ -n "$free" ] && [ "$free" -ge "$THRESHOLD_MIB" ]; then
    echo "[$(date '+%F %T')] 空きVRAM ${free}MiB >= ${THRESHOLD_MIB}MiB → Beatrice 学習を起動" | tee -a "$LOG"
    bash "${SCRIPT_DIR}/beatrice.sh" train >> "$LOG" 2>&1
    echo "[$(date '+%F %T')] Beatrice 学習プロセス終了 (exit=$?)" | tee -a "$LOG"
    break
  fi
  echo "[$(date '+%F %T')] 空きVRAM ${free}MiB < ${THRESHOLD_MIB}MiB → 待機" >> "$LOG"
  sleep "$POLL_SEC"
done
