#!/bin/bash
# ══════════════════════════════════════════════════════════
#  pyannote ラッパースクリプト（話者分離・再クラスタリング用・隔離環境呼び出し）
#
#  本体(Irodori-TTS)は v4-Small 対応で torch/torchaudio 2.10 に上がり、
#  torchaudio から AudioMetaData / list_audio_backends が削除されたため
#  pyannote.audio 3.4 が import できなくなった。話者分離まわりだけ
#  tools/pyannote/ の隔離環境（pyannote.audio 4.x + torch cu128）で動かす。
#
#  使い方:
#    bash scripts/pyannote.sh diarize   <diarize_speakers.py の引数...>
#    bash scripts/pyannote.sh recluster <refine_speaker_clusters.py の引数...>
#    bash scripts/pyannote.sh python    <args...>   # 環境内の python 直接実行
#
#  例（CLAUDE.md のデフォルト手順）:
#    bash scripts/pyannote.sh diarize --input data/output/separation/<name>_vocals.wav --no-separate
#    bash scripts/pyannote.sh recluster \
#      --input-dirs data/output/diarization/<name>/speakers/SPEAKER_* \
#      --output-dir data/output/diarization/<name>/reclustered
#
#  ⚠ pyannote のモデルは gated。事前に HF で利用条件へ同意し、
#    HF_TOKEN（または HUGGINGFACE_TOKEN）を環境変数に設定しておくこと。
# ══════════════════════════════════════════════════════════

set -e

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
PYANNOTE_PROJECT="${PROJECT_DIR}/tools/pyannote"
irodori_export_nv_lib_path "$PYANNOTE_PROJECT"

CMD="$1"
shift || true

case "$CMD" in
    diarize)
        exec uv run --project "$PYANNOTE_PROJECT" python \
            "${SCRIPT_DIR}/diarize_speakers.py" "$@"
        ;;
    recluster)
        exec uv run --project "$PYANNOTE_PROJECT" python \
            "${SCRIPT_DIR}/refine_speaker_clusters.py" "$@"
        ;;
    python)
        exec uv run --project "$PYANNOTE_PROJECT" python "$@"
        ;;
    *)
        echo "usage: bash scripts/pyannote.sh {diarize|recluster|python} <args...>" >&2
        exit 1
        ;;
esac
