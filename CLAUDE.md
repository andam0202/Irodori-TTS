# Irodori-TTS Project Conventions

## Package Management
- Use **uv** for all package management (`uv sync`, `uv add`, `uv run`)
- Python version: 3.10 (`.python-version`)
- Dependencies: `pyproject.toml` / `uv.lock`

## Script Organization
- Python utility scripts → `scripts/` directory
- Run: `uv run python scripts/<name>.py`
- Shell scripts → `scripts/`
- CLI: argparse を使用（`scripts/split_and_transcribe.py` のパターンに従う）
- `from __future__ import annotations` を冒頭に記述

## Data Paths
- Raw audio input: `data/input/<speaker_name>/`
- Processed segments: `data/<speaker_name>/wavs/`
- DACVAE latents: `data/<speaker_name>/latents/`
- Training manifest: `data/<speaker_name>/manifest.jsonl`
- Diarization output: `data/output/diarization/<stem>/`

## 音源分離（BGM/SE 除去）— デフォルトは BS-RoFormer
ゲーム音声など BGM/SE が乗った素材からボーカル（台詞）を抽出する場合、
**htdemucs ではなく BS-RoFormer を使う**（2026-06-12 策定）。

- htdemucs（`diarize_speakers.py` の Demucs）は台詞の語尾に重なる SE
  （爆発音・レーザー音・機械音）を除去しきれず、残留ノイズ化する。
  これが学習データに混ざると **生成音声の語尾が不自然に途切れる/濁る**
  （diana_v1〜v4demucs で実害。BS-RoFormer 版 v4 で解消）。
- 環境: `tools/audio-separator/`（uv 隔離、torch cu128 + onnxruntime）
- モデル: `model_bs_roformer_ep_317_sdr_12.9755.ckpt`（SDR 12.98、htdemucs ~10 より大幅に上）
- **長尺音声は必ず 30 分チャンクに分割してから分離する**。全長を一度に処理すると
  分離後の WAV 書き出し時にメモリ不足でプロセスが落ちる（ログを残さず死ぬので注意）。

```bash
# 1) 元音声を30分チャンクに分割
ffmpeg -y -i <input>.wav -f segment -segment_time 1800 -c copy data/output/separation/chunks/chunk_%03d.wav
# 2) 各チャンクを順次分離（モデルは tools/audio-separator/models にキャッシュ）
for f in chunks/chunk_*.wav; do
  uv run --project tools/audio-separator audio-separator "$f" \
    -m model_bs_roformer_ep_317_sdr_12.9755.ckpt \
    --model_file_dir tools/audio-separator/models \
    --output_dir <vocals_dir> --output_format WAV --single_stem Vocals
done
# 3) チャンクのボーカルを連結 → 全長ボーカル WAV
```

その後の話者分離は `diarize_speakers.py --no-separate`（分離済みボーカルを入力）
または diarization JSON からの再切り出し（`reextract_segments.py --vocals-wav`）で行う。

## Speaker Diarization Workflow (デフォルト手順)
ゲーム実況など多話者音源から特定話者を抽出する場合は、必ず以下の2段階で行う:

1. **diarize_speakers.py**: `--num-speakers を指定しない`（自動推定に任せる）。
   実際の話者数が不明な音源で話者数を強制すると、クラスタが誤統合され
   別話者が混入する（例: 主役2人が同一クラスタに統合される事故）。
2. **refine_speaker_clusters.py**: diarization 出力のセグメントを wespeaker 話者埋め込みで
   再クラスタリングする。セグメント境界は正しくクラスタ割り当てだけが悪い場合、
   diarization の再実行（数時間）なしで修正できる。各クラスタの中央値 F0 と
   試聴用サンプル（`samples/`）が出力されるので、目的の話者をユーザーが特定する。

```bash
uv run python scripts/refine_speaker_clusters.py \
  --input-dirs data/output/diarization/<name>/speakers/SPEAKER_* \
  --output-dir data/output/diarization/<name>/reclustered
```

- 1秒未満のセグメントは埋め込みが不安定なため自動除外される
- クラスタが過分割される場合は `--threshold` を上げる（デフォルト 0.7）

### 抽出セグメントを学習データ化する際の必須事項
diarization セグメント群を `split_and_transcribe.py` 用に1本へ連結するときは、
**必ず各ファイル間に2秒以上の無音を挿入する**こと。無音なしで連結すると:
- Whisper が文境界（`split_by_gap`）を検出できず、複数セリフが1セグメントに混ざり
  単語が泣き別れる（テキストと音声の対応が崩れる）
- `POST_ROLL`（1.2s）が次の無関係セリフの冒頭を全クリップ末尾に取り込み、
  「テキストに無い音声」を学習 → **生成音声の末尾に意味不明な声が続く不具合**になる
  （diana_v1 で実際に発生。2秒無音挿入で v2 にて解消）

```bash
ffmpeg -y -f lavfi -i anullsrc=r=44100:cl=stereo -t 2 -c:a pcm_s16le /tmp/silence_2s.wav
ls "$PWD"/<segments_dir>/*.wav \
  | sed "s|^|file '|;s|$|'\nfile '/tmp/silence_2s.wav'|" > /tmp/filelist.txt
ffmpeg -y -f concat -safe 0 -i /tmp/filelist.txt -ac 1 -ar 44100 <output>.wav
```

また、Demucs 処理済み音声（発話間がほぼデジタル無音）に対する末尾無音トリムは
**-60dB + 無音パディング 0.15s** を使うこと（`split_and_transcribe.py` で対応済み）。
-45dB だと語尾の自然減衰・無声子音が削られて発話エネルギーのままブツ切りになり、
**生成音声の語尾が途切れる**（diana_v2 で実害、v3 で修正）。
データ品質検証: クリップ末尾80msのRMSが全体RMSより十分低い（>6dB差）ことを確認する。

### 生成音声の語尾途切れ対策（infer.py オプション）
モデルは語尾の自然減衰を生成しきれず、フル音量から約50msで無音に落ちる「崖」を
作ることがある（duration predictor のわずかな過小予測も寄与）。対策オプション:
- `--tail-margin-ms`（デフォルト100）: trim-tail の平坦化カット位置にマージンを追加
- `--tail-fade-ms 120`: **発話終端を自動検出**してそこに cosine 減衰を掛ける
  （ファイル末尾ではなく崖の位置に作用する。デフォルト0=無効）
- `--tail-pad-out-ms 250`: 出力末尾に無音を付加（デフォルト0=無効）

LoRA テスト生成スクリプトでは `--tail-fade-ms 120 --tail-pad-out-ms 250` を推奨
（`scripts/run_test_diana_v3.sh` 参照）。`--duration-scale` は発話速度が変わるだけで
語尾問題には効かない。

## English TTS (F5-TTS)
Irodori-TTS は日本語特化。**英語の話者モデルは F5-TTS を使う**（2026-06-11 策定）。

- 環境: `tools/f5-tts/` に uv 隔離環境（torch cu128、本体と依存が衝突しないよう分離）
- 呼び出し: `bash scripts/f5tts.sh {infer|finetune|prepare|python} <args...>`
- データ形式: `prepare` サブコマンドで wavs + metadata.csv（`audio_file|text` パイプ区切り）
  から F5-TTS の arrow 形式データセットへ変換できる
- 役割分担: 日本語 + NSFW/感情表現（キャプション・絵文字制御）= Irodori-TTS、
  英語のクリーンな台詞 = F5-TTS（F5 は非言語発声・スタイル制御が弱い）

## Code Style
- Ruff (lint + format, config in pyproject.toml)
- Line length: 100, double quotes, 4-space indent
