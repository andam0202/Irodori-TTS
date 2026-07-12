# Irodori-TTS Project Conventions

## Package Management
- Use **uv** for all package management (`uv sync`, `uv add`, `uv run`)
- Python version: 3.10 (`.python-version`)
- Dependencies: `pyproject.toml` / `uv.lock`

## Script Organization
- **プロジェクト固有**（話者 LoRA・用途別検証）のスクリプト/config/ドキュメントは
  `projects/<name>/` に置く（索引: `projects/README.md`）。新話者 config は
  `scripts/new_speaker_lora_config.py` で生成する
- **共有ツール**（データ前処理パイプライン・エンジンラッパー・共通ヘルパー）→ `scripts/`
- 話者非依存の汎用学習 config → `configs/`、汎用ドキュメント → `docs/`
- Run: `uv run python scripts/<name>.py`
- CLI: argparse を使用（`scripts/split_and_transcribe.py` のパターンに従う）
- `from __future__ import annotations` を冒頭に記述
- 各プロジェクトの `archive/` は世代交代済みの無変更保管（本規約の適用外）

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
（`projects/diana/run_test_diana_v4.sh` 参照。ランナーは `TAIL_ARGS` を設定して `scripts/_infer_lora_common.sh` を source する方式）。`--duration-scale` は発話速度が変わるだけで
語尾問題には効かない。

## English TTS (F5-TTS)
Irodori-TTS は日本語特化。**英語の話者モデルは F5-TTS を使う**（2026-06-11 策定）。

- 環境: `tools/f5-tts/` に uv 隔離環境（torch cu128、本体と依存が衝突しないよう分離）
- 呼び出し: `bash scripts/f5tts.sh {infer|finetune|prepare|python} <args...>`
- データ形式: `prepare` サブコマンドで wavs + metadata.csv（`audio_file|text` パイプ区切り）
  から F5-TTS の arrow 形式データセットへ変換できる
- 役割分担: 日本語 + NSFW/感情表現（キャプション・絵文字制御）= Irodori-TTS、
  英語のクリーンな台詞 = F5-TTS（F5 は非言語発声・スタイル制御が弱い）

## 多言語 TTS / ボイスクローン（VoxCPM2）
Irodori-TTS（日本語特化・600M）の対抗馬。OpenBMB の 2B **tokenizer-free** モデルで、
30言語対応TTS、Voice Design（テキスト記述からボイス作成）、ゼロショット/Ultimate ボイスクローン、
48kHz出力、SFT/LoRA ファインチューンを持つ。

- 環境: `tools/voxcpm/` に uv 隔離（VoxCPM リポジトリを editable install、torch cu128）。
  RTX 5070 Ti(Blackwell) 対応のため torch==2.11.0+cu128 に固定（f5-tts と同じ）。
- モデル: `tools/voxcpm/models/VoxCPM2/`（HF `openbmb/VoxCPM2`、~4.6GB）。
  **huggingface_hub の Xet/LFS DL がこの環境で CloudFront 接続後に停止する** →
  aria2c で手動DLすること（`-x16 -s16 -o <name>` を各ファイルに）。`-j2` で複数ファイルを
  一度に渡すと2つ目が Xet 署名エラーで落ちるので**1ファイルずつ**落とす。DL後は
  `--model <path>` + `HF_HUB_OFFLINE=1` でネットワークを回避して実行。
- 呼び出し: `bash scripts/voxcpm.sh {test|design|clone|batch|train|app|python} <args...>`
  - `test`   : mamimi で全機能検証（`projects/mamimi/test_voxcpm_mamimi.py` → `data/output/voxcpm_test/`）
  - `design` / `clone` / `batch` : `voxcpm` CLI（Voice Design / クローン / 一括）
  - `train`  : LoRA/SFT ファインチューン（`VoxCPM/scripts/train_voxcpm_finetune.py`）
- API: `from voxcpm import VoxCPM`; `model.generate(text="(control)text",
  reference_wav_path=..., prompt_wav_path=..., prompt_text=..., cfg_value=2.0,
  inference_timesteps=10, normalize=, denoise=)`。初回推論前に torch.compile の
  ウォームアップ（~4分）が入るので、機能を分けて複数プロセスで回すより1プロセスで一括が速い。
- 役割分担: 多言語・クロスリンガル・Voice Design = VoxCPM2、日本語+NSFW 感情表現 = Irodori-TTS

## 多言語 NSFW 音声（Qwen3-TTS）— 英/露/中/韓
日本語以外（英語・ロシア語・中国語・韓国語）の **NSFW セリフ・喘ぎ声**は
**Qwen3-TTS**（Alibaba、Apache-2.0、2026-01 公開）を使う（2026-07-09 策定）。

- 選定理由: 要求4言語＋日本語を含む10言語対応、**Apache-2.0 で商用ゲーム利用可**、
  ローカル重みでコンテンツフィルタなし、`finetuning/` 同梱で喘ぎ声データでのFTも可能。
  対抗馬 OpenAudio S1-mini は (panting)(groaning) マーカーを持ち音響的には有力だが
  **CC-BY-NC-SA（商用不可）**のため不採用。Chatterbox Multilingual(MIT, 23言語)は次点。
- 環境: `tools/qwen3-tts/` に uv 隔離（PyPI `qwen-tts` + torch==2.11.0+cu128、python 3.12）。
  **torchaudio は必ず直接依存に書いて cu128 インデックスから入れる**こと。qwen-tts の
  推移依存で PyPI 版 torchaudio(CUDA 13 ビルド)が入ると `libcudart.so.13` 不在で落ちる。
- モデル: `tools/qwen3-tts/models/Qwen3-TTS-12Hz-1.7B-{Base,VoiceDesign}`（各 ~4.3GB）。
  aria2c 1ファイルずつDL（VoxCPM2 と同じ Xet 対策。`-x16` だと範囲リクエストの署名不一致で
  403 が混ざるが aria2c が自動リトライして完走する）。実行は `HF_HUB_OFFLINE=1`（ラッパーが設定）。
- 呼び出し: `bash scripts/qwen3tts.sh {design|clone|test|python} <args...>`
  - `design` : VoiceDesign（テキスト記述からボイス作成、参照音声不要）
  - `clone`  : Base モデルで3秒ボイスクローン（`--ref-audio` + `--ref-text`）
  - `test`   : `projects/multilingual/test_qwen3tts_nsfw.py`（EN/RU/ZH/KO × セリフ/喘ぎ/囁き →
    `data/output/qwen3tts_test/`、24kHz 出力）
- マーカー構文はなく、**instruct（ボイス記述）＋テキスト中の擬音**で表現する。
  喘ぎは "Ahh... mmm... hah..."（各言語の擬音表記）＋ "moaning in pleasure, breathless panting"
  系の instruct で生成できる。品質が足りない場合は Base モデルを喘ぎデータでFTする。
- 役割分担: 日本語 NSFW = Irodori-TTS、**外国語 NSFW/喘ぎ = Qwen3-TTS**、
  英語クリーン台詞 = F5-TTS、多言語クリーン・クロスリンガル = VoxCPM2

## Code Style
- Ruff (lint + format, config in pyproject.toml)
- Line length: 100, double quotes, 4-space indent
