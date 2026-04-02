# mamimi 音声データ LoRA 学習ガイド

`data/input/mamimi/tanakamamimi_clipped_full.mp3`（約73分）を使って
LoRA 微調整し、mamimi の声で TTS 生成するまでの手順。

## 全体フロー

```
tanakamamimi_clipped_full.mp3 (73分)
  ↓ Step 1: 音声分割 + 文字起こし（scripts/split_and_transcribe.py）
data/mamimi/wavs/seg_*.wav（自然な文単位、約300〜400ファイル）
data/mamimi/wavs/metadata.csv
  ↓ Step 2: DACVAE レイテント変換（prepare_manifest.py）
data/mamimi/latents/*.pt + data/mamimi/manifest.jsonl
  ↓ Step 3: LoRA 微調整（train.py）
data/lora/mamimi_lora/checkpoint_final.pt
  ↓ Step 4: チェックポイント変換
data/lora/mamimi_lora/mamimi_final.safetensors
  ↓ Step 5: 推論（infer.py）
outputs/mamimi_output.wav
```

---

## Step 1: 音声分割 + 文字起こし

固定秒数で分割すると発話の途中でカットされてしまうため、
**faster-whisper のセグメントタイムスタンプ**を使って自然な文・フレーズ単位で分割する。
分割と文字起こしを1スクリプトで同時に行う。

### 1.1 faster-whisper のインストール

```bash
uv run pip install faster-whisper
```

### 1.2 スクリプトの実行

`scripts/split_and_transcribe.py` が音声分割と文字起こしを同時に行う。
内部では Whisper の VAD + セグメントタイムスタンプを使って自然な文境界で分割し、
各区間を ffmpeg で WAV として切り出す。

```bash
uv run python scripts/split_and_transcribe.py
```

完了後の確認:
```bash
ls data/mamimi/wavs/*.wav | wc -l
# → 約300〜400 ファイル（自然な文単位）

head -5 data/mamimi/wavs/metadata.csv
# file_name,transcription
# seg_00001.wav,こんにちは、田中まみみです。
# seg_00002.wav,今日も元気にやっていきましょう。
# ...
```

> **所要時間目安**: RTX 5070 Ti で約10〜20分（large-v3-turbo 使用時）

---

## Step 2: DACVAE レイテント変換

wavファイルを DACVAE エンコードしてレイテントと学習マニフェストを生成する。

```bash
mkdir -p data/mamimi/latents

uv run python prepare_manifest.py \
  --dataset audiofolder \
  --data-files "data/mamimi/wavs" \
  --split train \
  --audio-column audio \
  --text-column transcription \
  --speaker-column file_name \
  --speaker-id-prefix mamimi \
  --output-manifest data/mamimi/manifest.jsonl \
  --latent-dir data/mamimi/latents/ \
  --device cuda \
  --normalize-db -16 \
  --max-seconds 15 \
  --min-sample-rate 16000 \
  --progress
```

完了後の確認:
```bash
wc -l data/mamimi/manifest.jsonl
# → 使用可能なサンプル数

head -2 data/mamimi/manifest.jsonl
# {"text": "こんにちは、田中まみみです。", "latent_path": "data/mamimi/latents/xxxx.pt", "speaker_id": "mamimi/...", "num_frames": 437}
```

> **所要時間目安**: 約5〜10分（GPU使用時）

---

## Step 3: LoRA 微調整

### 3.1 LoRA 設定ファイルを作成

以下の内容を `configs/train_mamimi_lora.yaml` として保存する。
（小規模データセット向けにステップ数・バッチサイズを調整済み）

```yaml
# configs/train_mamimi_lora.yaml
model:
  latent_dim: 32
  latent_patch_size: 1
  text_vocab_size: 99574
  text_tokenizer_repo: llm-jp/llm-jp-3-150m
  model_dim: 1280
  num_layers: 12
  num_heads: 20
  mlp_ratio: 2.875
  text_mlp_ratio: 2.6
  speaker_mlp_ratio: 2.6
  text_dim: 512
  text_layers: 10
  text_heads: 8
  speaker_dim: 768
  speaker_layers: 8
  speaker_heads: 12
  speaker_patch_size: 1
  timestep_embed_dim: 512
  adaln_rank: 192

train:
  # データ
  batch_size: 4
  gradient_accumulation_steps: 4      # 実効バッチサイズ = 16
  num_workers: 4
  dataloader_persistent_workers: true
  dataloader_prefetch_factor: 2
  max_latent_steps: 750
  fixed_target_latent_steps: 750
  fixed_target_full_mask: true
  max_text_len: 256

  # 精度・最適化
  allow_tf32: true
  compile_model: false
  precision: bf16
  optimizer: adamw
  learning_rate: 0.00005             # LoRA は低めの学習率
  weight_decay: 0.01

  # スケジューラ（小規模データ向け）
  lr_scheduler: wsd
  warmup_steps: 100
  stable_steps: 2600
  min_lr_scale: 0.01
  max_steps: 3000

  # ドロップアウト（CFG 用）
  text_condition_dropout: 0.1
  speaker_condition_dropout: 0.1
  timestep_stratified: true

  # ログ・チェックポイント
  log_every: 50
  save_every: 300
  checkpoint_best_n: 5
  valid_ratio: 0.05                  # 小規模データなので検証割合を多めに
  valid_every: 300

  # W&B（任意）
  wandb_enabled: false
  wandb_project: irodori-tts-mamimi
  wandb_run_name: mamimi-lora

  # LoRA 設定
  lora_enabled: true
  lora_r: 32                         # ランク（16〜64 推奨）
  lora_alpha: 64                     # 通常 2 × lora_r
  lora_dropout: 0.05
  lora_bias: none
  lora_target_modules: diffusion_attn_mlp  # 注意層 + MLP を微調整

  seed: 42
  ddp_find_unused_parameters: false
```

### 3.2 ベースモデルのダウンロード

```bash
uv run python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    'Aratako/Irodori-TTS-500M-v2',
    'model.safetensors',
    local_dir='models/base'
)
print('Downloaded:', path)
"
```

### 3.3 LoRA 学習の実行

```bash
uv run python train.py \
  --config configs/train_mamimi_lora.yaml \
  --manifest data/mamimi/manifest.jsonl \
  --output-dir data/lora/mamimi_lora/ \
  --init-checkpoint models/base/model.safetensors
```

学習ログの確認（別ターミナル）:
```bash
tail -f data/lora/mamimi_lora/train.log
```

> **所要時間目安**:
> - 3000ステップ、バッチ4 → RTX 5070 Ti で約30〜60分
> - 途中で中断した場合は `--resume data/lora/mamimi_lora/checkpoint_stepXXXX` で再開可能

---

## Step 4: チェックポイント変換

LoRA アダプタをベースモデルにマージして推論用 safetensors を生成する。

```bash
# 最終チェックポイントをマージ変換（LoRA アダプタディレクトリを指定）
uv run python convert_checkpoint_to_safetensors.py \
  data/lora/mamimi_lora/checkpoint_final

# → data/lora/mamimi_lora/mamimi_final.safetensors が生成される
```

バリデーション損失が最も低いチェックポイントを変換する場合:
```bash
# 例: val_loss が最小のチェックポイントディレクトリ名を確認してから実行
ls data/lora/mamimi_lora/checkpoint_best_*
uv run python convert_checkpoint_to_safetensors.py \
  data/lora/mamimi_lora/checkpoint_best_val_loss_XXXXXX_X.XXXXXX
```

---

## Step 5: 推論

### 5.1 参照音声なし（--no-ref）

```bash
uv run python infer.py \
  --checkpoint data/lora/mamimi_lora/mamimi_final.safetensors \
  --text "こんにちは！今日もよろしくお願いします！" \
  --no-ref \
  --output-wav outputs/mamimi_test_noref.wav \
  --show-timings
```

### 5.2 参照音声あり（クローン精度向上）

学習データの WAV を参照音声として渡すとより mamimi らしい声になる。

```bash
uv run python infer.py \
  --checkpoint data/lora/mamimi_lora/mamimi_final.safetensors \
  --text "今日のゲームはめちゃくちゃ楽しかったですよ。" \
  --ref-wav data/mamimi/wavs/seg_00010.wav \
  --cfg-scale-speaker 2.5 \
  --output-wav outputs/mamimi_test_ref.wav \
  --show-timings
```

### 5.3 複数候補から最良を選択

```bash
uv run python infer.py \
  --checkpoint data/lora/mamimi_lora/mamimi_final.safetensors \
  --text "ありがとうございます！すごく嬉しいです！" \
  --ref-wav data/mamimi/wavs/seg_00010.wav \
  --num-candidates 5 \
  --cfg-scale-text 3.5 \
  --cfg-scale-speaker 3.0 \
  --seed 42 \
  --output-wav outputs/mamimi_best.wav
```

---

## 品質調整のヒント

### 声が似ていない場合

| 原因 | 対処 |
|------|------|
| 参照音声が悪い | より発話量が多い・クリアな WAV を選ぶ |
| CFG スケールが低い | `--cfg-scale-speaker 3.0〜4.0` に上げる |
| 学習不足 | `max_steps` を 5000〜10000 に増やして再学習 |
| LoRA ランクが低い | `lora_r: 64, lora_alpha: 128` に増やして再学習 |

### テキストの発話が不自然な場合

| 原因 | 対処 |
|------|------|
| テキスト CFG が低い | `--cfg-scale-text 4.0〜5.0` に上げる |
| ステップ数が少ない | `--num-steps 64` に増やす |
| テキストが長すぎる | 文を短く分割する（目安: 50文字以内/文） |

### 学習が収束しない場合

```yaml
# configs/train_mamimi_lora.yaml を調整
train:
  learning_rate: 0.00002    # さらに下げる
  lora_r: 16                # ランクを下げてオーバーフィット抑制
  lora_dropout: 0.1         # ドロップアウトを上げる
```

---

## 参照音声として使いやすい WAV の選び方

```bash
# ファイルサイズ（≒音声量）で上位のファイルを確認
ls -lS data/mamimi/wavs/seg_*.wav | head -20

# 実際に再生して確認（WSL の場合は Windows 側で開く）
explorer.exe "$(wslpath -w data/mamimi/wavs/seg_00010.wav)"
```

推奨する参照音声の特徴:
- 5〜10秒程度の明瞭な発話
- BGM・効果音が少ない
- 感情が穏やか（推論時の声調に影響する）

---

## ファイル構成まとめ

```
data/
├── input/mamimi/
│   └── tanakamamimi_clipped_full.mp3    # 元音声
├── mamimi/
│   ├── wavs/
│   │   ├── metadata.csv                 # 文字起こし結果
│   │   ├── seg_00001.wav
│   │   └── ...
│   ├── latents/
│   │   ├── xxxxxx.pt                    # DACVAE レイテント
│   │   └── ...
│   └── manifest.jsonl                   # 学習マニフェスト
models/
└── base/
    └── model.safetensors                # ベースモデル
configs/
└── train_mamimi_lora.yaml               # LoRA 学習設定
data/
└── lora/
    └── mamimi_lora/
        ├── checkpoint_final/            # LoRA アダプタ（最終）
        ├── checkpoint_best_val_loss_*/  # バリデーション最良チェックポイント
        ├── mamimi_final.safetensors     # 推論用（変換後・最終）
        └── mamimi_best.safetensors      # 推論用（変換後・最良）
scripts/
├── split_and_transcribe.py              # 音声分割 + 文字起こし
├── encode_latents.py                    # DACVAE レイテント変換
└── generate_mamimi.sh                   # 摩美々音声生成スクリプト
```
