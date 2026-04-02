# Irodori-TTS 詳細使い方ガイド

Flow Matching ベースの日本語テキスト音声合成（TTS）システム。

## 目次

1. [システム要件](#1-システム要件)
2. [インストール](#2-インストール)
3. [モデルの種類](#3-モデルの種類)
4. [CLI 推論（infer.py）](#4-cli-推論inferpy)
5. [Gradio Web UI](#5-gradio-web-ui)
6. [データセット前処理（prepare_manifest.py）](#6-データセット前処理prepare_manifestpy)
7. [トレーニング（train.py）](#7-トレーニングtrainpy)
8. [LoRA 微調整](#8-lora-微調整)
9. [チェックポイント変換](#9-チェックポイント変換)
10. [推論パラメータ詳細](#10-推論パラメータ詳細)
11. [トレーニング設定詳細](#11-トレーニング設定詳細)
12. [アーキテクチャ概要](#12-アーキテクチャ概要)
13. [トラブルシューティング](#13-トラブルシューティング)

---

## 1. システム要件

| 項目 | 最小要件 | 推奨 |
|------|---------|------|
| Python | 3.10 以上 | 3.10 |
| PyTorch | 2.10.0 以上 | 2.10.0+cu128 |
| CUDA | 12.4 以上 | 12.8 |
| GPU VRAM | 8GB（推論） | 16GB 以上（トレーニング） |
| RAM | 16GB | 32GB 以上 |
| ストレージ | 10GB（モデル） | 100GB 以上（トレーニング） |

> **Note:** CPU 推論も可能だが、非常に低速（数分/文）。

---

## 2. インストール

### uv を使用した推奨インストール

```bash
# uv のインストール（未インストールの場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# リポジトリのクローン
git clone https://github.com/your-org/Irodori-TTS.git
cd Irodori-TTS

# 依存関係のインストール（自動で仮想環境作成）
uv sync
```

### pip を使用したインストール

```bash
pip install -e .
```

### 動作確認

```bash
# Python バージョンと torch の確認
uv run python -c "
import torch
print('torch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), 'GB')
import irodori_tts
print('irodori_tts: OK')
"
```

期待される出力例:
```
torch: 2.10.0+cu128
CUDA: True
GPU: NVIDIA GeForce RTX 5070 Ti
VRAM: 15.9 GB
irodori_tts: OK
```

---

## 3. モデルの種類

| モデル | HuggingFace ID | 用途 |
|--------|---------------|------|
| v2 基本モデル | `Aratako/Irodori-TTS-500M-v2` | テキスト + 参照音声クローン |
| VoiceDesign | `Aratako/Irodori-TTS-500M-v2-VoiceDesign` | テキスト + キャプション制御 |

モデルは初回実行時に自動ダウンロードされ、`~/.cache/huggingface/hub/` に保存されます。

---

## 4. CLI 推論（infer.py）

### 4.1 参照音声なし（--no-ref）

参照音声を使わずにデフォルトの声で合成する最もシンプルな使い方。

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v2 \
  --text "こんにちは。今日はいい天気ですね。" \
  --no-ref \
  --output-wav outputs/output.wav
```

### 4.2 参照音声あり（ゼロショット音声クローン）

参照音声ファイルを指定することで、その話者の声を模倣して合成する。

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v2 \
  --text "本日はよろしくお願いします。" \
  --ref-wav /path/to/speaker.wav \
  --output-wav outputs/cloned.wav
```

参照音声の推奨仕様:
- 形式: WAV（PCM）
- サンプリングレート: 任意（自動リサンプリング）
- 長さ: 3〜15 秒程度
- 内容: クリアな音声（BGM・ノイズなし）

### 4.3 VoiceDesign（キャプション条件付き）

テキストでスタイルを指定して音声合成する。参照音声不要。

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v2-VoiceDesign \
  --text "本日はお越しいただきありがとうございます。" \
  --caption "落ち着いた低めの男性の声で、ゆっくりと話す。" \
  --no-ref \
  --output-wav outputs/voicedesign.wav
```

キャプション例:
- `"元気な若い女性の声で、明るく話す。"`
- `"渋い中年男性の声で、落ち着いたトーンで話す。"`
- `"アナウンサー風の、はっきりとした標準語で話す。"`
- `"親しみやすい関西弁で、テンポよく話す。"`

### 4.4 ローカルチェックポイントの使用

```bash
uv run python infer.py \
  --checkpoint /path/to/model.safetensors \
  --text "テキストをここに入力してください。" \
  --no-ref \
  --output-wav outputs/output.wav
```

### 4.5 推論品質の調整

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v2 \
  --text "高品質な音声合成のテストです。" \
  --no-ref \
  --output-wav outputs/hq.wav \
  --num-steps 32 \              # サンプリングステップ数（デフォルト: 32）
  --cfg-scale-text 3.0 \        # テキスト CFG スケール（大きいほどテキスト忠実度↑）
  --cfg-scale-speaker 2.0 \     # 話者 CFG スケール
  --num-candidates 3 \          # 候補数（最良のものを選択）
  --seed 42                     # 再現性のためのシード
```

### 4.6 デバイスと精度の指定

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v2 \
  --text "テスト" \
  --no-ref \
  --output-wav outputs/output.wav \
  --model-device cuda \
  --model-precision bf16 \
  --codec-device cuda \
  --codec-precision bf16
```

### 4.7 torch.compile による高速化

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v2 \
  --text "torch.compile テスト" \
  --no-ref \
  --output-wav outputs/compiled.wav \
  --compile-model
# 初回は遅いが、2 回目以降は大幅高速化
```

---

## 5. Gradio Web UI

### 5.1 基本モデル用 Web UI

```bash
# デフォルト（localhost:7860）
uv run python gradio_app.py

# 外部公開
uv run python gradio_app.py --server-name 0.0.0.0 --server-port 7860

# HuggingFace Spaces デプロイ用（共有リンク）
uv run python gradio_app.py --share
```

ブラウザで `http://localhost:7860` を開く。

### 5.2 VoiceDesign 用 Web UI

```bash
uv run python gradio_app_voicedesign.py --server-port 7861
```

ブラウザで `http://localhost:7861` を開く。

### Web UI の操作方法

1. **Checkpoint**: `Aratako/Irodori-TTS-500M-v2` を入力（または HF ID）
2. **Text**: 合成したいテキストを入力
3. **Reference Audio**: 参照音声をアップロード（任意）
4. **Device / Precision**: `cuda` / `bf16` を推奨
5. **Generate** ボタンをクリック

---

## 6. データセット前処理（prepare_manifest.py）

HuggingFace データセットを学習用マニフェスト（JSONL）+ DACVAE レイテントに変換する。

### 基本的な使い方

```bash
uv run python prepare_manifest.py \
  --dataset your-org/your-tts-dataset \
  --split train \
  --audio-column audio \
  --text-column text \
  --output-manifest data/train_manifest.jsonl \
  --latent-dir data/latents/
```

### 話者情報あり

```bash
uv run python prepare_manifest.py \
  --dataset your-org/your-tts-dataset \
  --split train \
  --audio-column audio \
  --text-column text \
  --speaker-column speaker_id \
  --output-manifest data/train_manifest.jsonl \
  --latent-dir data/latents/
```

### VoiceDesign 用（キャプションあり）

```bash
uv run python prepare_manifest.py \
  --dataset your-org/your-tts-dataset \
  --split train \
  --audio-column audio \
  --text-column text \
  --caption-column style_caption \
  --output-manifest data/train_manifest_vd.jsonl \
  --latent-dir data/latents_vd/
```

### マニフェスト形式（JSONL）

出力されるマニフェストの各行形式:

```json
{"text": "こんにちは", "latent_path": "data/latents/000001.pt", "speaker_id": "speaker_a", "num_frames": 750}
{"text": "ありがとうございます", "caption": "落ち着いた男性の声", "latent_path": "data/latents/000002.pt", "num_frames": 512}
```

| フィールド | 必須 | 説明 |
|-----------|------|------|
| `text` | 必須 | 合成テキスト |
| `latent_path` | 必須 | DACVAE エンコード済みレイテントの相対パス |
| `num_frames` | 必須 | レイテントのフレーム数 |
| `speaker_id` | 任意 | 話者 ID（参照音声条件付け用） |
| `caption` | 任意 | スタイルキャプション（VoiceDesign 用） |

---

## 7. トレーニング（train.py）

### 7.1 設定ファイルの選択

| ファイル | 用途 |
|---------|------|
| `configs/train_500m_v2.yaml` | フルトレーニング（基本モデル） |
| `configs/train_500m_v2_lora.yaml` | LoRA 微調整（基本モデル） |
| `configs/train_500m_v2_voice_design.yaml` | VoiceDesign フルトレーニング |
| `configs/train_500m_v2_voice_design_lora.yaml` | VoiceDesign LoRA 微調整 |

### 7.2 単一 GPU トレーニング

```bash
uv run python train.py \
  --config configs/train_500m_v2.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts/
```

### 7.3 マルチ GPU DDP トレーニング

```bash
# 4 GPU
uv run torchrun --nproc_per_node 4 train.py \
  --config configs/train_500m_v2.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts/

# 特定 GPU 指定
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --nproc_per_node 4 train.py \
  --config configs/train_500m_v2.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts/
```

### 7.4 既存チェックポイントからの再開

```bash
uv run python train.py \
  --config configs/train_500m_v2.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts/ \
  --init-checkpoint outputs/irodori_tts/checkpoint_step10000.pt
```

### 7.5 W&B ログを有効にする

```bash
uv run python train.py \
  --config configs/train_500m_v2.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts/ \
  --wandb-enabled \
  --wandb-project irodori-tts \
  --wandb-run-name my-experiment
```

### 7.6 トレーニング出力

```
outputs/irodori_tts/
├── checkpoint_step1000.pt        # ステップごとチェックポイント
├── checkpoint_step2000.pt
├── ...
├── checkpoint_final.pt           # 最終チェックポイント
└── train_state.json              # トレーニング状態（再開用）
```

---

## 8. LoRA 微調整

### 8.1 事前学習済みモデルを基にした LoRA 微調整

```bash
# まず事前学習済みモデルをダウンロード
uv run python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Aratako/Irodori-TTS-500M-v2', 'model.safetensors', local_dir='.')
"

uv run python train.py \
  --config configs/train_500m_v2_lora.yaml \
  --manifest data/finetune_manifest.jsonl \
  --output-dir outputs/lora_finetune/ \
  --init-checkpoint model.safetensors
```

### 8.2 LoRA ターゲットモジュールのプリセット

`configs/train_500m_v2_lora.yaml` の `lora_target_modules` で指定:

| プリセット | 対象 | 用途 |
|-----------|------|------|
| `text_attn_mlp` | テキストエンコーダ | テキスト理解の調整 |
| `speaker_attn_mlp` | 参照エンコーダ | 話者適応 |
| `diffusion_attn` | Diffusion 注意層 | 生成スタイル調整 |
| `diffusion_attn_mlp` | Diffusion 注意 + MLP | より広範な生成調整 |
| `all_attn_mlp` | 全注意 + MLP | 包括的な調整 |

### 8.3 LoRA パラメータ設定例

```yaml
# configs/train_500m_v2_lora.yaml 抜粋
train:
  lora_enabled: true
  lora_r: 16           # ランク（大きいほど表現力↑、VRAM↑）
  lora_alpha: 32       # スケーリング係数（通常 2*r）
  lora_dropout: 0.0
  lora_target_modules: diffusion_attn_mlp
```

---

## 9. チェックポイント変換

### 9.1 学習チェックポイント → 推論用 safetensors

```bash
# フルモデルの変換
uv run python convert_checkpoint_to_safetensors.py \
  outputs/irodori_tts/checkpoint_final.pt

# 出力: outputs/irodori_tts/checkpoint_final.safetensors
```

### 9.2 LoRA アダプタ → マージ済み safetensors

```bash
uv run python convert_checkpoint_to_safetensors.py \
  outputs/lora_finetune/checkpoint_final.pt \
  --base-checkpoint model.safetensors

# 出力: outputs/lora_finetune/checkpoint_final_merged.safetensors
```

---

## 10. 推論パラメータ詳細

### 基本パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `--num-steps` | 32 | サンプリングステップ数。多いほど品質↑、速度↓ |
| `--num-candidates` | 1 | 候補生成数。多いほど最良解選択可能 |
| `--seed` | ランダム | 乱数シード（再現性確保） |

### CFG（Classifier-Free Guidance）

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `--cfg-scale-text` | 3.0 | テキスト忠実度（高→テキストに忠実） |
| `--cfg-scale-speaker` | 2.0 | 話者忠実度（高→参照音声に忠実） |
| `--cfg-scale-caption` | 3.0 | キャプション忠実度（VoiceDesign 用） |
| `--cfg-guidance-mode` | `independent` | CFG モード（independent/joint/alternating） |
| `--cfg-min-t` | 0.0 | CFG 適用開始タイムステップ |
| `--cfg-max-t` | 1.0 | CFG 適用終了タイムステップ |

### 参照音声設定

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `--max-ref-seconds` | 10.0 | 参照音声の最大長（秒） |
| `--ref-normalize-db` | -16.0 | 参照音声の音量正規化（dB） |
| `--ref-ensure-max` | True | ピーク音量の正規化 |

### 高度な制御

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `--truncation-factor` | 無効 | ノイズトランケーション（0.8〜0.95） |
| `--rescale-k` | 無効 | 時間的スコア再スケーリング k |
| `--rescale-sigma` | 無効 | 時間的スコア再スケーリング σ |
| `--context-kv-cache` | True | K/V キャッシュ事前計算（高速化） |
| `--trim-tail` | True | 末尾の無音トリミング |

---

## 11. トレーニング設定詳細

`configs/train_500m_v2.yaml` の主要設定:

```yaml
model:
  latent_dim: 32                      # DACVAE レイテント次元
  text_vocab_size: 99574
  text_tokenizer_repo: llm-jp/llm-jp-3-150m
  model_dim: 1280                     # Diffusion Transformer 次元
  num_layers: 12                      # Diffusion ブロック数
  num_heads: 20
  text_dim: 512                       # テキストエンコーダ次元
  text_layers: 10
  speaker_dim: 768                    # 参照エンコーダ次元
  speaker_layers: 8
  adaln_rank: 256                     # AdaLN ランク

train:
  # データ設定
  manifest_path: ""                   # CLI で上書き
  batch_size: 80
  max_latent_steps: 750               # 最大シーケンス長（フレーム）
  fixed_target_latent_steps: 750      # 固定長学習（パディング）
  
  # オプティマイザ
  optimizer: muon                     # muon または adamw
  learning_rate: 0.0001
  adam_beta1: 0.9
  adam_beta2: 0.999
  weight_decay: 0.01
  
  # スケジューラ（WSD: Warmup-Stable-Decay）
  lr_scheduler: wsd
  warmup_steps: 1000
  stable_steps: 44000
  max_steps: 50000
  min_lr_scale: 0.01
  
  # 精度
  precision: bf16                     # fp32 または bf16
  
  # ドロップアウト（CFG 用）
  text_condition_dropout: 0.1
  speaker_condition_dropout: 0.1
  
  # チェックポイント
  save_every: 1000
  checkpoint_best_n: 5
  
  # 検証
  valid_every: 1000
  valid_ratio: 0.0005
  
  # W&B
  wandb_enabled: false
```

---

## 12. アーキテクチャ概要

```
TextToLatentRFDiT (500M パラメータ)
├─ TextEncoder（10層 Transformer）
│   └─ テキストトークン → 文脈化表現
├─ ReferenceLatentEncoder（8層、参照音声用）
│   └─ 参照 DACVAE レイテント → 話者表現
├─ CaptionEncoder（10層、VoiceDesign 用）
│   └─ スタイルキャプション → 条件表現
└─ DiffusionTransformer（12層）
    ├─ JointAttention（テキスト + 参照のクロスアテンション）
    ├─ SwiGLU MLP
    └─ LowRankAdaLN（タイムステップ適応正規化）
```

### 推論フロー

```
テキスト
  → tokenize（llm-jp トークナイザ）
  → TextEncoder
  → K/V キャッシュ（高速化）
  ↓
[参照音声 WAV]
  → DACVAE エンコード（32次元レイテント）
  → ReferenceLatentEncoder
  → K/V キャッシュ（高速化）
  ↓
Gaussian Noise (z_1)
  → Euler Rectified Flow サンプリング（32ステップ）
  → CFG（テキスト/話者条件強化）
  → x_0（DACVAE レイテント）
  ↓
DACVAE デコード
  → WAV ファイル（44.1kHz）
```

---

## 13. トラブルシューティング

### CUDA Out of Memory

```bash
# 精度を下げる
--model-precision bf16 --codec-precision bf16

# 候補数を減らす
--num-candidates 1

# デコードモードを変更
--decode-mode sequential  # デフォルト: sequential（低VRAM）
```

### 音声品質が低い

```bash
# CFG スケールを調整
--cfg-scale-text 4.0 --cfg-scale-speaker 3.0

# ステップ数を増やす
--num-steps 64

# 複数候補から最良を選択
--num-candidates 5

# 参照音声を改善（クリアで3〜10秒の音声を使用）
```

### 参照音声のクローン精度が低い

```bash
# 参照音声の音量正規化を調整
--ref-normalize-db -12.0

# 話者 CFG を強化
--cfg-scale-speaker 4.0

# K/V スケール（実験的）
--speaker-kv-scale 1.5
```

### FutureWarning: weight_norm

```
FutureWarning: `torch.nn.utils.weight_norm` is deprecated...
```
このワーニングは無害。`dacvae` ライブラリ側の問題であり、動作に影響なし。

### HuggingFace ダウンロードが遅い/失敗

```bash
# HF_ENDPOINT で代替ミラーを使用
export HF_ENDPOINT=https://hf-mirror.com

# または手動ダウンロード後にローカルパスを指定
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('Aratako/Irodori-TTS-500M-v2', local_dir='./models/v2')
"

uv run python infer.py \
  --checkpoint ./models/v2/model.safetensors \
  --text "テスト" --no-ref --output-wav out.wav
```

### uv sync が遅い（WSL 環境）

Windows/WSL 環境では hardlink が使えないためコピーが発生する。
```bash
export UV_LINK_MODE=copy
uv sync
```

---

## 実行確認済み環境

| 項目 | 値 |
|------|---|
| OS | Linux (WSL2 6.6.87) |
| Python | 3.10.19 |
| PyTorch | 2.10.0+cu128 |
| CUDA | 12.8 |
| GPU | NVIDIA GeForce RTX 5070 Ti (16GB) |
| 推論速度（v2 基本/--no-ref） | ~27秒/文 |
| 推論速度（VoiceDesign） | ~3秒/文 |

実際の動作確認コマンド:

```bash
# 基本モデル（参照なし）
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v2 \
  --text "こんにちは。今日はいい天気ですね。" \
  --no-ref \
  --output-wav outputs/test_no_ref.wav \
  --show-timings

# VoiceDesign
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v2-VoiceDesign \
  --text "本日はお越しいただきありがとうございます。" \
  --caption "落ち着いた低めの男性の声で、ゆっくりと話す。" \
  --no-ref \
  --output-wav outputs/test_voicedesign.wav \
  --show-timings
```
