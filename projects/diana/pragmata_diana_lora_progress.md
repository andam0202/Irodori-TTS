# PRAGMATA Diana 日本語版 LoRA 作成 — 進捗メモ

## 目的

PRAGMATA（カプコン）のキャラクター **Diana**（日本語CV: 東山奈央）の音声を使用し、Irodori-TTS で学習可能な LoRA モデルを作成する。

---

## 完了済み作業

### Step 1: 音声ダウンロード ✅

- **動画**: 【観るゲーム】PRAGMATA / 日本語音声・日本語字幕
- **URL**: https://www.youtube.com/watch?v=n9cr5isOwfE
- **保存先**: `data/input/diana/【観るゲーム】PRAGMATA ⧸ 日本語音声・日本語字幕.wav`
- **容量**: 1.9GB
- **長さ**: 約2時間51分（10,292秒）
- **コマンド**:
  ```bash
  uv run python scripts/download_audio.py "https://www.youtube.com/watch?v=n9cr5isOwfE" -s diana
  ```

### Step 2: BGM除去 + 話者分離 ✅（精度に課題あり）

- **スクリプト**: `scripts/diarize_speakers.py`
- **パイプライン**:
  1. Demucs (htdemucs) → BGM/SE 除去、ボーカル抽出（4.8分で完了）
  2. pyannote/speaker-diarization-3.1 → 話者分離（3.3分で完了）
- **コマンド**:
  ```bash
  uv run python scripts/diarize_speakers.py \
    --input "data/input/diana/【観るゲーム】PRAGMATA ⧸ 日本語音声・日本語字幕.wav" \
    --output-dir data/output/diarization/pragmata_diana \
    --num-speakers 2 \
    --hf-token <HF_TOKEN>
  ```

#### 話者分離結果（精度不良・要改善）

| 話者 | セグメント数 | 発話時間 |
|------|------------|---------|
| SPEAKER_00 | 100ファイル | 3分31秒 |
| SPEAKER_01 | 2,142ファイル | 51分24秒 |

- **保存先**: `data/output/diarization/pragmata_diana/speakers/SPEAKER_XX/*.wav`
- **JSON**: `data/output/diarization/pragmata_diana/【観るゲーム】PRAGMATA ⧸ 日本語音声・日本語字幕_diarization.json`
- **※一時ボーカルファイル（.tmp_vocals_*）は自動削除済み**

#### 話者分離の問題点

- 精度が非常に悪い（ユーザー確認による）
- 推定原因:
  - Diana（少女）の声とHugh（大人男性）の声が明確に異なるため、2話者指定は妥当だが、BGM残留ノイズやSE混入が分離精度に悪影響している可能性
  - 長時間音声（3時間）で話者分離の安定性が低下している可能性
  - pyannote のクラスタリング閾値が適切でなかった可能性

### Step 2.5: 話者埋め込みによる再クラスタリング ✅（2026-06-11）

- **原因分析**: `--num-speakers 2` の強制指定が精度不良の根本原因。実際の音声には
  Diana・Hugh・複数モブ・電子音声と多数の話者が含まれるため、無理な2分割で
  「モブ+電子音声 vs 主役2人」という誤った統合が起きた。
  セグメント境界自体は正しいため、クラスタ割り当てのみを修正した。
- **スクリプト**: `scripts/refine_speaker_clusters.py`（新規作成）
  - wespeaker 話者埋め込み（diarization-3.1 内部と同一モデル）を各セグメントから抽出
  - cosine 距離の凝集型クラスタリング（閾値 0.7、話者数は自動決定）
  - librosa.pyin による中央値 F0 で男声/女声の目安を表示
- **コマンド**:
  ```bash
  uv run python scripts/refine_speaker_clusters.py \
    --input-dirs data/output/diarization/pragmata_diana/speakers/SPEAKER_00 \
                 data/output/diarization/pragmata_diana/speakers/SPEAKER_01 \
    --output-dir data/output/diarization/pragmata_diana/reclustered
  ```
- **結果**: 1,465 ファイル（1秒未満の777件は除外）→ 94 クラスタ

| クラスタ | ファイル数 | 時間 | 中央値F0 | 推定 |
|---------|----------|------|---------|------|
| cluster_00 | 475 | 14.6min | 133 Hz | **Hugh（男声）** |
| cluster_01 | 388 | 12.5min | 415 Hz | **Diana 最有力候補（女声）** |
| cluster_02 | 104 | 2.9min | 407 Hz | Diana の可能性（分裂クラスタ?） |
| cluster_04 | 68 | 2.1min | 304 Hz | 要試聴 |
| cluster_03/05/06... | 少数 | - | - | モブ・電子音声など |

- cluster_00 と 01 はどちらも旧 SPEAKER_01 由来 → 「主役2人が混在」というユーザー観察と一致
- **出力先**: `data/output/diarization/pragmata_diana/reclustered/cluster_XX/`
- **試聴用サンプル**: `data/output/diarization/pragmata_diana/reclustered/samples/`
- **割り当て一覧**: `data/output/diarization/pragmata_diana/reclustered/assignments.csv`

### Step 3: 話者特定（ユーザー作業・次にやること）

- `reclustered/samples/` の各クラスタサンプル（特に cluster_01, 02, 04 と
  上位の女声クラスタ）を試聴し、Diana のクラスタを特定する
- Diana と確認できたクラスタの WAV を `data/diana/` 配下にまとめる:
  ```bash
  mkdir -p data/diana/selected
  cp data/output/diarization/pragmata_diana/reclustered/cluster_01/*.wav data/diana/selected/
  # cluster_02 等も Diana なら同様にコピー
  ```
- PRAGMATA のキャラクター構成:
  - **Diana**（少女・東山奈央）: cluster_01 が最有力
  - **Hugh**（大人男性・田中美央）: cluster_00 が最有力

### Step 4: 音声セグメント化 + 文字起こし ✅（2026-06-11）

- cluster_01 の 388 ファイルを結合 → `data/input/diana/diana_v1_source.wav`（12分30秒）
- `split_and_transcribe.py` に `--backend faster` / `--device` オプションを追加
  （openai 形式 .pt 2.9GB のダウンロードが不要になり、Windows 側 HF キャッシュの
  CTranslate2 形式 `Systran/faster-whisper-large-v3` を流用できる）
- **結果**: 141 セグメント + `metadata.csv`

```bash
# 結合（filelist は絶対パスで記述すること）
ls "$PWD"/data/output/diarization/pragmata_diana/reclustered/cluster_01/*.wav \
  | sed "s/^/file '/;s/$/'/" > /tmp/diana_filelist.txt
ffmpeg -y -f concat -safe 0 -i /tmp/diana_filelist.txt -ac 1 -ar 44100 \
  data/input/diana/diana_v1_source.wav

# セグメント化 + 文字起こし（faster-whisper バックエンド）
HF_HUB_CACHE=/mnt/c/Users/mao0202/.cache/huggingface/hub \
uv run python scripts/split_and_transcribe.py \
  --input-mp3 data/input/diana/diana_v1_source.wav \
  --output-dir data/diana/wavs \
  --backend faster
```

### Step 5: DACVAE 潜在変数エンコード + Manifest 生成 ✅（2026-06-11）

- **結果**: 141 サンプル → `data/diana/latents/` + `data/diana/manifest.jsonl`

```bash
uv run python scripts/encode_latents.py \
  --wavs-dir data/diana/wavs \
  --latent-dir data/diana/latents \
  --manifest-path data/diana/manifest.jsonl \
  --speaker-id diana
```

### Step 6: LoRA 学習（diana_v1）✅（2026-06-11）

- **設定**: `configs/train_diana_v1.yaml`（`train_mamimi_v5_forV3.yaml` ベース、
  600M-v3-VoiceDesign 用 LoRA 構成）
- **学習時間**: 3000 ステップ / 約60分（RTX 5070 Ti）
- **結果**: val_loss ベストは **step 600 の 1.008**（以降は過学習傾向、train loss は 0.77 まで低下）
- **マージ済み推論用モデル**: `data/lora/diana_v1/diana_v1_best.safetensors`（2.3GiB）

```bash
uv run python train.py \
  --config configs/train_diana_v1.yaml \
  --manifest data/diana/manifest.jsonl \
  --output-dir data/lora/diana_v1/ \
  --init-checkpoint models/base_v3_voicedesign/model.safetensors

uv run python convert_checkpoint_to_safetensors.py \
  data/lora/diana_v1/checkpoint_best_val_loss_0000600_1.007564 \
  --base-checkpoint models/base_v3_voicedesign/model.safetensors \
  --output data/lora/diana_v1/diana_v1_best.safetensors
```

### Step 7: 推論テスト ✅（2026-06-11）

- **テストスクリプト**: `scripts/run_test_diana_v1.sh`（SFW のみ・15サンプル）
  - グループ1: ベースライン / 2: 感情バリエーション / 3: ため息・息遣い / 4: 世界観セリフ
- **出力先**: `data/output/diana_test_v1/`
- ※ Diana は少女型アンドロイドのため NSFW 台詞は対象外（ユーザー合意済み）

```bash
bash scripts/run_test_diana_v1.sh
```

### Step 8: v2 作り直し — 生成末尾のゴミ音声不具合の修正 ✅（2026-06-11）

- **症状**: diana_v1 の生成音声で、指定台詞の後に識別不可能な声が続く
- **原因**: cluster_01 連結時に無音を挟まなかったこと
  1. Whisper が文境界を検出できず、単語の泣き別れ・複数セリフ混入が発生
     （例: 「成功」が seg_00000 末尾の「成」と seg_00001 先頭の「功」に分裂）
  2. `POST_ROLL=1.2s` が次の無関係セリフの冒頭を全クリップ末尾に取り込み、
     「テキストに無い音声」を学習 → 生成時にテキスト終了後もゴミ音声が続く
- **修正**: 各ファイル間に **2秒無音を挿入**して再連結（POST_ROLL より長いので
  末尾は純無音となり、末尾トリムで除去される）
- **効果**: 文字密度 median 5.6 → 8.0 文字/秒、泣き別れ解消、195 セグメント
- **教訓は CLAUDE.md「抽出セグメントを学習データ化する際の必須事項」に恒久記録済み**
- v2 成果物:
  - データ: `data/diana_v2/`（wavs 195本 + latents + manifest.jsonl）
  - 設定: `configs/train_diana_v2.yaml`
  - モデル: `data/lora/diana_v2/diana_v2_best.safetensors`
  - テスト: `scripts/run_test_diana_v2.sh` → `data/output/diana_test_v2/`

### Step 9: v3 作り直し — 生成音声の語尾途切れの修正 ✅（2026-06-11）

- **症状**: diana_v2 の生成音声で台詞の最後がわずかに途切れる
- **原因**: `split_and_transcribe.py` の末尾無音トリム `TAIL_SILENCE_DB=-45dB` が
  Demucs 処理音声（発話間がほぼデジタル無音）では攻撃的に作用し、語尾の自然減衰・
  無声子音を削って発話エネルギーのままブツ切りにしていた
  - 検証: 末尾80ms RMS 分析で cluster_01 ソースは正常（切断疑い16%）、
    v2 学習クリップは 74% が切断 → トリム工程が原因と特定
- **修正**: `TAIL_SILENCE_DB` を -60dB に変更 + トリム後に `apad=0.15s` で無音付加
- **効果**: 切断疑い 74% → **0%**（末尾下落幅 median 3.3dB → 60.6dB）、238セグメント
- v3 成果物:
  - データ: `data/diana_v3/`（238本 + latents + manifest.jsonl）
  - 設定: `configs/train_diana_v3.yaml` / モデル: `data/lora/diana_v3/diana_v3_best.safetensors`
  - テスト: `scripts/run_test_diana_v3.sh` → `data/output/diana_test_v3/`

### Step 10: 語尾途切れの推論側対策（infer.py 拡張）✅（2026-06-11）

- **症状**: v3 でも台詞の最後がごくわずかに途切れる（v2 より大幅改善済みだが残存）
- **分析**: 末尾減衰プロファイル測定により、モデルが語尾の自然減衰を生成できず
  「-10dB（ほぼフル音量）→ 50ms で無音」の崖を作っていることを確認。
  学習クリップは -26〜-32dB まで減衰してから無音（健全）なので、推論側の問題
- **試行**: `--duration-scale` 1.05〜1.15 → 発話が伸びて埋めるだけで効果なし
- **対策**（infer.py に新オプション実装）:
  - `--tail-margin-ms`（デフォルト100）: trim-tail のカット位置にマージン追加
  - `--tail-fade-ms`: 発話終端を自動検出し、その位置に cosine 減衰を適用
    （崖 -10→-74dB を -12→-15→-24→-43→無音 の自然な減衰に変換）
  - `--tail-pad-out-ms`: 出力末尾に無音付加
- テストスクリプトには `--tail-fade-ms 120 --tail-pad-out-ms 250` を設定済み

### Step 11: SE残留の根本解決 — BS-RoFormer 分離（v4）✅（2026-06-12）

- **症状**: v3 でも語尾がごくわずかに途切れる。span 単位で生音声と比較した結果、
  **台詞の語尾に SE（工業機械音・レーザー銃発砲音）が重なっており、htdemucs が
  除去しきれず残留ノイズ化**していたのが真因と判明（ユーザー試聴で確認）
- **対策**: 音源分離を htdemucs → **BS-RoFormer**（`model_bs_roformer_ep_317_sdr_12.9755`,
  SDR 12.98）に変更。`tools/audio-separator/` に uv 隔離環境を構築
  - **長尺対策**: 2h51m を一括処理すると分離後の WAV 書き出しでメモリ不足により
    無言でプロセスが落ちる。30分×6チャンクに分割して順次分離→連結で回避
- **効果**: SE残留箇所の末尾80ms RMS が span_0358 で -30dB→-90dB（完全無音）、
  span_0005（レーザー銃）で -33dB→-71dB。ユーザー試聴 OK
- v4 成果物:
  - 分離: `data/output/separation/diana_jp_bsro_vocals.wav`（全長ボーカル）
  - 再切り出し: `data/output/diarization/pragmata_diana/cluster_01_bsro/`（374スパン）
  - データ: `data/diana_v4/`（245本 + latents + manifest）
  - 設定: `configs/train_diana_v4.yaml` / モデル: `data/lora/diana_v4/diana_v4_best.safetensors`
  - テスト: `scripts/run_test_diana_v4.sh`（tail-fade 120 + pad 250 込み）→ `data/output/diana_test_v4/`
- **恒久化**: CLAUDE.md「音源分離」セクション、`scripts/reextract_segments.py` の警告で
  BS-RoFormer をデフォルト化

### 今後の改善候補

- データ量が約12分と少ないため、`docs/pragmata_diana_videos.md` の候補動画から
  追加ソースを確保すると品質向上が見込める
- cluster_02（2.9min, F0 407Hz）も Diana なら追加して diana_v2 を学習
- 過学習傾向が強い場合は max_steps を 1000〜1500 に短縮、または lora_r を下げる

---

## データディレクトリ構造

```
data/
├── input/diana/                          # 元音声（ダウンロード済み）
│   └── 【観るゲーム】PRAGMATA ⧸ 日本語音声・日本語字幕.wav  (1.9GB)
├── output/diarization/pragmata_diana/    # 話者分離結果
│   ├── speakers/                          # 初回分離（2話者強制・精度不良）
│   │   ├── SPEAKER_00/  (100 files, 3.5min)
│   │   └── SPEAKER_01/  (2142 files, 51min)
│   ├── reclustered/                       # 埋め込み再クラスタリング結果 ✅
│   │   ├── cluster_00/  (475 files, 14.6min, 男声=Hugh候補)
│   │   ├── cluster_01/  (388 files, 12.5min, 女声=Diana候補)
│   │   ├── cluster_XX/  ...
│   │   ├── samples/     (試聴用サンプル)
│   │   └── assignments.csv
│   └── *_diarization.json
├── diana/                                 # Step 4 以降の出力先（未作成）
│   ├── wavs/seg_XXXXX.wav + metadata.csv
│   ├── latents/seg_XXXXX.pt
│   └── manifest.jsonl
└── lora/diana_lora/                       # Step 6 の出力先（未作成）
```

---

## 環境情報

- **GPU**: NVIDIA GeForce RTX 5070 Ti (16GB VRAM)
- **Python**: 3.10
- **パッケージ管理**: uv
- **依存関係（追加済み）**: yt-dlp, demucs, pyannote.audio
- **HF Token**: 必要（pyannote/speaker-diarization-3.1, pyannote/segmentation-3.0 の利用規約同意済み）

## 候補動画リスト

→ `docs/pragmata_diana_videos.md` に一覧あり

## Diana 日本語声優情報

- **東山奈央**（Nao Toyama）
- 他の候補動画で追加の音声ソースを確保できる場合は、データ量を増やすことで学習精度が向上する可能性あり
