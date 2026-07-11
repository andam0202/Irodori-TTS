# PRAGMATA Diana 英語版 音声生成 — 進捗メモ

## 方針

英語の話者モデルは **F5-TTS** を使う（プロジェクト方針、`english-tts-f5tts` / CLAUDE.md「English TTS」）。
まず**ゼロショット音声クローン**（学習不要・参照クリップ1本）で音質を確認し、
不足なら fine-tune に進む段階的アプローチ（2026-06-12、ユーザー選択）。

英語版 Diana は日本語版（東山奈央）とは**別の声優**のため、参照・学習には
必ず英語音声を使う（日本語データは流用不可）。

## 完了済み

### Step 1: 英語ソース動画ダウンロード ✅
- E1: PRAGMATA All Diana Dialogue & Cutscenes 4K（GamersPrey, 英語音声）
- `data/input/diana_en/PRAGMATA All Diana Dialogue & Cutscenes 4K.wav`（1.6GB, 2h19m）

### Step 2: F5-TTS 隔離環境の修正 ✅（2026-06-12）
torchcodec が読み込めず推論不能だった問題を解決:
- **症状1**: `libnvrtc.so.13 不在` → PyPI の torchcodec 0.14（CUDA13ビルド）が入っていた。
  torch は cu128（CUDA12.8）なので不整合。
- **症状2**: torch 2.10 に下げると core6 が `undefined symbol: torch_dtype_float4_e2m1fn_x2`。
  → torchcodec 0.11.1+cu128 は **torch 2.11 向け**ビルド。
- **症状3**: `undefined symbol: nppiNV12ToRGB_..., version libnppicc.so.12`
  → torchcodec が NPP にリンクするが torch は NPP を引かず、システムの古い
  libnppicc.so.12（12.0.1.104, 2023年/CUDA12.0）を拾っていた。
- **最終解**（`tools/f5-tts/pyproject.toml`）:
  - `torch==2.11.0` / `torchaudio==2.11.0` / `torchcodec==0.11.1`（cu128 インデックス）
  - `nvidia-npp-cu12==12.3.3.100` を明示追加
  - `scripts/f5tts.sh` で venv 内 `nvidia/*/lib` を `LD_LIBRARY_PATH` 先頭に追加
    （core ライブラリに runpath が無くシステムの古い NPP を拾うのを防ぐ）

### Step 3: クリーンな Diana 参照クリップの抽出 ✅（2026-06-12）
2h19m から短いクリーン台詞を確保するため、複数時間帯の窓を分離・解析:
1. 5ウィンドウ（240/1200/2400/4200/6000秒 × 各90秒）を抽出
2. 各窓を **BS-RoFormer** で分離（`data/output/diana_en/vocals/`）
3. `scripts/find_diana_clip.py`: 無音区切り→近接区間結合→pyin F0 で
   **高音(F0≥230Hz=Diana)** の連続発話を抽出
4. 3候補を faster-whisper(large-v3) で英語書き起こし:
   - **ref_a**（推奨）: 4.2秒 F0 299Hz / "You may forget who I am, but I will always remember you."
   - ref_b（比較用）: 9.4秒 F0 319Hz / "I am DI-03367, a state-of-the-art pragmatic created here at the Cradle..."
   - cand_01: 4.1秒 F0 271Hz / "...are you still sad that Nicholas is gone?"
- 出力: `data/output/diana_en/ref/diana_ref_{a,b}.wav`

### Step 4: ゼロショット生成テスト ✅（2026-06-12）
- バッチ: `scripts/f5_batch_infer.py`（モデル1回ロードで8文生成、SFW・感情/世界観バリエーション）
- 実行: `bash scripts/run_test_diana_en.sh`（ref_a 8文 + ref_b 8文 = 計16本）
- 出力先: `data/output/diana_en_test/`（g1_greeting / g1_calm / g2_worried / g2_happy /
  g2_sad / g3_intro / g3_curious / g4_long の各 _refa / _refb）
- 生成は高速（GPU占有時 ~2s/文）。長さは ref_a 4.4〜8.1s、ref_b 5.2〜9.6s で妥当
  （ref_b はクリップの話速がやや遅く全体的に長め）
- F5 のログで各ファイルが指定 gen_text 通りに生成されたことを確認済み
- ※ Diana は少女型アンドロイドのため NSFW は対象外（日本語版と同じ方針）

→ **ゼロショットで実用品質の英語 Diana 音声が生成できる状態。** ユーザー試聴で
  声色の一致を確認し、十分なら完了。物足りなければ fine-tune（下記）へ。

## ハマりどころ・教訓

- **GPU競合**: F5生成がモデルをGPUに載せた後 0% util で停滞して見えるのは、
  別セッションの生成ジョブ(infer.py)とGPU計算を奪い合っているため。ハングではない。
  同時実行を避け、空いてから流すこと。
- F5-TTS の `remove_silence=True` は今回の停滞とは無関係だった（GPU競合が原因）。

## データ/成果物パス

```
data/input/diana_en/                          # 英語ソース(1.6GB, 2h19m)
data/output/diana_en/
├── windows/        # 抽出した90秒窓×5
├── vocals/         # BS-RoFormer 分離後ボーカル×5
└── ref/            # 参照クリップ候補 + diana_ref_{a,b}.wav
data/output/diana_en_test/                    # ゼロショット生成結果
```

## Step 5: F5-TTS フル fine-tune（2026-06-12〜）

ユーザー判断: ゼロショットは ref_a が良好。再利用可能なモデルを得るため fine-tune に進む。

### 5-1: 全長 BS-RoFormer 分離 🔄
- 元ソース 2h19m を 30分チャンク×5に分割（`data/output/separation/diana_en_chunks/`）
- 各チャンクを BS-RoFormer で分離 → `data/output/separation/diana_en_chunks_vocals/`
- 連結して全長ボーカル WAV を作成（予定: `data/output/separation/diana_en_bsro_vocals.wav`）

### 5-2: 話者分離 + 再クラスタリング（予定）
- `diarize_speakers.py --no-separate`（分離済みボーカル入力, **num-speakers 非指定**）
- `refine_speaker_clusters.py` で wespeaker 埋め込み再クラスタリング
- **ユーザーが samples を試聴して英語 Diana のクラスタを特定**
  （日本語版とは別声優。F0 は ref クリップで 299〜319Hz が Diana 目安）

### 5-3: データセット化（予定）
- Diana クラスタの WAV を **各間に2秒無音を挿入して連結**（CLAUDE.md 必須事項）
- `split_and_transcribe.py`（英語）で wavs + metadata.csv
- `bash scripts/f5tts.sh prepare <dir> <out>` で F5-TTS arrow データセットへ変換

### 5-2〜5-3: 話者分離→Diana特定→データセット化 ✅（2026-06-12）
- 話者分離（`diarize_speakers.py --no-separate`）→ 16話者 → `refine_speaker_clusters.py` で再クラスタリング
- **cluster_00（451ファイル, F0 368Hz）を英語 Diana とユーザーが特定**（cluster_01 は Hugh, F0 120Hz）
- cluster_00 を2秒無音挿入で連結（`data/input/diana_en/diana_en_v1_source.wav`, 56.7分）
- `split_and_transcribe.py --language en`（英語対応を追加）で 291セグメント + metadata.csv
- F5データセット化: `data/diana_en/metadata_f5.csv`（audio_file|text）→ `bash scripts/f5tts.sh prepare`
  → `data/diana_en/f5_dataset/`（291サンプル, 0.19時間, vocab=ベース2545char）
  - **vocab.txt 配置必須**: prepare は finetune時 `data/Emilia_ZH_EN_pinyin/vocab.txt` を要求。
    `f5_tts/infer/examples/vocab.txt`（2545行）を `data/{name}_{tokenizer}/` と
    `data/Emilia_ZH_EN_pinyin/` にコピーして解決した

### 5-4: fine-tune 🔄（2026-06-12〜）
- データセットを `<venv>/data/diana_en_pinyin/` に配置（finetune_cli の探索パス）
- 起動:
  ```
  bash scripts/f5tts.sh finetune --exp_name F5TTS_v1_Base --dataset_name diana_en \
    --tokenizer pinyin --finetune --pretrain <F5TTS_v1_Base/model_1250000.safetensors> \
    --learning_rate 1e-5 --batch_size_per_gpu 3200 --batch_size_type frame --epochs 120 \
    --save_per_updates 400 --last_per_updates 400 --keep_last_n_checkpoints 3
  ```
- 出力: `<venv>/ckpts/diana_en/`
- 学習後 `run_test_diana_en.sh` 相当で再生成し、ゼロショット版（ref_a）と比較予定

## 参考

- 追加英語ソース候補は `docs/pragmata_diana_videos.md`（E2/E3 など）
- 英語セリフ書き起こし: https://www.dawnborn.com/.../pragmata-game-transcript-all-dialogues/
