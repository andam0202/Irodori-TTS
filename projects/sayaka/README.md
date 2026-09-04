# projects/sayaka — 美樹さやか（魔法少女まどか☆マギカ）

CV 喜多村英梨。一人称「あたし」・快活で勢いのある口調。ツッコミ気質。
「〜じゃん」「〜だよね」「〜っての」など砕けた語尾。感情の振れ幅が大きい。

## 現状

- **sayaka v1**: サンゴチャンネルのマギレコ ボイス集3本（22.0分）から
  **107 セグメント / 約5.6分**で学習（2026-09-04 完了）。ベースは `Irodori-TTS-v4.1-Small`、
  v4-Small 小データ実績レシピ。学習は vast.ai の RTX 5090（1.51 step/s）。
- **best = step300 / val_loss 0.749929**。以降は単調悪化（600→0.750 / 900→0.794 /
  1200→0.835 / 1500→0.985）したため **step1500 で打ち切った**。reisa（同規模）と同じ挙動で、
  `docs/REISA_HANDOVER.md` 5.1 の「同規模なら 1500step で打ち切ってよい」の通り。
- 推論用（マージ済み）は
  `data/lora/sayaka_v1/checkpoint_best_val_loss_0000300_0.749929.safetensors`。
- テスト30件生成済み（`data/output/sayaka_test_v1/`）。**語尾の崖は 0/30 件**。
  ref_wav は `data/sayaka/wavs/seg_00056.wav`（7.9秒・有声率0.91）。
- データ量は reisa（115セグ / 6.0分）とほぼ同規模。**下限ぎりぎり**なので、
  品質が足りない場合は増量が第一手（下記「増量候補」）。

## ソース（YouTube ID・サンゴチャンネル）

| ID | 尺 | 内容 |
|---|---|---|
| `hH_N4GH1_Cw` | 9:18 | 美樹さやか 晴着Ver. 変身シーン＆ボイス一式 |
| `Xkkiz0EajdU` | 8:23 | 美樹さやか 波乗りVer. 変身＆ボイス一式 |
| `yRGDsG_X6Oo` | 4:23 | 美樹さやか（水着ver.）ボイス一式 |

さやか単独のボイス集なので**話者分離（diarize）は不要**。

### 増量候補（未DL）

サンゴチャンネルにはこれ以上さやかのボイス集が無い。増やすなら別チャンネル:
`7GtztB6s3Lw`（嫁コレ ボイス集+α 18:43・最長）/ `Wh8lnz-EbRc`（Vi vi 7:34）/
`eZGLJRRuu6U`（ほっけみりん scene0 完凸 7:34）/ `gh3cyoYj1aI`（ふんふん 6:30）/
`XAXtC7qIQww`（ファントムオブキル 4:49・同CV別ゲー）。

## 前処理

BS-RoFormer 分離 → 2秒無音を挟んで連結（モノラル 44.1kHz・22:08）→
`split_and_transcribe.py`（faster-whisper large-v3）で 133セグ → **26件除去して 107セグ**。

`metadata.csv` の `speaker` 列は**全行 `sayaka`**。

### ⚠ 除去した26件（`data/sayaka/rejected/`）

**24件が Whisper の幻聴「ご視聴ありがとうございました」**だった（`seg_00125`〜`seg_00157`
の連続ブロック＋冒頭2件）。波乗りVer. 動画の BGM のみの区間を Whisper が
定型句として埋めたもので、**8〜17秒の長いセグメントに短い定型テキストが付く**のが特徴。
残り2件はリピート幻聴（`seg_00032`）と崩れた文字起こし（`seg_00077`）。

**元データが22分と少ないため、この幻聴を落とさないと学習データの2割が
「BGM を読み上げる音声」になっていた**。短尺素材ほどこの検査は必須。

品質チェック（文字数 ÷ 有声秒数）は median 6.22 / p10 3.21 / p90 8.33。
残る低 c/s は「さあ!」「どうよ!」等の短い掛け声で、これは正常。

## 主要コマンド

```bash
uv run python prepare_manifest.py --dataset audiofolder \
  --data-files "train=data/sayaka/wavs/*.wav,data/sayaka/wavs/metadata.csv" \
  --audio-column audio --text-column transcription --speaker-column speaker \
  --output-manifest data/sayaka/manifest.jsonl --latent-dir data/sayaka/latents

uv run torchrun --nproc_per_node=1 train.py \
  --config projects/sayaka/train_sayaka_v1.yaml \
  --manifest data/sayaka/manifest.jsonl \
  --output-dir data/lora/sayaka_v1 \
  --init-checkpoint models/Irodori-TTS-v4.1-Small/model.safetensors
```

## 関連パス

- 学習データ: `data/sayaka/` / LoRA: `data/lora/sayaka_v1/`
- テスト台詞: `projects/sayaka/sayaka_test_lines.txt`（30行）
- ベースモデル: `models/Irodori-TTS-v4.1-Small/`
