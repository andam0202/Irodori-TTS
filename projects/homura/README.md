# projects/homura — 暁美ほむら（魔法少女まどか☆マギカ）

CV 斎藤千和。一人称「私」・「〜わ」「〜のよ」「〜なさい」の落ち着いた断定調。
クールで淡々とした低めの発話。まどかに関することだけ感情が動く。

## 現状

- **homura v1**: サンゴチャンネルのマギレコ／まどドラ ボイス集5本（49.1分）から
  **483 セグメント / 約22.5分**で学習（2026-09-04 完了）。ベースは `Irodori-TTS-v4.1-Small`、
  v4-Small 小データ実績レシピ 3000step。学習は vast.ai の RTX 5090 で **33分44秒 / 1.49 step/s**。
- **best = step900 / val_loss 0.743602**。推論用（マージ済み）は
  `data/lora/homura_v1/checkpoint_best_val_loss_0000900_0.743602.safetensors`。
- テスト30件生成済み（`data/output/homura_test_v1/`）。**語尾の崖は 0/30 件**。
  ref_wav は `data/homura/wavs/seg_00578.wav`（4.5秒・落ち着いた語り・有声率0.92）。

### val_loss の推移

| step | 300 | 600 | 900 | 1200 | 1500 | 1800 | 2100 | 2400 | 3000 |
|---|---|---|---|---|---|---|---|---|---|
| val_loss | 0.784 | 0.789 | **0.744** | 0.753 | 0.773 | 0.768 | 0.765 | 0.754 | 0.757 |

483セグと小データ勢（reisa 115 / kiriko 173）より多いためか、**best が step900 まで後ろに
ずれ、その後も 0.75 前後で大きく崩れない**（reisa は step300 以降単調悪化）。

## ソース（YouTube ID・サンゴチャンネル）

| ID | 尺 | 内容 |
|---|---|---|
| `M7N2mubs0ic` | 12:53 | 暁美ほむら 晴着ver. ボイス一式 |
| `2z0cNxkyUUQ` | 11:45 | 悪魔ほむら 完凸 ボイス一式 |
| `pDos2ooLDsA` | 9:58 | 悪魔ほむらちゃん 変身シーン＆ボイス一式 |
| `bOpIxyHzUQk` | 9:15 | 限定 暁美ほむら(2凸) 変身シーン＆ボイス一式 |
| `wWzJ60xaONI` | 5:16 | Homura Akemi New Year's Outfit Full Voice |

いずれもほむら単独のボイス集なので**話者分離（diarize）は不要**。
DL は deno 必須（yt-dlp 単体だと n-signature で 403）。

## 前処理

BS-RoFormer 分離 → 2秒無音を挟んで連結（モノラル 44.1kHz・49:14）→
`split_and_transcribe.py`（faster-whisper large-v3）で 495セグ → **12件除去して 483セグ**。

`metadata.csv` の `speaker` 列は**全行 `homura`**（file_name を入れてはいけない。
`docs/REISA_HANDOVER.md` 4.2 参照）。

### 除去した12件（`data/homura/rejected/`）

| 種類 | 件数 | 例 |
|---|---|---|
| Whisper 幻聴「ご視聴ありがとうございました」 | 1 | `seg_00356` |
| リピート幻聴（同一フレーズの反復） | 5 | `seg_00280`（「未知の魔女が…」×3）・`seg_00074` |
| テキスト/音声の不一致（c/s 外れ値） | 5 | `seg_00490`（c/s 60）・`seg_00293`（33.1）・`seg_00081`（1.78） |
| 非日本語混入 | 1 | `seg_00209`（韓国語・英語が混ざった崩れ） |

**リピート幻聴は c/s が高い方向の外れ値になるため、低 c/s だけを見る
`REISA_HANDOVER` 4.1 の手順では取りこぼす**。連続反復フレーズの検出と
c/s > 13 の上側外れ値チェックを併せて行うこと。

品質チェック（文字数 ÷ 有声秒数）は median 7.14 / p10 5.13 / p90 9.42。

## 主要コマンド

```bash
uv run python prepare_manifest.py --dataset audiofolder \
  --data-files "train=data/homura/wavs/*.wav,data/homura/wavs/metadata.csv" \
  --audio-column audio --text-column transcription --speaker-column speaker \
  --output-manifest data/homura/manifest.jsonl --latent-dir data/homura/latents

uv run torchrun --nproc_per_node=1 train.py \
  --config projects/homura/train_homura_v1.yaml \
  --manifest data/homura/manifest.jsonl \
  --output-dir data/lora/homura_v1 \
  --init-checkpoint models/Irodori-TTS-v4.1-Small/model.safetensors
```

## 関連パス

- 学習データ: `data/homura/` / LoRA: `data/lora/homura_v1/`
- テスト台詞: `projects/homura/homura_test_lines.txt`（30行）
- ベースモデル: `models/Irodori-TTS-v4.1-Small/`
