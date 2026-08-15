# luca_song — 斑鳩ルカ「歌声由来」話者 LoRA + コメティック SVC 一式

作成: 2026-08-16 / 素材: シャニマス コメティックのソロ歌唱楽曲（YouTube 公開音源）

このプロジェクトは2つの実験をまとめたもの。

1. **歌声データだけで Irodori-TTS の話者 LoRA を作るとどうなるか**（luca_song v4）
2. **既存曲を別のキャラの声で歌わせる**（Seed-VC / RVC による SVC。3人分のモデルを学習）

⚠ 生成物・学習モデルは権利者の許諾が無い限り**公開・配布しない**。手元の私的実験に限る。

---

## 1. 成果物

| 種別 | パス |
|---|---|
| 話者 LoRA（マージ済み・推論用） | `data/lora/luca_song_v4/checkpoint_best_val_loss_0000900_1.047985.safetensors` |
| LoRA アダプタ | `data/lora/luca_song_v4/checkpoint_best_val_loss_{0000300,0000900}_*/` |
| 学習データ | `data/luca_song/wavs/`（240セグ）・`manifest.jsonl`・`latents/` |
| 除外セグメント | `data/luca_song/rejected/`（7件・消さずに保管） |
| 学習 config | `projects/luca_song/train_luca_song_v4.yaml` |
| セリフ生成テスト（8件） | `data/output/luca_song_test/` |
| RVC モデル3人分 | `tools/applio/logs/{luca,hana,haruki}_rvc/`（各 epoch 300 と 275 + index） |
| カバー音源・試聴ページ | `data/output/luca_svc/`（`listen.html` をブラウザで開く） |

## 2. 素材（YouTube・ソロ Ver.）

CoMETIK SOLO COLLECTION 系の**ソロ版**を使う（ユニット版は他メンバーが混ざるため不可）。

| 話者 | 曲数 | 尺 |
|---|---|---|
| 斑鳩ルカ | 9 | 約35分 |
| 鈴木羽那 | 11 | 約43分（オリジナルソロ曲を含む） |
| 郁田はるき | 10 | 約39分 |

- ダウンロードは `.venv/bin/yt-dlp`（PATH に `~/.deno/bin` を前置しないと n-sig で 403）
- **403 が散発するのでリトライループを必ず入れる**（2〜5回目で通る）
- 未収集のルカ ソロ版がまだある（「平行線の美学」等）。モデルを鍛え直す余地あり

## 3. 前処理

```bash
# 1) ボーカル抽出（歌モノは MelBand Roformer Vocals。CLAUDE.md 参照）
uv run --project tools/audio-separator audio-separator data/input/cometik/<who>/*.wav \
  -m vocals_mel_band_roformer.ckpt --model_file_dir tools/audio-separator/models \
  --output_dir data/output/luca_svc/<who>_vocals --output_format WAV --single_stem Vocals

# 2) TTS 用: 2秒無音を挟んで連結 → セグメント化 + 文字起こし
#    （無音なし連結は生成末尾のゴミ音声の原因。CLAUDE.md 参照）
uv run python scripts/split_and_transcribe.py \
  --input-mp3 data/input/luca_song/luca_song_concat.wav --output-dir data/luca_song/wavs

# 3) manifest + DACVAE latent
uv run python prepare_manifest.py --dataset audiofolder \
  --data-files "train=data/luca_song/wavs/*.wav,data/luca_song/wavs/metadata.csv" \
  --audio-column audio --text-column transcription --speaker-column speaker \
  --output-manifest data/luca_song/manifest.jsonl --latent-dir data/luca_song/latents
```

### 歌詞の文字起こしについて
Whisper は歌唱の聞き取り精度がセリフより落ちる。247セグのうち、
**「文字数 ÷ 有声秒数」が中央値(3.62)の 0.35 倍未満**の7件を外れ値として除外し 240セグにした
（`data/luca_song/rejected/` に保管）。テキストが音声に対して極端に短いものが対象。

`metadata.csv` の `speaker` 列は**全行 `luca_song`** にすること（reisa の教訓。
1発話＝1話者にすると v4-Small の長尺参照学習が効かなくなる）。

## 4. 学習

```bash
uv run python scripts/new_speaker_lora_config.py \
  --speaker luca_song --version 4 --template configs/train_v4_small_lora_speaker.yaml

uv run torchrun --nproc_per_node=1 train.py \
  --config projects/luca_song/train_luca_song_v4.yaml \
  --manifest data/luca_song/manifest.jsonl \
  --output-dir data/lora/luca_song_v4 \
  --init-checkpoint models/Irodori-TTS-v4-Small/model.safetensors
```

vast.ai RTX 5090 で 3000step / 19分25秒。**val_loss 最良は step900（1.048）**で、
step300 が 1.073、以降は 1.06〜1.13 を上下して大きくは下がらない。採用は step900。

### RVC（Applio）側
`tools/applio/` で `preprocess → extract → train`（48kHz / batch 16 / 300 epoch）。
5090 で1人あたり約45〜60分。3人の loss 最良は はるき16.77 / ルカ19.18 / 羽那21.02。
落とし穴（`assets/config.json` 不在で重み書き出しが失敗する等）は CLAUDE.md にまとめた。

## 5. 生成

```bash
uv run python projects/kaho/make_test_jsonl.py \
  --checkpoint data/lora/luca_song_v4/checkpoint_best_val_loss_0000900_1.047985.safetensors \
  --ref-wav data/luca_song/wavs/seg_00037.wav \
  --lines <lines>.txt --out-dir data/output/luca_song_test --name luca_song \
  > projects/luca_song/luca_song_test.jsonl

uv run python scripts/batch_infer.py --manifest projects/luca_song/luca_song_test.jsonl
```

参照音声は `data/luca_song/wavs/seg_00037.wav`（6.5秒）を使用。学習データが全て歌唱なので、
**参照に選ぶセグメントの歌い方が生成物の抑揚に強く出る**点に注意。

## 関連ドキュメント
- `CLAUDE.md`（歌声変換 SVC / 音源分離の使い分け）
- `docs/DTM_GUIDE.md`（DAW 選定・無料プラグイン・ミックス方針）
- `docs/REAPER_HANDS_ON.md`（REAPER の実操作）
