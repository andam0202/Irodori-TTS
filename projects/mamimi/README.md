# projects/mamimi — 田中摩美々

シャニマス 田中摩美々の音声プロジェクト。Irodori-TTS LoRA（v5/v6）、NSFW 生成、
絵文字スタイル制御の検証、VoxCPM2 との比較評価・VoxCPM LoRA 実験を含む。

## 現状

- **Irodori LoRA**: v6 が現行（diana_v4 同等前処理・データ1026件・3000step 学習完了、
  best/final 生成済み）。学習手順は `MAMIMI_TRAINING.md` 参照
- **VoxCPM2 評価**: `voxcpm2_mamimi_evaluation.md` 参照（役割分担の結論込み）
- **Beatrice VC**: リアルタイム声質変換は tools/beatrice-trainer 側（10000step 完了）

## 主要コマンド

```bash
bash projects/mamimi/run_test_mamimi_v6.sh      # v6 テスト生成
bash projects/mamimi/run_nsfw_mamimi_v6.sh      # NSFW テスト30本生成
bash scripts/voxcpm.sh test                     # VoxCPM2 全機能検証（mamimi 音声）
uv run python projects/mamimi/prepare_voxcpm_mamimi.py   # VoxCPM 学習用 JSONL 変換
uv run torchrun --nproc_per_node=1 train.py --config projects/mamimi/train_mamimi_v6.yaml ...
```

※ `train_mamimi_v6.yaml` は `scripts/new_speaker_lora_config.py` の実績テンプレート。

## 関連パス

- 学習データ: `data/mamimi_v6/` / LoRA: `data/lora/mamimi_v6/`
- 生成出力: `data/output/mamimi_test_v6/`, `data/output/mamimi_nsfw_v6/`, `data/output/voxcpm_test/`

## archive/

旧世代（v5_vd3 ランナー、絵文字検証の run.sh/run_nsfw.sh/generate_mamimi.sh 3本セット）の
無変更保管。当時のパス前提のためそのままでは動かない（CLAUDE.md 規約適用外）。
