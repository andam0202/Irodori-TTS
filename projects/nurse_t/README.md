# projects/nurse_t — VOICEVOX ナースロボ＿タイプＴ

VOICEVOX「ナースロボ＿タイプＴ」（ノーマルスタイル）の合成音声を学習データにした
LoRA プロジェクト。ASMR スタイル版の config も含む。

## 現状

- **nurse_t v1**: ITA コーパス422文 + ROHAN4600 先頭578文（計1001件）を
  `scripts/generate_voicevox_dataset.py` で一括合成して学習
- **asmr v1**: ASMR スタイル用 config（`train_nurse_t_asmr_v1.yaml`）

## 主要コマンド

```bash
uv run python scripts/generate_voicevox_dataset.py ...   # VOICEVOX API でデータ合成（汎用ツール）
bash projects/nurse_t/run_test_nurse_t_v1.sh             # テスト生成
uv run torchrun --nproc_per_node=1 train.py --config projects/nurse_t/train_nurse_t_v1.yaml ...
```

## 関連パス

- 学習データ: `data/nurse_t/`, `data/nurse_t_asmr/` / LoRA: `data/lora/nurse_t_v1/`
- 生成出力: `data/output/nurse_t_v1_test/`
