# projects/diana — PRAGMATA Diana

PRAGMATA の少女型アンドロイド Diana の音声プロジェクト。
日本語 LoRA（v1〜v4）、NSFW 生成、英語版（F5-TTS ゼロショット）、VC 実験（RVC/Seed-VC）を含む。

## 現状

- **日本語 LoRA**: v4 が現行（BS-RoFormer 分離で語尾ノイズ解消、`--tail-fade-ms 120 --tail-pad-out-ms 250` 推奨）。v1〜v3 の経緯は `pragmata_diana_lora_progress.md` 参照
- **英語版**: F5-TTS ゼロショットクローン（`pragmata_diana_en_progress.md` 参照）
- **VC 実験**: `diana_sbv2_rvc_notes.md` 参照

## 主要コマンド

```bash
bash projects/diana/run_test_diana_v4.sh        # 日本語 v4 テスト生成
bash projects/diana/run_test_diana_nsfw_v1.sh   # NSFW テスト生成（diana v4 LoRA 使用）
bash projects/diana/run_test_diana_en.sh        # 英語版 F5-TTS ゼロショット生成
uv run torchrun --nproc_per_node=1 train.py --config projects/diana/train_diana_v4.yaml ...
```

## 関連パス

- 学習データ: `data/diana_v4/`, `data/diana_en/` / LoRA: `data/lora/diana_v4/`
- 生成出力: `data/output/diana_test_v4/`, `data/output/diana_nsfw_v3/`, `data/output/diana_en_test/`

## archive/

世代交代済みランナー（v1〜v3）の無変更保管。当時のチェックポイント・パスを参照
しているためそのままでは動かない。実験再現記録として保存（CLAUDE.md 規約適用外）。
進捗ログ内の `scripts/run_test_diana_v*.sh` 等の旧パス言及は当時の記録としてそのまま。
