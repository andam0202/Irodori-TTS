# projects/multilingual — 多言語ゲームボイス検証（Qwen3-TTS / VoxCPM2）

ゲーム制作用の外国語ボイス（子供・中年男性、SFW/NSFW、吐息・うめき等の非言語音）を
生成・試聴評価するプロジェクト。

## エンジンの役割分担（2026-07 策定、詳細は CLAUDE.md）

- **Qwen3-TTS**（Apache-2.0・商用可・EN/RU/ZH/KO/ES/PT）: NSFW セリフ・喘ぎ含む主力
- **VoxCPM2**（30言語）: Qwen3-TTS 非対応の Swahili/Arabic を補完
- 日本語 NSFW = Irodori-TTS、英語クリーン台詞 = F5-TTS

## 構成

- `_tts_testgen.py` — 共通ランナー（GenSpec/build_specs/run_qwen_design/run_voxcpm。
  各 test_* から兄弟 import されるので同一ディレクトリに置くこと）
- `test_qwen3tts_{nsfw,man,child_sfw,child_sighs,child_groans,child_persona}.py`
- `test_voxcpm_{man,child_sighs,child_groans}.py`（Swahili/Arabic）

## 主要コマンド

```bash
bash scripts/qwen3tts.sh test                  # = test_qwen3tts_nsfw.py（EN/RU/ZH/KO）
bash scripts/qwen3tts.sh python projects/multilingual/test_qwen3tts_man.py --dry-run
bash scripts/voxcpm.sh python projects/multilingual/test_voxcpm_man.py
uv run pytest tests/test_tts_testgen.py        # 共通ランナーの回帰テスト
```

## 関連パス

- 生成出力: `data/output/qwen3tts_*`, `data/output/voxcpm_{man,child_*}`（manifest.json 付き）
- エンジン環境: `tools/qwen3-tts/`, `tools/voxcpm/`（ラッパーは `scripts/qwen3tts.sh` / `scripts/voxcpm.sh`）
