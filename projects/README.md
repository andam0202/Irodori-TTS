# projects/ — プロジェクト索引

話者・用途ごとのプロジェクト（スクリプト・config・ドキュメント・archive を各フォルダに集約）。
共有のデータ前処理パイプライン・エンジンラッパー・共通ヘルパーは `scripts/`、
話者非依存の汎用 config は `configs/`、汎用ドキュメントは `docs/` にある。

| プロジェクト | 内容 |
|---|---|
| [diana/](diana/) | PRAGMATA Diana — 日本語 LoRA v1〜v4 / NSFW / 英語版(F5-TTS) / VC 実験 |
| [mamimi/](mamimi/) | 田中摩美々 — LoRA v5/v6 / NSFW / 絵文字制御検証 / VoxCPM2 評価・LoRA |
| [tenchan/](tenchan/) | おるすばん双子ヒロイン — 単一声 LoRA + キャプション演じ分け |
| [nurse_t/](nurse_t/) | VOICEVOX ナースロボ＿タイプＴ — 合成データ LoRA（ノーマル/ASMR） |
| [kaho/](kaho/) | 小宮果穂 — LoRA v1（韓国コミュ動画由来） |
| [asuna/](asuna/) | 一之瀬アスナ — LoRA v1（ブルアカ ボイス集由来） |
| [karin/](karin/) | 角楯カリン — LoRA v1（ブルアカ ボイス集由来） |
| [toki/](toki/) | 飛鳥馬トキ — LoRA v1（ブルアカ ボイス集由来） |
| [reisa/](reisa/) | 宇沢レイサ — LoRA v4（ブルアカ ボイス集由来） |
| [kaai_yuki/](kaai_yuki/) | 歌愛ゆき — LoRA v4（AivisSpeech 合成データ由来・準備中） |
| [luca_song/](luca_song/) | 斑鳩ルカ 歌声由来 LoRA v4 + コメティック3人の SVC(RVC/Seed-VC) |
| [bluearchive_asmr/](bluearchive_asmr/) | 上記3キャラの ASMR / NSFW 音声量産（台本生成 + 種類別統合） |
| [multilingual/](multilingual/) | 多言語ゲームボイス検証 — Qwen3-TTS(EN/RU/ZH/KO/ES/PT) + VoxCPM2(SW/AR) |

新話者プロジェクトの始め方:

```bash
uv run python scripts/new_speaker_lora_config.py --speaker <name> --version 1
# → projects/<name>/train_<name>_v1.yaml が実績レシピから生成される
```

各プロジェクトの `archive/` は世代交代済みスクリプトの無変更保管場所
（当時のパス前提のためそのままでは動かない。CLAUDE.md 規約適用外）。
