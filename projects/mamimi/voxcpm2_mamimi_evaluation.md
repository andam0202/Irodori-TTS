# VoxCPM2 vs Irodori-TTS — mamimi 検証を通じた比較

2026-06-15。mamimi（田中まみみ）音声で VoxCPM2 を全機能検証し、Irodori-TTS（mamimi_v6 LoRA）と比較する。
検証音声は `data/output/voxcpm_test/`、試聴ガイドは `data/output/voxcpm_test/README.md`。

## モデル仕様の比較

| 項目 | Irodori-TTS (mamimi_v6) | VoxCPM2 |
|------|------------------------|---------|
| アーキテクチャ | diffusion AR（プロジェクト独自） | tokenizer-free diffusion AR (LocEnc→TSLM→RALM→LocDiT) |
| バックボーン | プロジェクト独自 | MiniCPM-4 |
| パラメータ | 600M（ベース）+ LoRA | **2B** |
| 言語 | 日本語特化 | **30言語**（日本語含む） |
| 出力レート | プロジェクト設定 | **48kHz**（AudioVAE V2 super-res 内蔵） |
| クローン方式 | Speaker Inversion embed + LoRA 学習 | ゼロショット / Controllable / Ultimate Cloning |
| 感情制御 | キャプション + **絵文字** | control instruction（英中推奨） |
| RTF（RTX 5070 Ti, bf16） | （別途計測） | **~0.35** |
| ライセンス | プロジェクト | Apache-2.0（商用可） |

## mamimi での検証結果

### ゼロショットクローン（学習不要）
VoxCPM2 は mamimi の参照音声（3-10秒）だけで即座にクローン可能（RTF ~0.35）。
Irodori は LoRA 学習済みモデルが必要。VoxCPM2 は「学習せずに mamimi の声を再現」できる点が強力。

### Voice Design（参照音声不要）
VoxCPM2 は `(description)text` 形式でテキスト記述から新規ボイスを作成。
- 英語 description が公式推奨。日本語 description の効きは `02_voice_design/` で検証。
- Irodori は VoiceDesign チェックポイント + キャプションで同等機能。

### Ultimate Cloning（最高精度）
参照音声 + transcript で音声継続ベースのクローン。全声質ニュアンスを再現（VoxCPM1.5 由来）。
→ `05_ultimate_clone/` と `03_clone_control/`（ゼロショット）の聴き比べで精度差を確認。

### 多言語・クロスリンガル
- VoxCPM2: 30言語ネイティブ + クロスリンガル（mamimi の声で英中）。`08_crosslingual/` で検証。
- Irodori: 日本語特化。英語は F5-TTS 別環境（`tools/f5-tts/`）。

### 感情・NSFW 表現
- **Irodori**: キャプション + 絵文字で感情/非言語発声を細かく制御（`run_nsfw_mamimi_v6.sh` で実証）。
- **VoxCPM2**: control instruction（英中推奨）で感情制御。日本語 control の効きは検証項目（`04_clone_style/`, `10_nsfw/`）。

### パラメータ感度（`07_params/`）
- `cfg_value`: 高いほど参照/記述に忠実、低いほど創造的。実用範囲 1.5-3.0。
- `inference_timesteps`: 多いほど品質向上（RTF 増）。実用デフォルト 10、品質重視 20-40。

## LoRA ファインチューン比較（`11_lora/`）

VoxCPM2 に mamimi データ（958件）で LoRA 学習（r=32, 1000 iters）。
- ゼロショットクローン（`03/05`）と LoRA 適用後（`11_lora/`）の品質比較で、学習の効果を測る。
- Irodori mamimi_v6（3000 step 学習）との最終比較は試聴による。

## 役割分担の提案

| 用途 | 推奨 | 理由 |
|------|------|------|
| 日本語 + NSFW/感情表現 | **Irodori-TTS** | キャプション・絵文字制御が強い、LoRA で声質固定 |
| 英語のクリーンな台詞 | **F5-TTS** | 英語話者モデルとして確立済み |
| 多言語・クロスリンガル | **VoxCPM2** | 30言語ネイティブ、1モデルで完結 |
| Voice Design（参照なし新規ボイス） | **VoxCPM2** | description だけで生成 |
| ゼロショットクローン（学習不要） | **VoxCPM2** | 参照音声だけで即クローン |
| 高速リアルタイム | いずれも | いずれも RTF ~0.3 |

## 結論（仮 — 試聴後に更新）

VoxCPM2 は**多言語・Voice Design・ゼロショットクローン**で Irodori を補完する。
mamimi の日本語 + NSFW 表現では、Irodori（LoRA + 絵文字/キャプション制御）が引き続き優位と見られるが、
最終判断は試聴による。VoxCPM2 は「1モデルで多言語 + 即クローン」を実現する点で、
Irodori（日本語特化・学習必須）とは異なる強みを持つ。

## 技術メモ

- VoxCPM2 モデル DL は huggingface_hub の Xet/LFS が本環境で停止したため aria2c 手動DL（CLAUDE.md 参照）。
- 初回推論前に torch.compile ウォームアップ（~4分）。1プロセスで一括実行が効率的。
- 参照音声は 16kHz にリサンプルして渡す（VoxCPM は16kHz入力を想定、48kHz出力）。
