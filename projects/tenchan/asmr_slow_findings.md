# tenchan ASMR「ゆっくり・ぽそぽそ」レシピ検証メモ（確定版）

対象: `tenchan_v2` LoRA（マージ済み checkpoint `data/lora/tenchan_v2/tenchan_v2_best.safetensors`）
＋ 話者条件付き（`use_speaker_condition: true`）。**ゼロショットではなく v2 LoRA を使用**。
ref_wav はアーキ上の話者条件づけ用（LoRAで焼いた表現 + 参照で音色/プロソディ）。
後処理は PosyoPosyo（`posyo asmrify -p posopso` 他）。

---
## ★ 確定レシピ（本番＝「D方式」）
「ゆっくり・愛情・こもった・ぽそぽそ」で母音がクリーンな仕上げ。

**生成（Irodori-TTS, `asmr_final.jsonl`）**
- caption: **通常の愛情caption**（「毛布ごしのようにこもった柔らかい声」「慈しむように」等）。
  ⚠️ **「ウィスパー/吐息多め」captionは使わない**（母音に息ノイズが焼き込まれ濁る。下記参照）。
- ref_wav: **遅い参照 `seg_00607`**（2.70字/秒。既定 seg_00012 は4.58字/秒で速い）。
- text: **母音を伸ばす**（`だいすきだよ`→`だぁいすきだよぉ`）。先頭の `……`→`…` に削減。
- `duration_scale: 1.3`（>1.4は間延びするので1.2〜1.3）。
- `uv run python scripts/batch_infer.py --manifest projects/tenchan/asmr_final.jsonl`

**後処理（PosyoPosyo）**
- `uv run posyo asmrify <gen>.wav -o <out>.wav -p posopso`
- posopso = 近接・こもり(lowpass14k)・小声(peak-3)・僅かな気息(有声部は抑制)。

---
## 検証で分かったこと（なぜこのレシピか）

### 1. ゆっくり化: duration-scaleは本質でない
`--duration-scale` は予測長を線形に伸ばすだけで、`……`(ポーズ)が多いと**伸びた分が無音に吸われる**
（1.8で無音71%＝間延び）。**本質は ①テキストの母音を伸ばす（無音率が下がり実発声が遅くなる）
②遅い参照音声（seg_00607等, `manifest.jsonl`の num_frames/25 と文字数で字/秒算出）**。併用が最良。
発声話速(無音除外の字/秒): 現ref6.31 → 遅ref5.71 → +母音伸ばし4.96。captionの「ゆっくり」単独は弱い。

### 2. ぽそぽそ化: LPC whisperize は不採用（ガサガサ）
声帯振動→ノイズ励起のLPCウィスパー化を実装したが、**ザラつき(rough指標 2.2→3.9)を生む上に
息っぽさも出ない**（共振でピーキー）。この用途では不適。→ `posopso` では不使用。

### 3. 息ノイズ(breathiness)の整形
- 旧実装は `highpass(4kHz)` の白色雑音で**ホワイトノイズ/ヒス**に聞こえた。
  → **気息帯域にバンドパス(既定1.8-6.5k、posopsoは2.5-5k)＋高域ロールオフ**で温かい息に。
- **エンベロープを累乗(env_power)**して発声ピークに集中、無音部のヒスを抑制。
- **`voiced_suppress`(0..1)**: 低域(母音倍音)/高域(気息)比で有声度を判定し、**母音上の息を
  ダッキング**（母音に乗るノイズを低減）。posopsoは1.0。

### 4. ★最重要: ノイズは素材に焼き込まれる
「ウィスパー/吐息多め」captionで生成すると**声そのものに息/ノイズが入る**:
母音上の2-6.5k比率 = 通常caption(exp_D) **0.86%** vs ウィスパーcaption(exp_E) **2.05%**。
後処理の息をいくら切っても素材の分は消えない（E由来は2.3%止まり、D由来は1.26%）。
→ **母音をクリーンに保つには「通常caption生成 + 後処理は息OFF/極少」**。ぽそぽその
こもり・小声・近接は息ノイズ無しでもEQ/レベルで出せる。

---
## 実験アセット（保持）
- `asmr_slow_exp.jsonl` … A/B/C/D（ref・母音伸ばし比較）→ `data/output/tenchan_asmr_exp/exp_A〜D`
- `asmr_whisper_exp.jsonl` … 息ref+ウィスパーcaption → exp_E/F（※母音濁りの反例として保持）
- `asmr_final.jsonl` … 本番3台詞（D方式）→ `data/output/tenchan_asmr_final/`
- 後処理デモ・最終ステム → PosyoPosyo `projects/tenchan_asmr_test/out/`

## 遅い参照候補（tenchan_v2, 字/秒が低い＝ゆっくり）
seg_00607(2.70), seg_00557(2.57), seg_00570(2.82)。既定 seg_00012 は 4.58字/秒。
