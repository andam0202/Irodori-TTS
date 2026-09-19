# projects/ginko_moan_cast — 腹上トランプ「ぎんこ」の喘ぎ CV 候補

Godot のゲーム **腹上トランプ（belly-section-cards）** のチュートリアル役「ぎんこ」に
どの声を当てるかを決めるための、**喘ぎ（吐息・喘ぎ声）だけ**の比較素材。
このゲームには台詞の音声は無く、声として鳴るのは喘ぎだけ（`game/scripts/game/VOICE.md`）。
主人公ヒロイン「リト」は VAMMoan の `yumi` で確定済み（2026-09-19）。

レシピは `projects/asuna_moan/` と同じ（VaM_Updater `docs/引き継ぎ_性行為の音.md` §1-c）:
テキストは擬音だけ・母音を伸ばす、caption は段ごとの固定ひな型、`duration_scale` 1.3。

## 話者（ベースモデルは各プロジェクトの現行世代に従う）

| 話者 | LoRA | ベース | 参照音声 |
|---|---|---|---|
| sayaka（美樹さやか） | `data/lora/sayaka_v1/checkpoint_best_val_loss_0000300_0.749929.safetensors` | **Irodori-TTS-v4.1-Small** | `data/sayaka/wavs/seg_00056.wav` |
| tenchan（おるすばん 双子ヒロイン） | `data/lora/tenchan_v2/tenchan_v2_best.safetensors` | v4-Small 系 | `data/tenchan_v2/wavs/seg_00023.wav` |
| diana（PRAGMATA Diana） | `data/lora/diana_v4/diana_v4_best.safetensors` | v3-VoiceDesign | `data/diana_v4/wavs/seg_00023.wav` |
| toki（飛鳥馬トキ） | `data/lora/toki_v4/checkpoint_best_val_loss_0000900_0.601365.safetensors` | v4-Small | `data/toki/wavs/seg_00225.wav` |

チェックポイントはマージ済みなので `--checkpoint` だけで動く（ベースは別途渡さない）。

## 手順

```bash
uv run python projects/ginko_moan_cast/make_cast_jsonl.py         # 204 句（4 話者 × 51 句）
uv run python scripts/batch_infer.py --manifest projects/ginko_moan_cast/jsonl/cast.jsonl
uv run python projects/ginko_moan_cast/build_packs.py             # 谷で切り出して素材置き場へ
# VaM_Updater 側で OGG に詰める（ゲームの assets へ）
cd /mnt/c/Users/mao0202/Desktop/VaM_Updater/tools
uv run python godot/voice_pack.py --voice sayaka   # …tenchan / diana / toki
```

- 生成物は `outputs/ginko_moan_cast/<話者>/raw/*.wav`（git 追跡外）。
- 切り出した本は VaM_Updater の `projects/vam-animation/runs/_audio_lib/vammoan/<話者>/`
  （VAMMoan と同じ `m0-01.wav` 命名なので `anatomy_sound.py --voice <話者>` でもそのまま使える）。
- **絶頂（`o`）は切り出さない**（句のまま 1 本）。弱い段（m0/m1）は句の中に谷が少ないので句を多く作ってある。
- `muffled` / `bright` の写しは作っていない（emma / lia / yumi / asuna にしか無い）。
  ゲーム側（`voice.gd`）は写しが無い段を `normal` に落とすので、低ランクの「くぐもった声」は素のまま鳴る。

## 権利

LoRA の元音源は著作物（まどマギ／おるすばん／PRAGMATA／ブルーアーカイブ）。
**配役を決めるための試聴用**であって、このまま製品に載せてよいものではない。
