# 部屋の声（台本 v2）のラウンドネス基準

Godot に書き出す ogg は、1 本ずつ次の基準を満たすようにする（2026-09-25 ユーザー指示）。
実装は `voices_v2.py`（`LOUDNESS_TARGETS` ほか）、適用と実測は `build_voices.py v2-export`。

## 目標値

| 行の種類 | 判定 | Integrated 目標 | 許容 |
|---|---|---|---|
| 吐息 | `kind = "breath"` | −20 LUFS | ±1 LU |
| 弱い喘ぎ | `level = "soft"` | −18 LUFS | ±1 LU |
| 中くらいの喘ぎ | `level = "mid"` | −16 LUFS | ±1 LU |
| 強い喘ぎ | `level = "hard"` | −14 LUFS | ±1 LU |
| 絶頂 | `climax = true` | −14 LUFS | ±1 LU |

- 基準は −16 LUFS。弱い → 中 → 強いの順に自然に大きく聞こえるよう、段ごとに 2 LU ずつ差をつけた。
  吐息は −16 だと喘ぎより目立つので −20 にした。
- **True Peak は全行 −1 dBTP 以下**（ogg の実測値で判定する）。

## 処理

1. 前後の無音を切る（−45 dBFS より静かな部分。前 50 ms・後 150 ms は残す）。
2. 目標のラウドネスへゲインを合わせ、ピークの頭だけを tanh で丸める（wav の上限は −1.5 dBTP）。
   丸めると少し小さくなるので、目標との差が 0.2 LU 以内になるまでゲインを合わせ直す。
3. ogg（vorbis q3・44.1 kHz・モノラル）にしてから `ffmpeg ebur128=peak=true` で実測する。
   - ラウドネスだけがずれた場合は、ogg に変換するときのゲインで補正する。
   - vorbis 化で True Peak が −1 dBTP を超えた場合は、wav の上限を −2.5 → −3.5 → −5 dBTP と
     下げて作り直す。
4. 実測の結果は `outputs/dorei_shokan_voices/v2/loudness_report.{json,md}` に書き出す。
   セルごと・種類ごとの平均と、基準から外れたファイルの一覧が入る。

## サイズの目安

vorbis q3 で 1 キャラ（56 行）あたり約 2 MB（inu:standard の実測は 2.0 MB）。
