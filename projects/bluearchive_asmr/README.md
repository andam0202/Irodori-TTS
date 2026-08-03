# bluearchive_asmr — ブルアカ3キャラ ASMR / NSFW 音声

`projects/{asuna,karin,toki}/` で学習した v1 LoRA を使って、ASMR・NSFW 音声を
量産するための台本生成プロジェクト。生成は必ず
[`scripts/batch_infer.py`](../../scripts/batch_infer.py)（モデル常駐バッチ）で行う。

## 構成

| パス | 内容 |
|---|---|
| `make_<batch>_jsonl.py` | バッチごとの台本定義（考案 → jsonl 出力）。1本1バッチで追記していく |
| `consolidate_jsonl.py` | バッチ別 jsonl を**種類ごと**に統合し直す |
| `jsonl/<category>.jsonl` | 統合済みマニフェスト（下表） |
| `jsonl/batches/` | 各バッチが出力した元マニフェスト（統合元・regenerable） |

## 種類（統合カテゴリ）

`consolidate_jsonl.py` が out_path の出力先と kind 名から自動分類する。

| カテゴリ | 件数 | 内容 |
|---|---|---|
| `asmr_general` | 168 | 通常 ASMR（耳かき・添い寝・生活音シチュ等） |
| `nonverbal` | 18 | 非言語発声（笑い・ため息・寝息等） |
| `nsfw` | 72 | 喘ぎ・うめき・吐息・誘惑・複合シチュ |
| `nsfw_asmr` | 30 | **ASMR と明示ディレクション**した官能音声（バイノーラル至近距離指定） |
| `nsfw_fera` | 15 | 口内音派生（水音・くぐもった発声主体） |

出力先は `~/Desktop/bluearchive_asmr/<char>/{asmr,nonverbal,nsfw}/<char>_<kind>.wav`。

## 使い方

```bash
# 1) バッチを考案して jsonl を出す
uv run python projects/bluearchive_asmr/make_batch14_nsfw_jsonl.py

# 2) モデル常駐バッチで一括生成（27行で約4分。キャラ切替時のみモデルロード約27秒）
uv run python scripts/batch_infer.py --manifest projects/bluearchive_asmr/jsonl/batches/_batch14_nsfw_all.jsonl

# 3) 種類ごとに統合し直す
uv run python projects/bluearchive_asmr/consolidate_jsonl.py
```

統合済みカテゴリを丸ごと再生成したいときは、そのまま manifest に渡せる:

```bash
uv run python scripts/batch_infer.py --manifest projects/bluearchive_asmr/jsonl/nsfw_asmr.jsonl
```

## 手動で台詞を書いて生成する（manual ワークフロー）

台詞を自分で調整しながら回す用。編集するのは `manual/lines.txt` **1ファイルだけ**。

```
話者 | kind | プリセット | 台詞 [| キャプション上書き [| duration_scale]]
```

```bash
# プリセット一覧（キャプション雛形と既定の話速）
bash projects/bluearchive_asmr/generate_manual.sh --list-presets

# 解決されたキャプション・出力先を確認（生成しない）
bash projects/bluearchive_asmr/generate_manual.sh --dry-run

# 生成
bash projects/bluearchive_asmr/generate_manual.sh
```

- 出力: **`~/Desktop/bluearchive_manual/<話者>_<kind>.wav`**（1フォルダにフラットで並ぶ）。
  話者ごとのフォルダに分けたいときは `--no-flat`、別の場所に出すときは `--outdir <path>`
- キャプションはプリセット名から自動生成される（話者ごとの声質記述が差し込まれる）。
  プリセットで表現しきれないときだけ5列目に直接書いて上書きする
- 同じ `kind` を上書き生成すると seed も変わらないため**同じ音**になる。
  ガチャを引き直したいときは kind を `01_sasoi_b` のように変える（seed は台詞と kind から導出）
- 話者は自動で固めて並べ替えられるので、行の順番は気にしなくてよい
  （モデルの再ロードが話者ごと1回で済む）

## 台本を書くときの実績値

- **キャプションは成人明示**（`20代の女性が…`）で書く。tenchan プロジェクトの実績書式に合わせている
- `duration_scale`: 通常 NSFW は 1.3〜1.35、**水音の連続する行は 1.45**
  （擬音を詰めるとテンポが速くなりすぎる）
- `tail_fade_ms=120` / `tail_pad_out_ms=250` は語尾途切れ対策の既定値（CLAUDE.md 準拠）
- `seed` は `md5(text + speaker + kind + バッチ識別子)` で決定論導出し、再実行時の再現性を保つ
- 台詞をほぼ持たない音（吐息のみ・口内音のみ）は、擬音だけの行にした方が崩れにくい
- キャラ性は口調で作る: アスナ=甘え語尾伸ばし「ご主人様ぁ」、カリン=クールな命令調が崩れる、
  トキ=報告口調が快感で破綻するギャップ
