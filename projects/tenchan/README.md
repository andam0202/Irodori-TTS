# projects/tenchan — おるすばん 双子ヒロイン

「おるすばん」の双子ヒロイン（ちひろ/ゆき）用音声プロジェクト。
声は単一の LoRA（tenchan）で、双子の演じ分けはキャプションで行う。

## 現状

- **tenchan v1**: てんしラジオ12話から話者分離 → cluster_00 再抽出 627件（58.3分）で学習。
  レシピは mamimi_v6 を踏襲（データが少なめのため過学習が見えたら max_steps 3000〜4000 に）

## 主要コマンド

```bash
bash projects/tenchan/run_test_tenchan_v1.sh   # 双子演じ分けテスト生成
uv run torchrun --nproc_per_node=1 train.py --config projects/tenchan/train_tenchan_v1.yaml ...
```

## 関連パス

- 学習データ: `data/tenchan/` / LoRA: `data/lora/tenchan_v1/`
- 生成出力: `data/output/tenchan_test_v1/`

## archive/

`_extract_tenchan_cluster0.py`: cluster_00 セグメントを 44.1kHz ボーカルから
再抽出した実行済みワンオフ（無変更保管・CLAUDE.md 規約適用外）。
