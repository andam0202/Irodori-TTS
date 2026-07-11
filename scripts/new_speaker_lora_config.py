"""新話者 LoRA 学習 config を実績レシピから生成する.

話者別 config（train_diana_v*/train_mamimi_v6/train_tenchan_v1/train_nurse_t_v1 等）は
実質 wandb_project/wandb_run_name（と必要なら step 数）しか違わないコピーだった。
本スクリプトは実績テンプレート（デフォルト: configs/train_mamimi_v6.yaml）を
テキストレベルで差し替えて configs/train_<speaker>_v<N>.yaml を生成する。
コメントを保持するためテキスト置換方式とし、生成後に yaml ロードで
「上書きキー以外はテンプレートとセマンティック同一」であることを自己検証する。

uv run python scripts/new_speaker_lora_config.py --speaker tenchan --version 2
uv run python scripts/new_speaker_lora_config.py --speaker foo --version 1 --max-steps 4000 --note "データ500件"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TEMPLATE = PROJECT_DIR / "configs" / "train_mamimi_v6.yaml"

# テンプレート内で置換対象となる実値（train_mamimi_v6.yaml のもの）
TEMPLATE_WANDB_PROJECT = "irodori-tts-mamimi"
TEMPLATE_WANDB_RUN_NAME = "mamimi-v6"


def replace_scalar(text: str, key: str, new_value: str | int) -> str:
    """`  key: value` 行の値を置換する（トップレベルでないインデント付きキー）."""
    pattern = re.compile(rf"^(\s+{re.escape(key)}:\s*).*$", flags=re.MULTILINE)
    if not pattern.search(text):
        raise SystemExit(f"error: テンプレートにキー {key!r} が見つからない")
    return pattern.sub(rf"\g<1>{new_value}", text, count=1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a speaker LoRA training config from the proven recipe"
    )
    parser.add_argument("--speaker", required=True, help="話者名（例: tenchan）")
    parser.add_argument("--version", type=int, required=True, help="バージョン番号（例: 1）")
    parser.add_argument(
        "--template", default=str(DEFAULT_TEMPLATE),
        help=f"テンプレート config（デフォルト: {DEFAULT_TEMPLATE.name}）",
    )
    parser.add_argument("--wandb-project", default=None,
                        help="デフォルト: irodori-tts-<speaker>")
    parser.add_argument("--wandb-run-name", default=None,
                        help="デフォルト: <speaker>-v<version>")
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--stable-steps", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--note", default=None,
                        help="ヘッダーに残すメモ（データ件数・注意点など）")
    parser.add_argument("--output", default=None,
                        help="デフォルト: configs/train_<speaker>_v<version>.yaml")
    parser.add_argument("--force", action="store_true", help="既存ファイルを上書きする")
    args = parser.parse_args()

    speaker = args.speaker.strip().lower().replace("-", "_")
    wandb_project = args.wandb_project or f"irodori-tts-{speaker.replace('_', '-')}"
    wandb_run_name = args.wandb_run_name or f"{speaker.replace('_', '-')}-v{args.version}"
    out_path = Path(args.output) if args.output else (
        PROJECT_DIR / "configs" / f"train_{speaker}_v{args.version}.yaml"
    )
    if out_path.exists() and not args.force:
        raise SystemExit(f"error: {out_path} は既に存在する（--force で上書き）")

    template_path = Path(args.template)
    text = template_path.read_text(encoding="utf-8")

    text = text.replace(
        f"wandb_project: {TEMPLATE_WANDB_PROJECT}", f"wandb_project: {wandb_project}"
    ).replace(
        f"wandb_run_name: {TEMPLATE_WANDB_RUN_NAME}", f"wandb_run_name: {wandb_run_name}"
    )
    overrides: dict[str, int] = {}
    for key in ("warmup_steps", "stable_steps", "max_steps"):
        value = getattr(args, key)
        if value is not None:
            text = replace_scalar(text, key, value)
            overrides[key] = value

    header_lines = [
        f"# train_{speaker}_v{args.version}.yaml — {speaker} LoRA 学習設定",
        f"# {template_path.name} の実績レシピから scripts/new_speaker_lora_config.py で生成",
    ]
    if args.note:
        header_lines.append(f"# {args.note}")
    text = "\n".join(header_lines) + "\n" + text

    # 自己検証: 上書きキー以外はテンプレートとセマンティック同一であること
    generated = yaml.safe_load(text)
    expected = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    expected["train"]["wandb_project"] = wandb_project
    expected["train"]["wandb_run_name"] = wandb_run_name
    for key, value in overrides.items():
        expected["train"][key] = value
    if generated != expected:
        raise SystemExit("error: 生成結果がテンプレート+上書きと一致しない（テンプレート構造を確認）")

    out_path.write_text(text, encoding="utf-8")
    print(f"generated: {out_path}")
    print(f"  wandb_project : {wandb_project}")
    print(f"  wandb_run_name: {wandb_run_name}")
    for key, value in overrides.items():
        print(f"  {key}: {value}")
    print(f"train: uv run torchrun --nproc_per_node=1 train.py --config {out_path} ...")


if __name__ == "__main__":
    main()
