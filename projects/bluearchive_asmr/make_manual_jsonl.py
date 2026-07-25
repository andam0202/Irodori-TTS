#!/usr/bin/env python3
"""手動編集した台詞ファイルから batch_infer 用 jsonl を作る。

入力は `manual/lines.txt`（パイプ区切り、`#` 行はコメント、空行は無視）:

    話者 | kind | プリセット | 台詞 [| キャプション上書き [| duration_scale]]

例:
    asuna | mimimoto_01 | asmr_whisper | ご主人様ぁ……もっと近くにおいでぇ……♡

キャプションはプリセット名から自動生成される（話者ごとの声質記述が差し込まれる）。
プリセットで表現しきれない場合だけ5列目にキャプションを直接書いて上書きする。

    uv run python projects/bluearchive_asmr/make_manual_jsonl.py --list-presets
    uv run python projects/bluearchive_asmr/make_manual_jsonl.py --dry-run
    uv run python projects/bluearchive_asmr/make_manual_jsonl.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DESKTOP = Path("/mnt/c/Users/mao0202/Desktop/bluearchive_asmr")

# 話者 -> (checkpoint, ref_wav, キャプションに差し込む声質記述)
CHARS = {
    "asuna": (
        REPO / "data/lora/asuna_v1/checkpoint_best_val_loss_0000300_0.764766.safetensors",
        REPO / "data/asuna/wavs/seg_00005.wav",
        "甘えた高めの声で語尾を伸ばす20代の女性",
    ),
    "karin": (
        REPO / "data/lora/karin_v1/checkpoint_best_val_loss_0001200_0.606443.safetensors",
        REPO / "data/karin/wavs/seg_00059.wav",
        "クールで低めの声の20代の女性",
    ),
    "toki": (
        REPO / "data/lora/toki_v1/checkpoint_best_val_loss_0000300_0.642423.safetensors",
        REPO / "data/toki/wavs/seg_00225.wav",
        "淡々とした報告口調の20代の女性",
    ),
}

# プリセット名 -> (キャプション雛形（{voice} に声質記述が入る）, 既定の duration_scale)
PRESETS: dict[str, tuple[str, float]] = {
    # --- ASMR 明示（バイノーラル・至近距離を指定する） ---
    "asmr_whisper": (
        "ASMR音声。バイノーラル至近距離で{voice}が極小音量で囁く。"
        "息が多く混じり、非常にゆっくり、こもった柔らかい残響。",
        1.40,
    ),
    "asmr_ear": (
        "ASMR音声。耳のすぐそばで{voice}が湿った吐息まじりに囁く。"
        "息がマイクに触れる近さ、こもった響き、非常にゆっくり。",
        1.40,
    ),
    "asmr_breath": (
        "ASMR音声。台詞をほぼ含まず、バイノーラル至近距離で{voice}の吐息と呼吸音だけを聴かせる。"
        "囁きより小さく、湿った息がマイクに触れる。",
        1.45,
    ),
    "asmr_kuchi": (
        "ASMR音声。口内音・リップノイズ主体で、台詞をほぼ含まない。"
        "{voice}の湿った音と微かな吐息のみ、至近距離のバイノーラル収録。",
        1.45,
    ),
    "asmr_sleep": (
        "ASMR音声。寝かしつけるような極小音量で{voice}が甘く囁く。"
        "息が多く混じり、語尾がふんわり溶ける、非常にゆっくり。",
        1.45,
    ),
    # --- NSFW ---
    "aegi_soft": (
        "{voice}が快感で小さく甘く喘いでいる。息が弾み、語尾が甘く伸びて溶ける、"
        "耳元でこもった柔らかい響き。",
        1.30,
    ),
    "aegi_hard": (
        "{voice}が激しく短い間隔で喘いでいる。息継ぎが忙しなく、声が上ずって連続し、抑えが効かない。",
        1.25,
    ),
    "gaman": (
        "{voice}が声を必死に押し殺そうとして漏れる、くぐもったうめき混じりの喘ぎ。"
        "息を詰め、途切れ途切れに震える。",
        1.35,
    ),
    "toiki": (
        "{voice}が耳元のごく近い距離で吐く、湿った吐息。"
        "ほとんど声にならない息だけの囁き、ゆっくり。",
        1.40,
    ),
    "umeki": (
        "{voice}が切なさに耐えかねて漏らす低いうめき声。喉の奥でくぐもり、息が荒く、語尾が震える。",
        1.35,
    ),
    "sasoi": (
        "{voice}が、しなだれかかるように誘っている。とろけた声で、ゆっくりねだるように囁く。",
        1.35,
    ),
    "jirashi": (
        "{voice}が焦らして楽しむように囁く。息を吹きかけるような近さで、"
        "意地悪な余裕を含んでゆっくり。",
        1.35,
    ),
    "zecchou": (
        "{voice}が限界を迎えて声が跳ね上がる。息が完全に乱れ、声が震えて途切れ、"
        "最後は荒い呼吸だけが残る。",
        1.25,
    ),
    "jigo": (
        "{voice}が余韻に浸る気だるげな声。呼吸がまだ整わないまま、"
        "満ち足りた柔らかさを含んでゆっくり囁く。",
        1.40,
    ),
    "mizuoto": (
        "{voice}の口淫の水音。口が塞がった状態のくぐもった発声で、"
        "粘つく吸引音と鼻から抜ける荒い息が連続する。台詞はほぼ含まない。",
        1.45,
    ),
    "jikkyou": (
        "{voice}が耳元で相手の反応を実況しながら追い詰めるように囁く。"
        "余裕のある口調で、息が多く、ゆっくり間を置く。",
        1.35,
    ),
}

TAIL_FADE_MS = 120
TAIL_PAD_OUT_MS = 250


def seed_of(speaker: str, kind: str, text: str) -> int:
    return int(hashlib.md5(f"{speaker}/{kind}/{text}".encode()).hexdigest()[:8], 16) % 100000


def parse_lines(path: Path) -> list[dict]:
    """パイプ区切りの台詞ファイルを読む。エラーは行番号つきで全件報告する。"""
    records: list[dict] = []
    errors: list[str] = []
    seen: dict[tuple[str, str], int] = {}

    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        cols = [c.strip() for c in line.split("|")]
        if len(cols) < 4:
            errors.append(f"{path.name}:{lineno}: 列が足りません（最低4列: 話者|kind|プリセット|台詞）")
            continue
        speaker, kind, preset, text = cols[0], cols[1], cols[2], cols[3]
        caption_override = cols[4] if len(cols) > 4 and cols[4] else None
        scale_raw = cols[5] if len(cols) > 5 and cols[5] else None

        if speaker not in CHARS:
            errors.append(f"{path.name}:{lineno}: 未知の話者 '{speaker}'（{'/'.join(CHARS)}）")
            continue
        if preset not in PRESETS:
            errors.append(f"{path.name}:{lineno}: 未知のプリセット '{preset}'（--list-presets で一覧）")
            continue
        if not kind:
            errors.append(f"{path.name}:{lineno}: kind が空です（出力ファイル名になります）")
            continue
        if not text:
            errors.append(f"{path.name}:{lineno}: 台詞が空です")
            continue
        if (speaker, kind) in seen:
            errors.append(
                f"{path.name}:{lineno}: 話者 {speaker} の kind '{kind}' が "
                f"{seen[(speaker, kind)]} 行目と重複しています（出力が上書きされます）"
            )
            continue
        seen[(speaker, kind)] = lineno

        template, default_scale = PRESETS[preset]
        scale = default_scale
        if scale_raw is not None:
            try:
                scale = float(scale_raw)
            except ValueError:
                errors.append(f"{path.name}:{lineno}: duration_scale が数値ではありません '{scale_raw}'")
                continue

        _, _, voice = CHARS[speaker]
        records.append(
            {
                "speaker": speaker,
                "kind": kind,
                "text": text,
                "caption": caption_override or template.format(voice=voice),
                "duration_scale": scale,
            }
        )

    if errors:
        for e in errors:
            print(f"[error] {e}", file=sys.stderr)
        raise SystemExit(f"{len(errors)} 件のエラーがあります。修正してから再実行してください。")
    return records


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lines", type=Path, default=here / "manual/lines.txt", help="台詞ファイル")
    ap.add_argument("--out-jsonl", type=Path, default=here / "jsonl/manual.jsonl", help="出力マニフェスト")
    ap.add_argument("--outdir", type=Path, default=DESKTOP, help="wav 出力先のルート")
    ap.add_argument("--subdir", default="manual", help="話者ディレクトリ下のサブフォルダ名")
    ap.add_argument("--list-presets", action="store_true", help="プリセット一覧を表示して終了")
    ap.add_argument("--dry-run", action="store_true", help="jsonl を書かずに解決結果を表示")
    args = ap.parse_args()

    if args.list_presets:
        width = max(len(p) for p in PRESETS)
        for name, (template, scale) in PRESETS.items():
            print(f"{name:{width}s}  scale={scale}  {template.format(voice='<話者の声質>')}")
        return 0

    if not args.lines.exists():
        print(f"台詞ファイルがありません: {args.lines}", file=sys.stderr)
        return 1

    records = parse_lines(args.lines)
    if not records:
        print(f"{args.lines} に有効な行がありません（# はコメント行）", file=sys.stderr)
        return 1

    out_lines = []
    for rec in records:
        ckpt, ref, _ = CHARS[rec["speaker"]]
        out_path = args.outdir / rec["speaker"] / args.subdir / f"{rec['speaker']}_{rec['kind']}.wav"
        out_lines.append(
            {
                "text": rec["text"],
                "caption": rec["caption"],
                "checkpoint": str(ckpt),
                "ref_wav": str(ref),
                "out_path": str(out_path),
                "seed": seed_of(rec["speaker"], rec["kind"], rec["text"]),
                "duration_scale": rec["duration_scale"],
                "tail_fade_ms": TAIL_FADE_MS,
                "tail_pad_out_ms": TAIL_PAD_OUT_MS,
            }
        )

    if args.dry_run:
        for rec, obj in zip(records, out_lines):
            print(f"--- {rec['speaker']}_{rec['kind']}  (scale={rec['duration_scale']})")
            print(f"  text   : {rec['text']}")
            print(f"  caption: {rec['caption']}")
            print(f"  out    : {obj['out_path']}")
        print(f"\n{len(out_lines)} 行（--dry-run のため書き出していません）")
        return 0

    # 話者ごとに固めるとモデルの再ロードが1回で済む（batch_infer は checkpoint をキャッシュする）
    order = list(CHARS)
    out_lines.sort(key=lambda o: order.index(Path(o["out_path"]).parts[-3]))

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.out_jsonl.write_text(
        "\n".join(json.dumps(o, ensure_ascii=False) for o in out_lines) + "\n", encoding="utf-8"
    )
    per_speaker: dict[str, int] = {}
    for obj in out_lines:
        per_speaker[Path(obj["out_path"]).parts[-3]] = per_speaker.get(Path(obj["out_path"]).parts[-3], 0) + 1
    summary = " / ".join(f"{k}:{v}" for k, v in per_speaker.items())
    print(f"{len(out_lines)} 行 -> {args.out_jsonl}  （{summary}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
