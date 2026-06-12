#!/usr/bin/env python3
"""diarization JSON から特定クラスタのセグメントをパディング付きで再切り出しする。

diarize_speakers.py の extract_speaker_audio は turn.end ぴったりで切り出すため、
語尾の自然減衰や無声子音が切れることがある（pragmata_diana cluster_01 で16%が該当）。
このスクリプトは元音声から Demucs を再実行し、対象クラスタのセグメントを
前後パディング付きで再抽出する。近接セグメントの結合（泣き別れ解消）も行う。

使い方:
  uv run python scripts/reextract_segments.py \
    --input "data/input/diana/【観るゲーム】PRAGMATA ⧸ 日本語音声・日本語字幕.wav" \
    --json "data/output/diarization/pragmata_diana/【観るゲーム】PRAGMATA ⧸ 日本語音声・日本語字幕_diarization.json" \
    --cluster-dir data/output/diarization/pragmata_diana/reclustered/cluster_01 \
    --speaker SPEAKER_01 \
    --output-dir data/output/diarization/pragmata_diana/cluster_01_padded
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf

# diarize_speakers.py の torchaudio/torch.load 互換パッチと separate_vocals を再利用
sys.path.insert(0, str(Path(__file__).parent))
from diarize_speakers import separate_vocals  # noqa: E402


def reconstruct_index_mapping(
    segments: list[dict],
    speaker: str,
    min_duration: float,
    sample_rate: int,
) -> dict[int, dict]:
    """extract_speaker_audio と同じ採番ロジックで idx → セグメントの対応を再構築する。"""
    min_samples = int(min_duration * sample_rate)
    mapping: dict[int, dict] = {}
    idx = 0
    for seg in segments:
        if seg["speaker"] != speaker:
            continue
        n_samples = int(seg["end"] * sample_rate) - int(seg["start"] * sample_rate)
        if n_samples < min_samples:
            continue
        idx += 1
        mapping[idx] = seg
    return mapping


def validate_mapping(
    mapping: dict[int, dict],
    cluster_indices: list[int],
    cluster_dir: Path,
    speaker: str,
    tolerance: float = 0.05,
) -> None:
    """クラスタ内ファイルの実尺と JSON 上のセグメント長を突き合わせて採番を検証する。"""
    mismatch = 0
    checked = 0
    for i in cluster_indices:
        path = cluster_dir / f"{speaker}_{i:04d}.wav"
        if i not in mapping or not path.exists():
            mismatch += 1
            continue
        info = sf.info(str(path))
        file_dur = info.frames / info.samplerate
        seg_dur = mapping[i]["end"] - mapping[i]["start"]
        checked += 1
        if abs(file_dur - seg_dur) > tolerance:
            mismatch += 1
    if checked == 0 or mismatch > max(1, checked * 0.01):
        print(
            f"ERROR: 採番検証に失敗しました（照合 {checked} 件中 {mismatch} 件不一致）。\n"
            "  diarization JSON とクラスタディレクトリの対応が取れません。",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"[validate] 採番検証 OK（{checked} 件照合, 不一致 {mismatch} 件）")


def build_spans(
    all_segments: list[dict],
    target_segs: list[dict],
    pad_start: float,
    pad_end: float,
    merge_gap: float,
) -> list[tuple[float, float, int]]:
    """対象セグメントを時系列で結合し、他セグメントにめり込まないパディングを付ける。

    Returns: (start, end, n_merged) のリスト
    """
    target_keys = {(s["start"], s["end"]) for s in target_segs}
    others = sorted(
        [s for s in all_segments if (s["start"], s["end"]) not in target_keys],
        key=lambda s: s["start"],
    )
    targets = sorted(target_segs, key=lambda s: s["start"])

    # 近接する対象セグメントを結合
    merged: list[list[float]] = []
    counts: list[int] = []
    for seg in targets:
        if merged and seg["start"] - merged[-1][1] < merge_gap:
            merged[-1][1] = max(merged[-1][1], seg["end"])
            counts[-1] += 1
        else:
            merged.append([seg["start"], seg["end"]])
            counts.append(1)

    # パディング（他話者セグメントにめり込まない範囲で）
    spans: list[tuple[float, float, int]] = []
    for (s, e), n in zip(merged, counts):
        prev_end = max((o["end"] for o in others if o["end"] <= s), default=0.0)
        next_start = min((o["start"] for o in others if o["start"] >= e), default=float("inf"))
        new_s = max(s - pad_start, prev_end, 0.0)
        new_e = min(e + pad_end, next_start)
        spans.append((new_s, max(new_e, e), n))
    return spans


def main() -> None:
    parser = argparse.ArgumentParser(
        description="diarization セグメントをパディング付きで再切り出しする",
    )
    parser.add_argument("--input", type=Path, required=True, help="元音声ファイル")
    parser.add_argument("--json", type=Path, required=True, help="diarization JSON")
    parser.add_argument(
        "--cluster-dir", type=Path, required=True,
        help="対象クラスタのディレクトリ（SPEAKER_XX_NNNN.wav 形式のファイル名から対象を特定）",
    )
    parser.add_argument("--speaker", type=str, required=True, help="話者ラベル（例: SPEAKER_01）")
    parser.add_argument("--output-dir", type=Path, required=True, help="出力ディレクトリ")
    parser.add_argument("--pad-start", type=float, default=0.10, help="先頭パディング秒（デフォルト: 0.1）")
    parser.add_argument("--pad-end", type=float, default=0.30, help="末尾パディング秒（デフォルト: 0.3）")
    parser.add_argument(
        "--merge-gap", type=float, default=0.30,
        help="このギャップ秒未満の同一クラスタ隣接セグメントは結合（泣き別れ解消。デフォルト: 0.3）",
    )
    parser.add_argument(
        "--min-duration", type=float, default=0.5,
        help="extract_speaker_audio と同じ値にすること（採番再現用。デフォルト: 0.5）",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--vocals-wav", type=Path, default=None,
        help="Demucs 済みボーカル WAV（指定時は Demucs をスキップ）",
    )
    args = parser.parse_args()

    data = json.loads(args.json.read_text(encoding="utf-8"))
    segments = data["segments"]

    # クラスタディレクトリから対象インデックスを取得
    prefix = f"{args.speaker}_"
    cluster_indices = sorted(
        int(p.stem[len(prefix):]) for p in args.cluster_dir.glob(f"{prefix}*.wav")
    )
    if not cluster_indices:
        print(f"ERROR: {args.cluster_dir} に {prefix}*.wav が見つかりません", file=sys.stderr)
        sys.exit(1)
    print(f"[cluster] 対象: {len(cluster_indices)} セグメント")

    # 採番の再構築と検証（diarize_speakers の抽出は元音声のSRベース）
    info = sf.info(str(args.input))
    # 抽出時は demucs 出力 (44.1kHz) を使ったため、44100 で再現する
    mapping = reconstruct_index_mapping(segments, args.speaker, args.min_duration, 44100)
    validate_mapping(mapping, cluster_indices, args.cluster_dir, args.speaker)

    target_segs = [mapping[i] for i in cluster_indices if i in mapping]
    spans = build_spans(segments, target_segs, args.pad_start, args.pad_end, args.merge_gap)
    n_merged = sum(1 for *_, n in spans if n > 1)
    total = sum(e - s for s, e, _ in spans)
    print(f"[spans] {len(spans)} スパン（結合発生: {n_merged} 箇所）, 合計 {total / 60:.1f} 分")

    # ボーカル抽出（推奨: --vocals-wav に BS-RoFormer 分離済みボーカルを渡す）
    if args.vocals_wav and args.vocals_wav.exists():
        vocals_path, sr = args.vocals_wav, sf.info(str(args.vocals_wav)).samplerate
        cleanup = False
        print(f"[vocals] 既存ボーカルを使用: {vocals_path}")
    else:
        print(
            "[warn] --vocals-wav 未指定のため Demucs(htdemucs) で分離します。\n"
            "  htdemucs は語尾に重なる SE を除去しきれず生成音声の語尾が濁る原因になります。\n"
            "  BS-RoFormer で分離したボーカルを --vocals-wav で渡すことを推奨します"
            "（CLAUDE.md「音源分離」参照）。",
            file=sys.stderr,
        )
        vocals_path, sr = separate_vocals(args.input, device=args.device)
        cleanup = True

    # スパンを切り出して書き出し
    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with sf.SoundFile(str(vocals_path)) as f:
            for i, (s, e, _) in enumerate(spans, start=1):
                start_frame = int(s * f.samplerate)
                n_frames = int((e - s) * f.samplerate)
                f.seek(start_frame)
                audio = f.read(n_frames, dtype="float32", always_2d=True)
                if audio.shape[1] > 1:
                    audio = audio.mean(axis=1, keepdims=True)
                out = args.output_dir / f"span_{i:04d}.wav"
                sf.write(str(out), audio, f.samplerate, subtype="PCM_16")
                written += 1
    finally:
        if cleanup and vocals_path.exists():
            vocals_path.unlink()
            print(f"[cleanup] 一時ファイルを削除: {vocals_path}")

    print(f"[done] {written} ファイル → {args.output_dir}")


if __name__ == "__main__":
    main()
