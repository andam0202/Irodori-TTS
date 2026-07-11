"""tenchan cluster_00（テンちゃん）のセグメントを、BS-RoFormer分離済みの
44.1kHzボーカルから話ごとに再抽出する（tenchan 前処理用の一時スクリプト）。

背景:
- diarization は --no-separate で pyannote の 16kHz 音声からセグメントを切り出したため
  低品質・BGM残留。assignments.csv には各 cluster_00 セグメントの元パス
  （話ディレクトリ/SPEAKER_XX/NNNN.wav）が完全に残っている。
- ここでは各話の diarization JSON から採番を sr=16000 で厳密再現して時間範囲を復元し、
  BS-RoFormer で分離した 44.1kHz ボーカルから前後パディング付きで再切り出す。
- reextract_segments.py の reconstruct_index_mapping / build_spans を再利用する。
"""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from reextract_segments import build_spans, reconstruct_index_mapping  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
CSV = PROJECT / "data/output/diarization/tenchan/reclustered/assignments.csv"
VOCALS_DIR = PROJECT / "data/output/separation/tenchan/vocals"
OUT_DIR = PROJECT / "data/tenchan/segments_44k"
TARGET_CLUSTER = "0"

# extract_speaker_audio が --no-separate 時に使った out_sr（pyannote lq）
EXTRACT_SR = 16000
MIN_DURATION = 0.5
PAD_START = 0.10
PAD_END = 0.30
MERGE_GAP = 0.30
DUR_TOLERANCE = 0.05


def parse_source(path: str) -> tuple[str, str, int]:
    """'.../<ep_dir>/speakers/SPEAKER_XX/NNNN.wav' → (ep_dir, speaker, index)。"""
    p = Path(path)
    index = int(p.stem)
    speaker = p.parent.name
    ep_dir = p.parent.parent.parent.name  # speakers/ の親
    return ep_dir, speaker, index


def find_vocals(ep_stem: str) -> Path | None:
    matches = sorted(VOCALS_DIR.glob(f"{ep_stem}*Vocals*.wav"))
    return matches[0] if matches else None


def main() -> None:
    # cluster_00 を (ep_dir -> speaker -> [(index, csv_dur)]) に整理
    by_ep: dict[str, dict[str, list[tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))
    with CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["cluster"] != TARGET_CLUSTER:
                continue
            ep_dir, speaker, index = parse_source(r["source"])
            by_ep[ep_dir][speaker].append((index, float(r["duration_sec"])))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    diar_root = PROJECT / "data/output/diarization/tenchan"

    global_idx = 0
    total_dur = 0.0
    grand_validate = {"checked": 0, "mismatch": 0}

    for ep_dir in sorted(by_ep):
        json_path = diar_root / ep_dir / f"{ep_dir}_diarization.json"
        if not json_path.exists():
            print(f"ERROR: JSON なし: {json_path}", file=sys.stderr)
            sys.exit(1)
        import json

        segments = json.loads(json_path.read_text(encoding="utf-8"))["segments"]

        vocals = find_vocals(ep_dir)
        if vocals is None:
            print(f"ERROR: 分離ボーカルなし: {ep_dir}", file=sys.stderr)
            sys.exit(1)

        # 全話者の対象セグメントを集約
        target_segs: list[dict] = []
        for speaker, items in by_ep[ep_dir].items():
            mapping = reconstruct_index_mapping(segments, speaker, MIN_DURATION, EXTRACT_SR)
            for index, csv_dur in items:
                seg = mapping.get(index)
                if seg is None:
                    grand_validate["mismatch"] += 1
                    continue
                seg_dur = seg["end"] - seg["start"]
                grand_validate["checked"] += 1
                if abs(seg_dur - csv_dur) > DUR_TOLERANCE:
                    grand_validate["mismatch"] += 1
                target_segs.append(seg)

        spans = build_spans(segments, target_segs, PAD_START, PAD_END, MERGE_GAP)

        with sf.SoundFile(str(vocals)) as vf:
            sr = vf.samplerate
            for s, e, _ in spans:
                global_idx += 1
                vf.seek(int(s * sr))
                audio = vf.read(int((e - s) * sr), dtype="float32", always_2d=True)
                if audio.shape[1] > 1:
                    audio = audio.mean(axis=1, keepdims=True)
                sf.write(str(OUT_DIR / f"seg_{global_idx:05d}.wav"), audio, sr, subtype="PCM_16")
                total_dur += e - s

        ep_label = re.search(r"第\d+回", ep_dir)
        print(f"  {ep_label.group(0) if ep_label else ep_dir}: "
              f"{len(target_segs)}セグ → {len(spans)}スパン  vocals_sr={sr}")

    print("\n" + "=" * 60)
    print(f"採番検証: {grand_validate['checked']}件照合 / 不一致 {grand_validate['mismatch']}件")
    print(f"出力: {global_idx} スパン, 合計 {total_dur / 60:.1f} 分 → {OUT_DIR}")


if __name__ == "__main__":
    main()
