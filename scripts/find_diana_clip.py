from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def analyze_file(path: Path, sr: int, min_dur: float, max_dur: float,
                 top_db: float, f0_min: float) -> list[dict]:
    """分離済みボーカルから発話区間を切り出し、各区間の F0/クリーンさを評価する。"""
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    if y.size == 0:
        return []
    # 無音区切りで発話区間を検出
    intervals = librosa.effects.split(y, top_db=top_db, frame_length=2048, hop_length=512)
    # 近接区間（ギャップ < merge_gap 秒）を1つの連続発話に結合する
    merge_gap = int(0.35 * sr)
    merged: list[list[int]] = []
    for s, e in intervals:
        if merged and s - merged[-1][1] <= merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    intervals = merged
    results: list[dict] = []
    for start, end in intervals:
        dur = (end - start) / sr
        if dur < min_dur or dur > max_dur:
            continue
        seg = y[start:end]
        # 有声フレームの F0 中央値（pyin）
        try:
            f0, voiced, _ = librosa.pyin(
                seg, fmin=120, fmax=600, sr=sr,
                frame_length=2048, hop_length=256,
            )
        except Exception:
            continue
        f0v = f0[~np.isnan(f0)]
        if f0v.size < 5:
            continue
        f0_med = float(np.median(f0v))
        voiced_ratio = float(np.mean(voiced)) if voiced is not None else 0.0
        rms = float(np.sqrt(np.mean(seg ** 2)))
        results.append({
            "file": path.name,
            "start_s": start / sr,
            "end_s": end / sr,
            "dur": dur,
            "f0_med": f0_med,
            "voiced_ratio": voiced_ratio,
            "rms": rms,
            "_y": seg,
        })
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocals-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--min-dur", type=float, default=6.0)
    ap.add_argument("--max-dur", type=float, default=12.0)
    ap.add_argument("--top-db", type=float, default=35.0)
    ap.add_argument("--f0-min", type=float, default=230.0,
                    help="Diana 候補とみなす F0 中央値の下限（高音の少女声）")
    ap.add_argument("--top-n", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_res: list[dict] = []
    for wav in sorted(Path(args.vocals_dir).glob("*.wav")):
        all_res.extend(analyze_file(wav, args.sr, args.min_dur, args.max_dur,
                                    args.top_db, args.f0_min))

    # Diana 候補: F0 高め + 有声比率高め + 十分なエネルギー
    cands = [r for r in all_res if r["f0_med"] >= args.f0_min
             and r["voiced_ratio"] >= 0.55 and r["rms"] >= 0.01]
    # スコア: 有声比率 × エネルギー（クリーンで連続した発話を優先）
    cands.sort(key=lambda r: r["voiced_ratio"] * np.log1p(r["rms"]), reverse=True)

    print(f"全発話区間: {len(all_res)} / Diana候補(F0>={args.f0_min}): {len(cands)}")
    print(f"{'rank':>4} {'file':28} {'start':>7} {'dur':>5} {'F0':>6} {'voiced':>6} {'rms':>6}")
    for i, r in enumerate(cands[:args.top_n]):
        clip_path = out / f"cand_{i:02d}.wav"
        sf.write(str(clip_path), r["_y"], args.sr)
        print(f"{i:>4} {r['file'][:28]:28} {r['start_s']:7.1f} {r['dur']:5.1f} "
              f"{r['f0_med']:6.0f} {r['voiced_ratio']:6.2f} {r['rms']:6.3f}  -> {clip_path.name}")


if __name__ == "__main__":
    main()
