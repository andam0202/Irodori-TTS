#!/usr/bin/env python3
"""話者分離済みセグメントを話者埋め込みで再クラスタリングするスクリプト。

diarize_speakers.py の出力（speakers/SPEAKER_XX/*.wav）に対して、
wespeaker 話者埋め込み + 凝集型クラスタリングで再分類する。
--num-speakers の強制指定などでクラスタ割り当てが崩れた場合の修正に使う。
各クラスタの中央値 F0 も算出するため、男声/女声/電子音声の見当が付けやすい。

使い方:
  uv run python scripts/refine_speaker_clusters.py \
    --input-dirs data/output/diarization/pragmata_diana/speakers/SPEAKER_00 \
                 data/output/diarization/pragmata_diana/speakers/SPEAKER_01 \
    --output-dir data/output/diarization/pragmata_diana/reclustered

出力:
  <output-dir>/cluster_XX/*.wav     - クラスタ別セグメント（元ファイル名は <元話者>_<元名>.wav）
  <output-dir>/samples/             - 各クラスタの試聴用サンプル（長い順に数本）
  <output-dir>/assignments.csv      - 全ファイルの割り当て・F0・類似度一覧
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

# ---- torchaudio 2.9+ 互換パッチ（diarize_speakers.py と同様） ----
if not hasattr(torchaudio, "AudioMetaData"):
    from dataclasses import dataclass

    @dataclass
    class AudioMetaData:
        sample_rate: int
        num_frames: int
        num_channels: int
        bits_per_sample: int
        encoding: str

    torchaudio.AudioMetaData = AudioMetaData

if not hasattr(torchaudio, "info"):

    def _torchaudio_info(filepath, backend=None):
        info = sf.info(filepath)
        return torchaudio.AudioMetaData(
            sample_rate=int(info.samplerate),
            num_frames=int(info.frames),
            num_channels=int(info.channels),
            bits_per_sample=0,
            encoding=info.subtype or "",
        )

    torchaudio.info = _torchaudio_info

if not hasattr(torchaudio, "list_audio_backends"):

    def _list_audio_backends():
        return ["soundfile"]

    torchaudio.list_audio_backends = _list_audio_backends

# ---- PyTorch 2.6+ 互換パッチ ----
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

EMBEDDING_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"
TARGET_SR = 16000


def load_wav_16k_mono(path: Path) -> torch.Tensor:
    """WAV をモノラル 16kHz の (1, samples) テンソルとして読み込む。"""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)  # (ch, samples)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav


def compute_f0_median(wav_np: np.ndarray) -> float:
    """librosa.pyin で有声区間の中央値 F0 (Hz) を返す。有声区間なしは NaN。"""
    import librosa

    f0, voiced_flag, _ = librosa.pyin(
        wav_np,
        fmin=65,
        fmax=600,
        sr=TARGET_SR,
        frame_length=1024,
    )
    voiced = f0[voiced_flag] if voiced_flag is not None else f0[~np.isnan(f0)]
    if voiced is None or len(voiced) == 0:
        return float("nan")
    return float(np.nanmedian(voiced))


def collect_files(input_dirs: list[Path], min_duration: float) -> list[tuple[Path, str]]:
    """入力ディレクトリから WAV を収集する。(path, 元話者ラベル) のリストを返す。"""
    files: list[tuple[Path, str]] = []
    skipped = 0
    for d in input_dirs:
        if not d.is_dir():
            print(f"ERROR: {d} が見つかりません", file=sys.stderr)
            sys.exit(1)
        label = d.name
        for p in sorted(d.glob("*.wav")):
            info = sf.info(str(p))
            if info.frames / info.samplerate < min_duration:
                skipped += 1
                continue
            files.append((p, label))
    print(f"[collect] {len(files)} ファイル（{min_duration}s 未満の {skipped} 件をスキップ）")
    return files


def extract_embeddings(
    files: list[tuple[Path, str]],
    device: str,
    compute_f0: bool = True,
) -> tuple[np.ndarray, list[float], list[float], list[int]]:
    """各ファイルの話者埋め込みと F0 を計算する。

    Returns:
        (embeddings, durations, f0_medians, valid_indices)
        embeddings は L2 正規化済み。埋め込みが NaN になったファイルは除外される。
    """
    from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

    print(f"[embed] loading model: {EMBEDDING_MODEL}")
    model = PretrainedSpeakerEmbedding(EMBEDDING_MODEL, device=torch.device(device))

    embeddings: list[np.ndarray] = []
    durations: list[float] = []
    f0_medians: list[float] = []
    valid_indices: list[int] = []

    t0 = time.time()
    for i, (path, _) in enumerate(files):
        wav = load_wav_16k_mono(path)
        emb = model(wav.unsqueeze(0))  # (1, ch, samples) -> (1, dim)
        emb = np.asarray(emb)[0]
        if np.isnan(emb).any():
            continue
        norm = np.linalg.norm(emb)
        if norm == 0:
            continue
        embeddings.append(emb / norm)
        durations.append(wav.shape[-1] / TARGET_SR)
        f0_medians.append(compute_f0_median(wav.squeeze(0).numpy()) if compute_f0 else float("nan"))
        valid_indices.append(i)

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"[embed] {i + 1}/{len(files)} ({elapsed:.0f}s elapsed)")

    print(f"[embed] {len(embeddings)}/{len(files)} ファイルの埋め込みを計算 "
          f"({time.time() - t0:.0f}s)")
    return np.stack(embeddings), durations, f0_medians, valid_indices


def cluster_embeddings(
    embeddings: np.ndarray,
    num_clusters: int | None,
    threshold: float,
) -> np.ndarray:
    """凝集型クラスタリング（cosine 距離・average linkage）でラベルを返す。"""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    print("[cluster] computing pairwise cosine distances...")
    dist = pdist(embeddings, metric="cosine")
    z = linkage(dist, method="average")

    if num_clusters is not None:
        labels = fcluster(z, t=num_clusters, criterion="maxclust")
        print(f"[cluster] num_clusters={num_clusters} で分割")
    else:
        labels = fcluster(z, t=threshold, criterion="distance")
        print(f"[cluster] distance threshold={threshold} → {labels.max()} クラスタ")
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="話者分離済みセグメントを話者埋め込みで再クラスタリングする",
    )
    parser.add_argument(
        "--input-dirs", type=Path, nargs="+", required=True,
        help="セグメント WAV のディレクトリ（複数指定可）",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="出力ディレクトリ")
    parser.add_argument(
        "--num-clusters", type=int, default=None,
        help="クラスタ数（省略時は --threshold による自動決定）",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.7,
        help="凝集型クラスタリングの cosine 距離閾値（デフォルト: 0.7、小さいほど細分化）",
    )
    parser.add_argument(
        "--min-duration", type=float, default=1.0,
        help="この秒数未満のセグメントは除外（埋め込みが不安定なため。デフォルト: 1.0）",
    )
    parser.add_argument("--device", type=str, default="cuda", help="推論デバイス (cuda/cpu)")
    parser.add_argument(
        "--num-samples", type=int, default=5,
        help="各クラスタの試聴用サンプル数（デフォルト: 5）",
    )
    parser.add_argument("--no-f0", action="store_true", help="F0 計算をスキップ（高速化）")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA が利用できないため CPU に切り替えます", file=sys.stderr)
        args.device = "cpu"

    files = collect_files(args.input_dirs, args.min_duration)
    if not files:
        print("ERROR: 対象ファイルがありません", file=sys.stderr)
        sys.exit(1)

    embeddings, durations, f0_medians, valid_indices = extract_embeddings(
        files, args.device, compute_f0=not args.no_f0,
    )
    labels = cluster_embeddings(embeddings, args.num_clusters, args.threshold)

    # クラスタを合計発話時間の降順で並べ替えて cluster_00, 01, ... に再番号付け
    unique = np.unique(labels)
    cluster_durations = {c: sum(d for d, l in zip(durations, labels) if l == c) for c in unique}
    ordered = sorted(unique, key=lambda c: -cluster_durations[c])
    rename = {c: i for i, c in enumerate(ordered)}

    out_dir = args.output_dir
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # 割り当て CSV + ファイルコピー
    rows = []
    for emb_idx, file_idx in enumerate(valid_indices):
        path, orig_label = files[file_idx]
        cid = rename[labels[emb_idx]]
        cluster_dir = out_dir / f"cluster_{cid:02d}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        dest = cluster_dir / f"{orig_label}_{path.name}"
        shutil.copy2(path, dest)
        rows.append({
            "cluster": cid,
            "source": str(path),
            "orig_speaker": orig_label,
            "duration_sec": round(durations[emb_idx], 2),
            "f0_median_hz": round(f0_medians[emb_idx], 1)
            if not np.isnan(f0_medians[emb_idx]) else "",
        })

    csv_path = out_dir / "assignments.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["cluster", "source", "orig_speaker", "duration_sec", "f0_median_hz"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[output] assignments: {csv_path}")

    # 試聴用サンプル（各クラスタの長いセグメント上位 N 本）
    for cid in sorted(set(rename.values())):
        members = [(emb_idx, durations[emb_idx]) for emb_idx, lb in enumerate(labels)
                   if rename[lb] == cid]
        members.sort(key=lambda x: -x[1])
        for rank, (emb_idx, _) in enumerate(members[: args.num_samples]):
            src, orig_label = files[valid_indices[emb_idx]]
            shutil.copy2(src, samples_dir / f"cluster_{cid:02d}_sample{rank}_{orig_label}_{src.name}")

    # サマリー表示
    print(f"\n{'=' * 70}")
    print(f"  再クラスタリング結果: {len(ordered)} クラスタ")
    print(f"{'=' * 70}")
    print(f"  {'cluster':<12}{'files':>7}{'duration':>12}{'median F0':>12}  目安")
    for cid in sorted(set(rename.values())):
        idxs = [i for i, lb in enumerate(labels) if rename[lb] == cid]
        total = sum(durations[i] for i in idxs)
        f0s = [f0_medians[i] for i in idxs if not np.isnan(f0_medians[i])]
        f0_med = float(np.median(f0s)) if f0s else float("nan")
        if np.isnan(f0_med):
            hint = ""
        elif f0_med >= 165:
            hint = "女声/子供の可能性"
        elif f0_med >= 85:
            hint = "男声の可能性"
        else:
            hint = "低周波（電子音声等?）"
        f0_str = f"{f0_med:.0f} Hz" if not np.isnan(f0_med) else "-"
        print(f"  cluster_{cid:02d}  {len(idxs):>7}{total / 60:>10.1f}min{f0_str:>12}  {hint}")
    print(f"{'=' * 70}")
    print(f"\n試聴用サンプル: {samples_dir}/")
    print("各クラスタのサンプルを聴いて目的の話者を特定してください。")


if __name__ == "__main__":
    main()
