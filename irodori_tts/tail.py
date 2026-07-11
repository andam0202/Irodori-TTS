"""生成音声の語尾後処理.

モデルは語尾の自然減衰を生成しきれず、フル音量から約50msで無音に落ちる
「崖」を作ることがある。ファイル末尾ではなく発話終端の位置を検出して
コサイン減衰を掛け、必要なら末尾に無音を付加する。
infer.py の --tail-fade-ms / --tail-pad-out-ms から使われる。
"""

from __future__ import annotations

import math

import torch


def find_speech_end(audio: torch.Tensor, sample_rate: int, threshold_db: float = -45.0) -> int:
    """末尾から走査し、発話が終わるサンプル位置を返す（20ms RMS 窓）。"""
    mono = audio.mean(dim=0) if audio.dim() > 1 else audio
    win = max(1, int(0.02 * sample_rate))
    thr = 10.0 ** (threshold_db / 20.0)
    n = mono.shape[-1]
    for end in range(n, 0, -win):
        seg = mono[max(0, end - win): end]
        if seg.pow(2).mean().sqrt().item() > thr:
            return end
    return n


def postprocess_tail(
    audio: torch.Tensor,
    sample_rate: int,
    *,
    fade_ms: float,
    pad_out_ms: float,
) -> torch.Tensor:
    """発話終端にコサインフェードを掛け、末尾に無音を付加する。

    fade_ms / pad_out_ms が 0 以下ならそれぞれの処理をスキップする。
    """
    fade_n = int(float(fade_ms) / 1000.0 * sample_rate)
    if fade_n > 0 and audio.shape[-1] > fade_n:
        # モデルが語尾の減衰を生成しきれず高音量のまま無音に落ちる「崖」を、
        # 発話終端の位置に減衰カーブを掛けることで自然な余韻に変える
        speech_end = find_speech_end(audio, sample_rate)
        fade_start = max(0, speech_end - fade_n)
        audio = audio.clone()
        seg_len = speech_end - fade_start
        if seg_len > 0:
            fade = torch.cos(
                torch.linspace(0, math.pi / 2, seg_len, dtype=audio.dtype)
            ) ** 2
            audio[..., fade_start:speech_end] = audio[..., fade_start:speech_end] * fade
            audio[..., speech_end:] = 0.0
            audio = audio[..., : speech_end]
    pad_n = int(float(pad_out_ms) / 1000.0 * sample_rate)
    if pad_n > 0:
        pad = torch.zeros((*audio.shape[:-1], pad_n), dtype=audio.dtype)
        audio = torch.cat([audio, pad], dim=-1)
    return audio
