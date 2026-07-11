"""irodori_tts.tail の回帰テスト（GPU 不要）.

「フル音量→即無音」の崖を持つ合成波形に対して、発話終端検出・
コサインフェード・無音パディングの各動作を検証する。
"""

from __future__ import annotations

import torch

from irodori_tts.tail import find_speech_end, postprocess_tail

SR = 48_000


def make_cliff(speech_sec: float = 1.0, silence_sec: float = 0.5) -> torch.Tensor:
    """一定振幅の発話区間の直後にデジタル無音が続く「崖」波形（mono）."""
    n_speech = int(speech_sec * SR)
    n_sil = int(silence_sec * SR)
    t = torch.arange(n_speech, dtype=torch.float32) / SR
    speech = 0.5 * torch.sin(2 * torch.pi * 220.0 * t)
    return torch.cat([speech, torch.zeros(n_sil)])


def test_find_speech_end_locates_cliff() -> None:
    audio = make_cliff(speech_sec=1.0, silence_sec=0.5)
    end = find_speech_end(audio, SR)
    # 20ms 窓の粒度で崖の位置（1.0s = 48000サンプル）を検出する
    assert abs(end - SR) <= int(0.02 * SR) + 1


def test_find_speech_end_all_speech_returns_length() -> None:
    audio = make_cliff(speech_sec=1.0, silence_sec=0.0)
    assert find_speech_end(audio, SR) == audio.shape[-1]


def test_find_speech_end_stereo() -> None:
    mono = make_cliff()
    stereo = torch.stack([mono, mono])
    end = find_speech_end(stereo, SR)
    assert abs(end - SR) <= int(0.02 * SR) + 1


def test_postprocess_tail_disabled_is_identity() -> None:
    audio = make_cliff()
    out = postprocess_tail(audio, SR, fade_ms=0.0, pad_out_ms=0.0)
    assert torch.equal(out, audio)


def test_postprocess_tail_fades_cliff() -> None:
    audio = make_cliff(speech_sec=1.0, silence_sec=0.5)
    out = postprocess_tail(audio, SR, fade_ms=120.0, pad_out_ms=0.0)
    speech_end = find_speech_end(audio, SR)
    # フェード後は発話終端で切り詰められる
    assert out.shape[-1] == speech_end
    # 終端直前 5ms の RMS はフェードでほぼゼロに落ちる
    tail_rms = out[-int(0.005 * SR):].pow(2).mean().sqrt().item()
    orig_rms = audio[:speech_end][-int(0.005 * SR):].pow(2).mean().sqrt().item()
    assert tail_rms < orig_rms * 0.1
    # フェード開始位置より前は変更されない
    fade_n = int(120.0 / 1000.0 * SR)
    assert torch.equal(out[: speech_end - fade_n], audio[: speech_end - fade_n])


def test_postprocess_tail_pads_silence() -> None:
    audio = make_cliff()
    out = postprocess_tail(audio, SR, fade_ms=0.0, pad_out_ms=250.0)
    pad_n = int(250.0 / 1000.0 * SR)
    assert out.shape[-1] == audio.shape[-1] + pad_n
    assert torch.equal(out[-pad_n:], torch.zeros(pad_n))


def test_postprocess_tail_original_not_mutated() -> None:
    audio = make_cliff()
    backup = audio.clone()
    postprocess_tail(audio, SR, fade_ms=120.0, pad_out_ms=250.0)
    assert torch.equal(audio, backup)
