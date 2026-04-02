#!/usr/bin/env python3
"""Audio visualizer — generate MP4 videos from audio files for TTS demos.

Supports four visualization modes:
  waveform    — amplitude waveform with playhead
  spectrogram — mel spectrogram heatmap with scrolling
  bar         — frequency band analyzer bars
  particle    — waveform + particle effects
  circular    — radial frequency bars around a pulsing circle
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torchaudio
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SR = 22050
HOP_LENGTH = 512
N_FFT = 2048
N_MELS = 128
N_BARS = 48
FPS = 30

DEFAULT_THEME = {
    "bg_color": (10, 10, 25),
    "primary_color": (0, 200, 255),
    "secondary_color": (140, 80, 255),
    "text_color": (220, 220, 240),
}


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------


def load_audio(path: Path) -> tuple[torch.Tensor, int]:
    """Load audio file, resample to TARGET_SR, and mix to mono."""
    try:
        wav, sr = torchaudio.load(str(path))
    except RuntimeError:
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32")
        wav = torch.from_numpy(data)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        else:
            wav = wav.T

    # Mix to mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Resample
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
        sr = TARGET_SR

    return wav.squeeze(0), sr


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def precompute_features(wav: torch.Tensor, sr: int) -> dict:
    """Pre-compute all audio features needed for visualization."""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    amp_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

    # Mel spectrogram in dB
    mel_spec = amp_to_db(mel_transform(wav)).numpy()  # (n_mels, T)

    # STFT magnitude for spectrogram mode
    stft_transform = torchaudio.transforms.Spectrogram(n_fft=N_FFT, hop_length=HOP_LENGTH)
    stft_mag = amp_to_db(stft_transform(wav)).numpy()  # (n_freq, T)

    # RMS envelope (per hop)
    frame_length = N_FFT
    num_frames = 1 + (len(wav) - frame_length) // HOP_LENGTH
    rms = np.zeros(num_frames, dtype=np.float32)
    for i in range(num_frames):
        start = i * HOP_LENGTH
        end = start + frame_length
        if end <= len(wav):
            rms[i] = np.sqrt(np.mean(wav[start:end].numpy() ** 2))
    # Pad or trim to match mel_spec time axis
    if len(rms) < mel_spec.shape[1]:
        rms = np.pad(rms, (0, mel_spec.shape[1] - len(rms)))
    else:
        rms = rms[: mel_spec.shape[1]]

    # Frequency band energies (sum mel bands into N_BARS groups)
    n_time = mel_spec.shape[1]
    mel_positive = mel_spec - mel_spec.min()
    mel_positive = mel_positive / (mel_positive.max() + 1e-10)
    band_energies = np.zeros((N_BARS, n_time), dtype=np.float32)
    mel_per_bar = N_MELS // N_BARS
    for i in range(N_BARS):
        start = i * mel_per_bar
        end = start + mel_per_bar if i < N_BARS - 1 else N_MELS
        band_energies[i] = mel_positive[start:end].mean(axis=0)

    duration = len(wav) / sr
    return {
        "mel_spec": mel_spec,
        "stft_mag": stft_mag,
        "rms": rms,
        "band_energies": band_energies,
        "duration": duration,
        "n_time": n_time,
    }


# ---------------------------------------------------------------------------
# Colormap
# ---------------------------------------------------------------------------


def _build_magma_lut() -> np.ndarray:
    """Build a 256-entry RGB LUT approximating matplotlib's 'magma' colormap."""
    keypoints = [
        (0.0, (0, 0, 4)),
        (0.13, (28, 16, 68)),
        (0.25, (79, 18, 123)),
        (0.38, (129, 37, 129)),
        (0.50, (182, 54, 121)),
        (0.63, (231, 92, 96)),
        (0.75, (251, 136, 97)),
        (0.88, (254, 194, 140)),
        (1.0, (252, 253, 191)),
    ]
    lut = np.zeros((256, 3), dtype=np.uint8)
    for idx in range(256):
        v = idx / 255.0
        # Find surrounding keypoints
        for k in range(len(keypoints) - 1):
            v0, c0 = keypoints[k]
            v1, c1 = keypoints[k + 1]
            if v0 <= v <= v1:
                t = (v - v0) / (v1 - v0) if v1 > v0 else 0.0
                r = int(c0[0] + t * (c1[0] - c0[0]))
                g = int(c0[1] + t * (c1[1] - c0[1]))
                b = int(c0[2] + t * (c1[2] - c0[2]))
                lut[idx] = [r, g, b]
                break
    return lut


MAGMA_LUT = _build_magma_lut()


def spectrogram_to_image(spec_db: np.ndarray) -> Image.Image:
    """Convert dB spectrogram array to a PIL Image using the magma colormap."""
    vmin, vmax = spec_db.min(), spec_db.max()
    if vmax - vmin < 1e-10:
        normalized = np.zeros_like(spec_db)
    else:
        normalized = (spec_db - vmin) / (vmax - vmin)
    indices = (normalized * 255).clip(0, 255).astype(np.uint8)
    rgb = MAGMA_LUT[indices]  # (H, W, 3)
    return Image.fromarray(rgb, "RGB")


# ---------------------------------------------------------------------------
# Utility: load font
# ---------------------------------------------------------------------------


def _get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try to load a good system font, falling back to default."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/meiryo.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Visualization modes
# ---------------------------------------------------------------------------


def _make_canvas(w: int, h: int, bg: tuple) -> Image.Image:
    return Image.new("RGBA", (w, h), (*bg, 255))


def _draw_title(img: Image.Image, title: str, theme: dict, font: ImageFont.ImageFont) -> None:
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), title, font=font)
    tw = bbox[2] - bbox[0]
    x = (img.width - tw) // 2
    draw.text((x, 20), title, fill=(*theme["text_color"], 255), font=font)


# --- Waveform ----------------------------------------------------------------


def render_waveform_frame(
    envelope: np.ndarray,
    duration: float,
    current_time: float,
    W: int,
    H: int,
    theme: dict,
    static_wave: Image.Image | None = None,
    title: str | None = None,
    title_font: ImageFont.ImageFont | None = None,
    label_font: ImageFont.ImageFont | None = None,
) -> Image.Image:
    img = _make_canvas(W, H, theme["bg_color"])
    draw = ImageDraw.Draw(img)

    cy = H // 2
    margin = 60

    if static_wave is None:
        # Draw waveform polygon
        n = len(envelope)
        pts_upper = []
        pts_lower = []
        for i in range(n):
            x = margin + (i / max(n - 1, 1)) * (W - 2 * margin)
            amp = envelope[i]
            h_px = int(amp * (cy - margin - 20))
            pts_upper.append((x, cy - h_px))
            pts_lower.append((x, cy + h_px))
        polygon = pts_upper + pts_lower[::-1]
        if len(polygon) >= 3:
            draw.polygon(polygon, fill=(*theme["primary_color"], 80))
            # Outline (upper)
            if len(pts_upper) >= 2:
                draw.line(pts_upper, fill=(*theme["primary_color"], 200), width=2)
                draw.line(pts_lower, fill=(*theme["primary_color"], 200), width=2)
    else:
        img.alpha_composite(static_wave)

    # Center line
    draw.line([(margin, cy), (W - margin, cy)], fill=(255, 255, 255, 30), width=1)

    # Playhead
    if duration > 0:
        px = margin + (current_time / duration) * (W - 2 * margin)
        # Glow
        for w, a in [(8, 30), (5, 60), (3, 120), (1, 255)]:
            draw.line([(px, margin - 10), (px, H - margin + 10)], fill=(255, 255, 255, a), width=w)

        # Time label at playhead
        if label_font:
            time_str = f"{current_time:.1f}s"
            draw.text((int(px) + 5, H - margin + 15), time_str, fill=(*theme["text_color"], 200), font=label_font)

    # Time markers
    if label_font and duration > 0:
        max_seconds = int(duration)
        for s in range(0, max_seconds + 1, max(1, max_seconds // 10)):
            tx = margin + (s / duration) * (W - 2 * margin)
            draw.line([(int(tx), cy - 3), (int(tx), cy + 3)], fill=(255, 255, 255, 60), width=1)
            if s % max(1, max_seconds // 10) == 0:
                draw.text((int(tx) - 8, H - margin + 15), f"{s}s", fill=(255, 255, 255, 80), font=label_font)

    if title and title_font:
        _draw_title(img, title, theme, title_font)

    return img


def pre_render_waveform(envelope: np.ndarray, W: int, H: int, theme: dict) -> Image.Image:
    """Pre-render the static waveform for faster per-frame rendering."""
    img = _make_canvas(W, H, theme["bg_color"])
    draw = ImageDraw.Draw(img)

    cy = H // 2
    margin = 60
    n = len(envelope)
    pts_upper = []
    pts_lower = []
    for i in range(n):
        x = margin + (i / max(n - 1, 1)) * (W - 2 * margin)
        amp = envelope[i]
        h_px = int(amp * (cy - margin - 20))
        pts_upper.append((x, cy - h_px))
        pts_lower.append((x, cy + h_px))
    polygon = pts_upper + pts_lower[::-1]
    if len(polygon) >= 3:
        draw.polygon(polygon, fill=(*theme["primary_color"], 80))
        if len(pts_upper) >= 2:
            draw.line(pts_upper, fill=(*theme["primary_color"], 200), width=2)
            draw.line(pts_lower, fill=(*theme["primary_color"], 200), width=2)
    # Center line
    draw.line([(margin, cy), (W - margin, cy)], fill=(255, 255, 255, 30), width=1)
    return img


# --- Spectrogram -------------------------------------------------------------


def pre_render_spectrogram(spec_db: np.ndarray, W: int, H: int, margin_left: int = 70, margin_bottom: int = 50) -> Image.Image:
    """Pre-render the full spectrogram as a PIL Image."""
    spec_img = spectrogram_to_image(spec_db)  # (n_mels, T)
    # Scale height to fit the drawing area
    draw_h = H - 2 * 40  # top and bottom margins
    draw_w = W - margin_left - 40
    spec_img = spec_img.resize((max(draw_w, 1), max(draw_h, 1)), Image.BILINEAR)
    return spec_img


def render_spectrogram_frame(
    spec_img: Image.Image,
    duration: float,
    current_time: float,
    W: int,
    H: int,
    theme: dict,
    title: str | None = None,
    title_font: ImageFont.ImageFont | None = None,
    label_font: ImageFont.ImageFont | None = None,
) -> Image.Image:
    img = _make_canvas(W, H, theme["bg_color"])
    draw = ImageDraw.Draw(img)

    margin_left = 70
    margin_right = 40
    margin_top = 60
    margin_bottom = 50
    draw_w = W - margin_left - margin_right
    draw_h = H - margin_top - margin_bottom

    # Scale spectrogram to fit — ensure RGB mode for alpha_composite
    if spec_img.mode != "RGBA":
        spec_img = spec_img.convert("RGBA")
    scaled = spec_img.resize((max(draw_w, 1), max(draw_h, 1)), Image.BILINEAR)
    img.paste(scaled, (margin_left, margin_top))

    # Playhead
    if duration > 0:
        px = margin_left + (current_time / duration) * draw_w
        for w, a in [(6, 30), (3, 80), (1, 200)]:
            draw.line([(px, margin_top), (px, margin_top + draw_h)], fill=(255, 255, 255, a), width=w)

    # Frequency labels (left side)
    if label_font:
        freq_labels = ["0", "2k", "4k", "8k", "11k"]
        for i, label in enumerate(freq_labels):
            y = margin_top + int((i / (len(freq_labels) - 1)) * draw_h)
            draw.text((2, y - 8), f"{label} Hz", fill=(*theme["text_color"], 150), font=label_font)

    # Time markers (bottom)
    if label_font and duration > 0:
        max_seconds = int(duration)
        step = max(1, max_seconds // 8)
        for s in range(0, max_seconds + 1, step):
            tx = margin_left + (s / duration) * draw_w
            draw.text((int(tx) - 5, H - margin_bottom + 10), f"{s}s", fill=(255, 255, 255, 100), font=label_font)

    if title and title_font:
        _draw_title(img, title, theme, title_font)

    return img


# --- Bar analyzer -------------------------------------------------------------


def render_bar_frame(
    band_energies: np.ndarray,
    feat_idx: int,
    W: int,
    H: int,
    theme: dict,
    smoothed: np.ndarray,
    peaks: np.ndarray,
    peak_vel: np.ndarray,
    title: str | None = None,
    title_font: ImageFont.ImageFont | None = None,
) -> Image.Image:
    img = _make_canvas(W, H, theme["bg_color"])
    draw = ImageDraw.Draw(img)

    n_bars = band_energies.shape[0]
    gap = 3
    bar_area_w = W - 60
    bar_w = max((bar_area_w - (n_bars + 1) * gap) // n_bars, 1)
    offset_x = (W - n_bars * bar_w - (n_bars - 1) * gap) // 2
    max_h = int(H * 0.6)
    base_y = int(H * 0.7)

    # Clamp feat_idx
    if feat_idx >= band_energies.shape[1]:
        feat_idx = band_energies.shape[1] - 1

    alpha = 0.35
    for i in range(n_bars):
        energy = band_energies[i, feat_idx]
        smoothed[i] = alpha * energy + (1 - alpha) * smoothed[i]

        h = int(smoothed[i] * max_h)

        # Peak hold
        if h > peaks[i]:
            peaks[i] = h
            peak_vel[i] = 0.0
        else:
            peak_vel[i] += 0.15  # gravity
            peaks[i] = max(0, peaks[i] - peak_vel[i])

        x = offset_x + i * (bar_w + gap)

        # Bar color: gradient from primary to secondary
        t = i / max(n_bars - 1, 1)
        r = int(theme["primary_color"][0] + t * (theme["secondary_color"][0] - theme["primary_color"][0]))
        g = int(theme["primary_color"][1] + t * (theme["secondary_color"][1] - theme["primary_color"][1]))
        b = int(theme["primary_color"][2] + t * (theme["secondary_color"][2] - theme["primary_color"][2]))

        # Main bar
        if h > 0:
            draw.rectangle([x, base_y - h, x + bar_w, base_y], fill=(r, g, b, 220))

        # Reflection (faded mirror below)
        refl_h = h // 3
        if refl_h > 0:
            for dy in range(refl_h):
                a = max(0, 80 - dy * 3)
                draw.line([(x, base_y + 2 + dy), (x + bar_w, base_y + 2 + dy)], fill=(r, g, b, a))

        # Peak marker
        if peaks[i] > 2:
            py = base_y - int(peaks[i])
            draw.rectangle([x, py - 2, x + bar_w, py + 1], fill=(255, 255, 255, 180))

    # Base line
    draw.line([(offset_x, base_y + 1), (offset_x + n_bars * (bar_w + gap), base_y + 1)], fill=(255, 255, 255, 40), width=1)

    if title and title_font:
        _draw_title(img, title, theme, title_font)

    return img


# --- Particle -----------------------------------------------------------------


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    size: float
    r: int
    g: int
    b: int


def render_particle_frame(
    envelope: np.ndarray,
    band_energies: np.ndarray,
    feat_idx: int,
    current_time: float,
    duration: float,
    W: int,
    H: int,
    theme: dict,
    particles: list[Particle],
    rng: np.random.Generator,
    title: str | None = None,
    title_font: ImageFont.ImageFont | None = None,
) -> Image.Image:
    img = _make_canvas(W, H, theme["bg_color"])

    cy = H // 2
    n = len(envelope)

    # Current audio energy
    if 0 <= feat_idx < len(envelope):
        energy = float(envelope[feat_idx])
    else:
        energy = 0.0

    if feat_idx >= band_energies.shape[1]:
        feat_idx = band_energies.shape[1] - 1

    # Background glow pulse based on bass (low frequency bands)
    bass_energy = float(band_energies[:8, feat_idx].mean())
    if bass_energy > 0.05:
        glow_radius = int(200 + bass_energy * 300)
        glow_alpha = int(20 + bass_energy * 60)
        glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(glow)
        glow_draw.ellipse(
            [(W // 2 - glow_radius, cy - glow_radius), (W // 2 + glow_radius, cy + glow_radius)],
            fill=(*theme["secondary_color"], glow_alpha),
        )
        img = Image.alpha_composite(img, glow)

    # Draw waveform line with glow
    draw = ImageDraw.Draw(img)
    pts = []
    margin = 80
    draw_w = W - 2 * margin
    for px_i in range(0, draw_w, 2):
        # Map pixel to time
        t_frac = px_i / draw_w
        env_idx = int(t_frac * (n - 1))
        if env_idx >= n:
            env_idx = n - 1
        amp = envelope[env_idx]
        y = cy - int(amp * (cy - margin))
        pts.append((margin + px_i, y))

    if len(pts) >= 2:
        # Glow layers (wide to narrow)
        for w, a in [(7, 20), (5, 40), (3, 80), (2, 160)]:
            draw.line(pts, fill=(*theme["primary_color"], a), width=w)
        # Core line
        draw.line(pts, fill=(*theme["primary_color"], 240), width=1)

    # Spawn particles along waveform at high-energy regions
    if energy > 0.15:
        spawn_count = int(energy * 8)
        for _ in range(spawn_count):
            px_i = rng.integers(0, draw_w)
            t_frac = px_i / draw_w
            env_idx = int(t_frac * (n - 1))
            if env_idx >= n:
                env_idx = n - 1
            amp = envelope[env_idx]
            y = cy - int(amp * (cy - margin))
            t_blend = rng.random()
            r = int(theme["primary_color"][0] * (1 - t_blend) + theme["secondary_color"][0] * t_blend)
            g = int(theme["primary_color"][1] * (1 - t_blend) + theme["secondary_color"][1] * t_blend)
            b = int(theme["primary_color"][2] * (1 - t_blend) + theme["secondary_color"][2] * t_blend)
            particles.append(
                Particle(
                    x=float(margin + px_i),
                    y=float(y),
                    vx=float(rng.uniform(-2, 2)),
                    vy=float(rng.uniform(-3, -0.5) if amp > 0.3 else rng.uniform(0.5, 3)),
                    life=1.0,
                    max_life=1.0,
                    size=float(rng.uniform(1.5, 4.0)),
                    r=r,
                    g=g,
                    b=b,
                )
            )

    # Update and draw particles
    alive = []
    for p in particles:
        p.x += p.vx
        p.y += p.vy
        p.life -= 0.018
        if p.life > 0:
            alive.append(p)
            alpha = int(p.life * 200)
            s = max(int(p.size * p.life), 1)
            draw.ellipse(
                [(p.x - s, p.y - s), (p.x + s, p.y + s)],
                fill=(p.r, p.g, p.b, alpha),
            )
    particles.clear()
    particles.extend(alive[-200:])  # cap at 200

    if title and title_font:
        _draw_title(img, title, theme, title_font)

    return img


# --- Circular ----------------------------------------------------------------


def render_circular_frame(
    band_energies: np.ndarray,
    envelope: np.ndarray,
    feat_idx: int,
    W: int,
    H: int,
    theme: dict,
    smoothed: np.ndarray,
    rotation: float,
    title: str | None = None,
    title_font: ImageFont.ImageFont | None = None,
) -> Image.Image:
    img = _make_canvas(W, H, theme["bg_color"])
    draw = ImageDraw.Draw(img)

    cx, cy = W // 2, H // 2
    n_bands = band_energies.shape[0]
    inner_r = min(W, H) // 6  # inner circle radius
    max_bar_len = min(W, H) // 2 - 60  # max bar extension

    if feat_idx >= band_energies.shape[1]:
        feat_idx = band_energies.shape[1] - 1

    # Overall energy for pulsing center
    if 0 <= feat_idx < len(envelope):
        energy = float(envelope[feat_idx])
    else:
        energy = 0.0

    # Exponential smoothing
    alpha = 0.4
    for i in range(n_bands):
        smoothed[i] = alpha * band_energies[i, feat_idx] + (1 - alpha) * smoothed[i]

    # Background glow
    glow_r = int(inner_r + 60 + energy * 120)
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    glow_draw.ellipse(
        [(cx - glow_r, cy - glow_r), (cx + glow_r, cy + glow_r)],
        fill=(*theme["secondary_color"], int(15 + energy * 40)),
    )
    img = Image.alpha_composite(img, glow)
    draw = ImageDraw.Draw(img)

    # Inner pulsing circle
    pulse_r = int(inner_r + energy * 20)
    for w, a in [(pulse_r + 8, 20), (pulse_r + 4, 40), (pulse_r, 80)]:
        draw.ellipse(
            [(cx - w, cy - w), (cx + w, cy + w)],
            outline=(*theme["primary_color"], a),
            width=2,
        )
    draw.ellipse(
        [(cx - pulse_r, cy - pulse_r), (cx + pulse_r, cy + pulse_r)],
        outline=(*theme["primary_color"], 180),
        width=2,
    )

    # Radial bars
    angle_step = 2 * np.pi / n_bands
    bar_width_angle = angle_step * 0.65  # thin bars with gaps

    for i in range(n_bands):
        angle = i * angle_step + rotation
        val = smoothed[i]
        bar_len = int(val * max_bar_len) + 2

        # Color gradient: primary → secondary around the circle
        t = i / n_bands
        r = int(theme["primary_color"][0] * (1 - t) + theme["secondary_color"][0] * t)
        g = int(theme["primary_color"][1] * (1 - t) + theme["secondary_color"][1] * t)
        b = int(theme["primary_color"][2] * (1 - t) + theme["secondary_color"][2] * t)

        # Draw bar as a thick line from inner circle outward
        x0 = cx + int(pulse_r * np.cos(angle))
        y0 = cy + int(pulse_r * np.sin(angle))
        x1 = cx + int((pulse_r + bar_len) * np.cos(angle))
        y1 = cy + int((pulse_r + bar_len) * np.sin(angle))

        # Glow
        draw.line([(x0, y0), (x1, y1)], fill=(r, g, b, 60), width=5)
        # Core bar
        draw.line([(x0, y0), (x1, y1)], fill=(r, g, b, 200), width=2)

        # Mirror bar (inward, shorter)
        mirror_len = bar_len // 3
        x2 = cx + int((pulse_r - mirror_len) * np.cos(angle))
        y2 = cy + int((pulse_r - mirror_len) * np.sin(angle))
        draw.line([(x0, y0), (x2, y2)], fill=(r, g, b, 100), width=2)

    # Outer ring connecting bar tips
    tip_pts = []
    for i in range(n_bands):
        angle = i * angle_step + rotation
        val = smoothed[i]
        bar_len = int(val * max_bar_len) + 2
        tip_r = pulse_r + bar_len
        tip_pts.append((cx + int(tip_r * np.cos(angle)), cy + int(tip_r * np.sin(angle))))
    if len(tip_pts) >= 3:
        draw.polygon(tip_pts, outline=(*theme["primary_color"], 50))

    # Center dot
    dot_r = 6
    draw.ellipse(
        [(cx - dot_r, cy - dot_r), (cx + dot_r, cy + dot_r)],
        fill=(*theme["primary_color"], 200),
    )

    if title and title_font:
        _draw_title(img, title, theme, title_font)

    return img


# ---------------------------------------------------------------------------
# ffmpeg video assembly
# ---------------------------------------------------------------------------


def render_video(
    features: dict,
    audio_path: Path,
    output_path: Path,
    mode: str,
    theme: dict,
    fps: int = FPS,
    resolution: tuple[int, int] = (1280, 720),
    preset: str = "medium",
    title: str | None = None,
) -> None:
    """Render the full video by piping frames to ffmpeg."""
    W, H = resolution
    duration = features["duration"]
    total_frames = int(duration * fps) + 1

    # Fonts
    title_font = _get_font(28) if title else None
    label_font = _get_font(16)

    # Pre-compute static assets per mode
    static_wave = None
    spec_img = None
    if mode == "waveform":
        static_wave = pre_render_waveform(features["rms"], W, H, theme)
    elif mode == "spectrogram":
        spec_img = pre_render_spectrogram(features["mel_spec"], W, H)

    # Normalize envelope for waveform/particle
    rms_max = features["rms"].max()
    if rms_max > 0:
        envelope_norm = features["rms"] / rms_max
    else:
        envelope_norm = features["rms"]

    # Mode-specific mutable state
    smoothed = np.zeros(N_BARS, dtype=np.float32)
    circ_smoothed = np.zeros(N_BARS, dtype=np.float32)
    peaks = np.zeros(N_BARS, dtype=np.float32)
    peak_vel = np.zeros(N_BARS, dtype=np.float32)
    particles: list[Particle] = []
    rng = np.random.default_rng(42)

    # Launch ffmpeg
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{W}x{H}",
        "-pix_fmt", "rgba",
        "-r", str(fps),
        "-i", "-",
        "-i", str(audio_path),
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(output_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    print(f"Rendering {total_frames} frames ({duration:.1f}s, {mode} mode)...")
    start_time = time.time()

    for frame_idx in range(total_frames):
        t = frame_idx / fps
        feat_idx = int(t * TARGET_SR / HOP_LENGTH)
        feat_idx = min(feat_idx, features["n_time"] - 1)

        if mode == "waveform":
            img = render_waveform_frame(
                features["rms"], features["duration"], t,
                W, H, theme, static_wave=static_wave,
                title=title, title_font=title_font, label_font=label_font,
            )
        elif mode == "spectrogram":
            img = render_spectrogram_frame(
                spec_img, features["duration"], t,
                W, H, theme, title=title, title_font=title_font, label_font=label_font,
            )
        elif mode == "bar":
            img = render_bar_frame(
                features["band_energies"], feat_idx,
                W, H, theme, smoothed, peaks, peak_vel,
                title=title, title_font=title_font,
            )
        elif mode == "particle":
            img = render_particle_frame(
                envelope_norm, features["band_energies"], feat_idx,
                t, features["duration"],
                W, H, theme, particles, rng,
                title=title, title_font=title_font,
            )
        elif mode == "circular":
            rotation = t * 0.3  # slow rotation
            img = render_circular_frame(
                features["band_energies"], envelope_norm, feat_idx,
                W, H, theme, circ_smoothed, rotation,
                title=title, title_font=title_font,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        proc.stdin.write(img.tobytes())  # type: ignore[union-attr]

        # Progress
        if (frame_idx + 1) % (fps * 2) == 0 or frame_idx == total_frames - 1:
            elapsed = time.time() - start_time
            pct = (frame_idx + 1) / total_frames
            remaining = (elapsed / pct - elapsed) if pct > 0 else 0
            print(f"  [{pct:6.1%}] {frame_idx + 1}/{total_frames}  ETA {remaining:.0f}s")

    proc.stdin.close()  # type: ignore[union-attr]
    proc.wait()
    if proc.returncode != 0:
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        print(f"ffmpeg error:\n{stderr}", file=sys.stderr)
        sys.exit(1)

    elapsed = time.time() - start_time
    print(f"Done: {output_path} ({elapsed:.1f}s)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_color(s: str) -> tuple[int, int, int]:
    parts = s.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Color must be R,G,B (got: {s})")
    return tuple(int(p.strip()) for p in parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate audio visualization videos for TTS demos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/visualize_audio.py input.wav -o output.mp4 -m waveform
  python scripts/visualize_audio.py input.wav -o output.mp4 -m spectrogram --title "Irodori TTS"
  python scripts/visualize_audio.py input.wav -o output.mp4 -m bar --theme-color 255,100,50
  python scripts/visualize_audio.py input.wav -o output.mp4 -m particle --fps 60
""",
    )
    parser.add_argument("input", type=Path, help="Input audio file (WAV, MP3, FLAC, etc.)")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output MP4 (default: <stem>_<mode>.mp4)")
    parser.add_argument("-m", "--mode", choices=["waveform", "spectrogram", "bar", "particle", "circular"], default="waveform")
    parser.add_argument("--title", type=str, default=None, help="Title text overlay")
    parser.add_argument("--theme-color", type=parse_color, default="0,200,255", help="Accent color R,G,B")
    parser.add_argument("--bg-color", type=parse_color, default="10,10,25", help="Background color R,G,B")
    parser.add_argument("--fps", type=int, default=FPS, choices=[24, 30, 60])
    parser.add_argument("--resolution", type=str, default="1280x720", help="WxH")
    parser.add_argument("--preset", type=str, default="medium", choices=["ultrafast", "fast", "medium", "slow"])

    args = parser.parse_args()

    # Check ffmpeg
    if not shutil.which("ffmpeg"):
        print("Error: ffmpeg not found in PATH. Install it first.", file=sys.stderr)
        sys.exit(1)

    # Resolve output path
    if args.output is None:
        args.output = args.input.with_name(f"{args.input.stem}_{args.mode}.mp4")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Resolution
    try:
        w, h = args.resolution.split("x")
        resolution = (int(w), int(h))
    except ValueError:
        print(f"Error: invalid resolution format: {args.resolution}", file=sys.stderr)
        sys.exit(1)

    theme = {
        "bg_color": args.bg_color,
        "primary_color": args.theme_color,
        "secondary_color": (
            min(255, args.theme_color[0] + 140),
            max(0, args.theme_color[1] - 120),
            255,
        ),
        "text_color": (220, 220, 240),
    }

    # Load audio
    print(f"Loading: {args.input}")
    wav, sr = load_audio(args.input)
    print(f"  Duration: {len(wav) / sr:.2f}s, SR: {sr}")

    # Extract features
    print("Extracting features...")
    features = precompute_features(wav, sr)

    # Render
    render_video(
        features,
        args.input,
        args.output,
        args.mode,
        theme,
        fps=args.fps,
        resolution=resolution,
        preset=args.preset,
        title=args.title,
    )


if __name__ == "__main__":
    main()
