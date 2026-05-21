# Irodori-TTS Project Conventions

## Package Management
- Use **uv** for all package management (`uv sync`, `uv add`, `uv run`)
- Python version: 3.10 (`.python-version`)
- Dependencies: `pyproject.toml` / `uv.lock`

## Script Organization
- Python utility scripts → `scripts/` directory
- Run: `uv run python scripts/<name>.py`
- Shell scripts → `scripts/`
- CLI: argparse を使用（`scripts/split_and_transcribe.py` のパターンに従う）
- `from __future__ import annotations` を冒頭に記述

## Data Paths
- Raw audio input: `data/input/<speaker_name>/`
- Processed segments: `data/<speaker_name>/wavs/`
- DACVAE latents: `data/<speaker_name>/latents/`
- Training manifest: `data/<speaker_name>/manifest.jsonl`
- Diarization output: `data/output/diarization/<stem>/`

## Code Style
- Ruff (lint + format, config in pyproject.toml)
- Line length: 100, double quotes, 4-space indent
