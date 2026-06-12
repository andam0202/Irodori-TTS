"""Download audio from YouTube as WAV using yt-dlp."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "input"


def download(url: str, output_dir: Path, speaker: str | None = None, quality: int = 0) -> Path:
    """Download audio from URL as WAV. Returns the output file path."""
    if speaker:
        output_dir = output_dir / speaker
    output_dir.mkdir(parents=True, exist_ok=True)

    out_template = str(output_dir / "%(title)s.%(ext)s")
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-x", "--audio-format", "wav",
        "--audio-quality", str(quality),
        "-o", out_template,
        "--no-playlist",
        url,
    ]
    print(f"Downloading: {url}")
    print(f"Output dir : {output_dir}")
    subprocess.run(cmd, check=True)

    # Find the downloaded file
    wav_files = sorted(output_dir.glob("*.wav"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not wav_files:
        print("ERROR: No WAV file found after download", file=sys.stderr)
        sys.exit(1)
    return wav_files[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download audio from YouTube as WAV")
    parser.add_argument("url", help="YouTube URL")
    parser.add_argument("-s", "--speaker", default=None, help="Speaker name (creates subdirectory)")
    parser.add_argument("-o", "--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("-q", "--quality", type=int, default=0, help="Audio quality (0=best, default: 0)")
    args = parser.parse_args()

    path = download(args.url, Path(args.output_dir), args.speaker, args.quality)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
