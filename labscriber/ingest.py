# labscriber/ingest.py
from pathlib import Path

import ffmpeg

AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".mp4", ".m4a", ".aac",
    ".flac", ".ogg", ".aiff", ".wma",
}


def discover_audio_files(ingest_path: Path) -> list[Path]:
    """Return sorted list of audio files in ingest_path, skipping duplicate stems."""
    files: list[Path] = []
    seen_stems: dict[str, Path] = {}

    for path in sorted(ingest_path.rglob("*")):
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        stem = path.stem
        if stem in seen_stems:
            print(f"Warning: duplicate stem '{stem}' — skipping {path} (keeping {seen_stems[stem]})")
            continue
        seen_stems[stem] = path
        files.append(path)

    return files


def convert_to_wav(input_path: Path, output_path: Path) -> None:
    """Convert any audio file to 16 kHz mono 16-bit PCM WAV via ffmpeg."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    (
        ffmpeg
        .input(str(input_path))
        .output(str(output_path), ar=16000, ac=1, sample_fmt="s16")
        .overwrite_output()
        .run(quiet=True)
    )
