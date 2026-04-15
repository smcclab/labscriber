# labscriber/config.py
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    ingest_path: Path = field(default_factory=lambda: Path("ingest"))
    output_path: Path = field(default_factory=lambda: Path("output"))
    work_path: Path = field(default_factory=lambda: Path("work"))
    model: str = "large-v2"
    language: str | None = None
    min_speakers: int = 1
    max_speakers: int | None = None
    merge_gap: float = 1.5
    no_diarize: bool = False
    force: bool = False
    verbose: bool = False
    device: str = "cpu"
    compute_type: str = "int8"
    hf_token: str | None = None


def detect_device() -> tuple[str, str]:
    """Return (device, compute_type) based on available hardware."""
    import torch

    if torch.cuda.is_available():
        return "cuda", "float16"
    else:
        # ctranslate2 (faster-whisper backend) does not support MPS; fall back to CPU.
        import click
        click.echo("Notice: No CUDA GPU detected. Processing will be slow. Consider --model medium.")
        return "cpu", "int8"
