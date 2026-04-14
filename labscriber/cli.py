# labscriber/cli.py
import os
import shutil
from pathlib import Path

import click
from dotenv import load_dotenv

from labscriber.config import Config, detect_device
from labscriber.ingest import discover_audio_files

MODEL_CHOICES = click.Choice(["tiny", "base", "small", "medium", "large-v2", "large-v3"])


@click.group()
def main() -> None:
    """Local batch transcription for multi-speaker HCI research recordings."""
    load_dotenv(Path.cwd() / ".env")


@main.command()
def setup() -> None:
    """Interactive HuggingFace token setup and validation."""
    from labscriber.setup_wizard import run_setup
    run_setup(Path.cwd())


@main.command()
@click.option("--ingest-path", type=click.Path(path_type=Path), default=Path("ingest"),
              show_default=True, help="Input audio directory.")
@click.option("--output-path", type=click.Path(path_type=Path), default=Path("output"),
              show_default=True, help="Output transcript directory.")
@click.option("--work-path", type=click.Path(path_type=Path), default=Path("work"),
              show_default=True, help="Intermediate files directory.")
@click.option("--model", type=MODEL_CHOICES, default="large-v2",
              show_default=True, help="Whisper model size.")
@click.option("--language", default=None, help="Force language code, e.g. 'en'. Default: auto-detect.")
@click.option("--min-speakers", type=int, default=1, show_default=True,
              help="Minimum speaker count hint.")
@click.option("--max-speakers", type=int, default=None,
              help="Maximum speaker count hint (recommended for known speaker count).")
@click.option("--merge-gap", type=float, default=1.5, show_default=True,
              help="Max silence (sec) before splitting same-speaker segments.")
@click.option("--no-diarize", is_flag=True, help="Skip speaker diarization; produce plain transcript only.")
@click.option("--force", is_flag=True, help="Re-process all files, ignoring existing outputs.")
@click.option("--verbose", is_flag=True, help="Print detailed debug output.")
def process(ingest_path, output_path, work_path, model, language, min_speakers,
            max_speakers, merge_gap, no_diarize, force, verbose) -> None:
    """Run the full transcription pipeline."""
    from labscriber.pipeline import process as run_pipeline
    device, compute_type = detect_device()
    config = Config(
        ingest_path=ingest_path,
        output_path=output_path,
        work_path=work_path,
        model=model,
        language=language,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        merge_gap=merge_gap,
        no_diarize=no_diarize,
        force=force,
        verbose=verbose,
        device=device,
        compute_type=compute_type,
        hf_token=os.environ.get("HF_TOKEN"),
    )
    run_pipeline(config)


@main.command("download-models")
@click.option("--model", type=MODEL_CHOICES, default="large-v2",
              show_default=True, help="Whisper model to download.")
def download_models(model) -> None:
    """Pre-download all ML models."""
    from labscriber.models import download_all
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise click.ClickException("HF_TOKEN not set. Run 'labscriber setup' first.")
    download_all(model, hf_token)


@main.command()
@click.option("--ingest-path", type=click.Path(path_type=Path), default=Path("ingest"), show_default=True)
@click.option("--work-path", type=click.Path(path_type=Path), default=Path("work"), show_default=True)
@click.option("--output-path", type=click.Path(path_type=Path), default=Path("output"), show_default=True)
def status(ingest_path, work_path, output_path) -> None:
    """Show processing state of files in ingest/."""
    if not ingest_path.exists():
        click.echo(f"Ingest directory not found: {ingest_path}")
        return

    audio_files = discover_audio_files(ingest_path)
    if not audio_files:
        click.echo(f"No audio files in {ingest_path}")
        return

    wav_dir = work_path / "wav"
    asr_dir = work_path / "asr"
    diar_dir = work_path / "diarization"

    for audio_path in audio_files:
        stem = audio_path.stem
        wav = "✓" if (wav_dir / (stem + ".wav")).exists() else "✗"
        asr = "✓" if (asr_dir / (stem + ".json")).exists() else "✗"
        diar = "✓" if (diar_dir / (stem + ".json")).exists() else "✗"
        out = "✓" if (output_path / (stem + ".md")).exists() else "✗"
        click.echo(f"{audio_path}  {wav} wav  {asr} asr  {diar} diarize  {out} output")


@main.command()
@click.option("--work-path", type=click.Path(path_type=Path), default=Path("work"),
              show_default=True, help="Intermediate files directory to delete.")
@click.confirmation_option(prompt="Delete all intermediate files in work/?")
def clean(work_path) -> None:
    """Delete work/ intermediate files."""
    if work_path.exists():
        shutil.rmtree(work_path)
        click.echo(f"Deleted {work_path}")
    else:
        click.echo(f"{work_path} does not exist — nothing to clean.")
