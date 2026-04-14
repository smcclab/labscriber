# labscriber/cli.py
import argparse
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

from labscriber.config import Config, detect_device
from labscriber.ingest import discover_audio_files


def main() -> None:
    """Entry point for the labscriber CLI."""
    # Load .env from current working directory (not searched up the tree)
    load_dotenv(Path.cwd() / ".env")

    parser = argparse.ArgumentParser(
        prog="labscriber",
        description="Local batch transcription for multi-speaker HCI research recordings.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── setup ──────────────────────────────────────────────────────────────────
    subparsers.add_parser("setup", help="Interactive HuggingFace token setup and validation.")

    # ── process ───────────────────────────────────────────────────────────────
    proc = subparsers.add_parser("process", help="Run the full transcription pipeline.")
    proc.add_argument("--ingest-path", type=Path, default=Path("ingest"),
                      help="Input audio directory (default: ./ingest)")
    proc.add_argument("--output-path", type=Path, default=Path("output"),
                      help="Output transcript directory (default: ./output)")
    proc.add_argument("--work-path", type=Path, default=Path("work"),
                      help="Intermediate files directory (default: ./work)")
    proc.add_argument("--model", default="large-v2",
                      choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                      help="Whisper model (default: large-v2)")
    proc.add_argument("--language", default=None,
                      help="Force language, e.g. 'english'. Default: auto-detect per file.")
    proc.add_argument("--min-speakers", type=int, default=1,
                      help="Minimum speaker count hint (default: 1)")
    proc.add_argument("--max-speakers", type=int, default=None,
                      help="Maximum speaker count hint (recommended for known speaker count)")
    proc.add_argument("--merge-gap", type=float, default=1.5,
                      help="Max silence (sec) before splitting same-speaker segments (default: 1.5)")
    proc.add_argument("--no-diarize", action="store_true",
                      help="Skip speaker diarization; produce plain transcript only.")
    proc.add_argument("--force", action="store_true",
                      help="Re-process all files, ignoring existing outputs.")
    proc.add_argument("--verbose", action="store_true",
                      help="Print detailed debug output.")

    # ── download-models ────────────────────────────────────────────────────────
    dm = subparsers.add_parser("download-models", help="Pre-download all ML models.")
    dm.add_argument("--model", default="large-v2",
                    choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                    help="Whisper model to download (default: large-v2)")

    # ── status ────────────────────────────────────────────────────────────────
    st = subparsers.add_parser("status", help="Show processing state of files in ingest/.")
    st.add_argument("--ingest-path", type=Path, default=Path("ingest"))
    st.add_argument("--work-path", type=Path, default=Path("work"))
    st.add_argument("--output-path", type=Path, default=Path("output"))

    # ── clean ─────────────────────────────────────────────────────────────────
    cl = subparsers.add_parser("clean", help="Delete work/ intermediate files.")
    cl.add_argument("--work-path", type=Path, default=Path("work"),
                    help="Intermediate files directory to delete (default: ./work)")

    args = parser.parse_args()

    # ── Dispatch ──────────────────────────────────────────────────────────────
    if args.command == "setup":
        from labscriber.setup_wizard import run_setup
        run_setup(Path.cwd())

    elif args.command == "process":
        from labscriber.pipeline import process
        device, compute_type = detect_device()
        config = Config(
            ingest_path=args.ingest_path,
            output_path=args.output_path,
            work_path=args.work_path,
            model=args.model,
            language=args.language,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            merge_gap=args.merge_gap,
            no_diarize=args.no_diarize,
            force=args.force,
            verbose=args.verbose,
            device=device,
            compute_type=compute_type,
            hf_token=os.environ.get("HF_TOKEN"),
        )
        process(config)

    elif args.command == "download-models":
        from labscriber.models import download_all
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            print("Error: HF_TOKEN not set. Run 'labscriber setup' first.")
            sys.exit(1)
        download_all(args.model, hf_token)

    elif args.command == "status":
        _cmd_status(args)

    elif args.command == "clean":
        _cmd_clean(args)


def _cmd_status(args) -> None:
    """Print per-file processing state."""
    if not args.ingest_path.exists():
        print(f"Ingest directory not found: {args.ingest_path}")
        return

    audio_files = discover_audio_files(args.ingest_path)
    if not audio_files:
        print(f"No audio files in {args.ingest_path}")
        return

    wav_dir = args.work_path / "wav"
    asr_dir = args.work_path / "asr"
    diar_dir = args.work_path / "diarization"

    for audio_path in audio_files:
        stem = audio_path.stem
        wav = "\u2713" if (wav_dir / (stem + ".wav")).exists() else "\u2717"
        asr = "\u2713" if (asr_dir / (stem + ".json")).exists() else "\u2717"
        diar = "\u2713" if (diar_dir / (stem + ".json")).exists() else "\u2717"
        out = "\u2713" if (args.output_path / (stem + ".md")).exists() else "\u2717"
        print(f"{audio_path}  {wav} wav  {asr} asr  {diar} diarize  {out} output")


def _cmd_clean(args) -> None:
    """Delete the work/ intermediate directory."""
    if args.work_path.exists():
        shutil.rmtree(args.work_path)
        print(f"Deleted {args.work_path}")
    else:
        print(f"{args.work_path} does not exist — nothing to clean.")
