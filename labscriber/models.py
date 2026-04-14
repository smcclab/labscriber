# labscriber/models.py
from labscriber.config import detect_device


def download_all(whisper_model: str, hf_token: str) -> None:
    """Download and cache all ML models needed for the pipeline.

    Safe to re-run — existing cached models are not re-downloaded.
    """
    import whisperx

    device, compute_type = detect_device()

    print(f"Downloading Whisper model '{whisper_model}' (device={device}, compute_type={compute_type})...")
    whisperx.load_model(whisper_model, device, compute_type=compute_type)
    print("  ✓ ASR model ready.")

    print("Downloading forced-alignment model for English...")
    whisperx.load_align_model(language_code="en", device=device)
    print("  ✓ Alignment model ready.")

    print("Downloading pyannote diarization pipeline (requires HF_TOKEN)...")
    from whisperx.diarize import DiarizationPipeline
    DiarizationPipeline(token=hf_token, device=device)
    print("  ✓ Diarization pipeline ready.")

    print("\nAll models downloaded. You're ready to run: labscriber process")
