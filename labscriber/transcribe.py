import json
from pathlib import Path


def load_asr_model(model_name: str, device: str, compute_type: str):
    """Load and return a WhisperX ASR model. Call once, reuse across files."""
    import whisperx
    return whisperx.load_model(model_name, device, compute_type=compute_type)


def transcribe_file(model, wav_path: Path, language: str | None, device: str) -> dict:
    """Run ASR + forced alignment on a WAV file.

    Returns:
        {"language": "en", "segments": [...with per-word timestamps...]}
    """
    import whisperx

    audio = whisperx.load_audio(str(wav_path))

    # ASR pass
    result = model.transcribe(audio, language=language)
    detected_language = result["language"]

    # Forced alignment for word-level timestamps
    align_model, metadata = whisperx.load_align_model(
        language_code=detected_language, device=device
    )
    aligned = whisperx.align(result["segments"], align_model, metadata, audio, device)

    return {"language": detected_language, "segments": aligned["segments"]}


def save_asr(result: dict, path: Path) -> None:
    """Write ASR result dict to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def load_asr(path: Path) -> dict:
    """Read ASR result dict from JSON."""
    return json.loads(path.read_text(encoding="utf-8"))
