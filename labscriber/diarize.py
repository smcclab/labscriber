import json
from pathlib import Path


def load_diarize_model(hf_token: str, device: str):
    """Load pyannote diarization pipeline via WhisperX. Call once, reuse across files."""
    from whisperx.diarize import DiarizationPipeline
    return DiarizationPipeline(token=hf_token, device=device)


def diarize_file(
    model,
    wav_path: Path,
    min_speakers: int,
    max_speakers: int | None,
) -> dict:
    """Run speaker diarization on a WAV file.

    Returns:
        {"segments": [{"start": float, "end": float, "speaker": str}, ...]}
    """
    kwargs: dict = {"min_speakers": min_speakers}
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    diarize_df = model(str(wav_path), **kwargs)

    # diarize_df is a pandas DataFrame with columns: segment (Segment), label, speaker
    segments = []
    for _, row in diarize_df.iterrows():
        seg = row["segment"]
        segments.append(
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "speaker": str(row["speaker"]),
            }
        )

    return {"segments": segments}


def save_diarization(result: dict, path: Path) -> None:
    """Write diarization result dict to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def load_diarization(path: Path) -> dict:
    """Read diarization result dict from JSON."""
    return json.loads(path.read_text(encoding="utf-8"))
