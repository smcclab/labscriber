# labscriber/pipeline.py
import datetime
from pathlib import Path

from tqdm import tqdm

from labscriber.config import Config
from labscriber.diarize import (
    diarize_file,
    load_diarization,
    load_diarize_model,
    save_diarization,
)
from labscriber.ingest import convert_to_wav, discover_audio_files
from labscriber.merge import merge
from labscriber.output import write_markdown, write_plaintext
from labscriber.transcribe import (
    load_asr,
    load_asr_model,
    save_asr,
    transcribe_file,
)


def _load_speakers(stem: str, ingest_path: Path) -> dict[str, str]:
    """Return speaker label substitutions from <stem>.speakers.txt, if present.

    File format: one name per line; line 1 → SPEAKER_00, line 2 → SPEAKER_01, …
    """
    sidecar = ingest_path / f"{stem}.speakers.txt"
    if not sidecar.exists():
        return {}
    names = [line.strip() for line in sidecar.read_text(encoding="utf-8").splitlines()]
    return {
        f"SPEAKER_{i:02d}": name
        for i, name in enumerate(names)
        if name
    }


def _fallback_utterances(asr_data: dict) -> list[dict]:
    """Build single-speaker utterances from ASR segments (no diarization)."""
    utterances = []
    for seg in asr_data.get("segments", []):
        utterances.append(
            {
                "speaker": "SPEAKER_00",
                "start": seg["start"],
                "end": seg["end"],
                "text": seg.get("text", "").strip(),
            }
        )
    return utterances


def process(config: Config) -> None:
    """Run the full ingest → transcribe → diarize → merge → output pipeline."""
    # ── Discover files ────────────────────────────────────────────────────────
    audio_files = discover_audio_files(config.ingest_path)
    if not audio_files:
        print(f"No audio files found in {config.ingest_path}. Place files there and re-run.")
        return

    # ── Create working directories ────────────────────────────────────────────
    wav_dir = config.work_path / "wav"
    asr_dir = config.work_path / "asr"
    diar_dir = config.work_path / "diarization"
    for d in (wav_dir, asr_dir, diar_dir, config.output_path):
        d.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: Ingest (convert to WAV) ─────────────────────────────────────
    print("Stage 1/4: Converting audio to WAV...")
    for audio_path in tqdm(audio_files, unit="file"):
        wav_path = wav_dir / (audio_path.stem + ".wav")
        if wav_path.exists() and not config.force:
            continue
        try:
            convert_to_wav(audio_path, wav_path)
        except Exception as exc:
            print(f"  [ERROR] Could not convert {audio_path.name}: {exc}")

    wav_files = sorted(wav_dir.glob("*.wav"))
    if not wav_files:
        print("No WAV files available after conversion. Check ffmpeg is installed.")
        return

    # ── Stage 2: Transcribe (ASR) ─────────────────────────────────────────────
    print("Stage 2/4: Transcribing (ASR)...")
    asr_model = None
    for wav_path in tqdm(wav_files, unit="file"):
        asr_path = asr_dir / (wav_path.stem + ".json")
        if asr_path.exists() and not config.force:
            continue
        if asr_model is None:
            print(f"  Loading WhisperX model '{config.model}'...")
            asr_model = load_asr_model(config.model, config.device, config.compute_type, config.language)
        try:
            result = transcribe_file(asr_model, wav_path, config.language, config.device)
            save_asr(result, asr_path)
        except Exception as exc:
            print(f"  [ERROR] Transcription failed for {wav_path.name}: {exc}")

    # ── Stage 3: Diarize ──────────────────────────────────────────────────────
    if not config.no_diarize:
        if not config.hf_token:
            print(
                "\n[ERROR] HF_TOKEN not set. Diarization requires a HuggingFace token.\n"
                "  Run: labscriber setup\n"
                "  Or use: labscriber process --no-diarize"
            )
            return
        print("Stage 3/4: Diarizing (speaker detection)...")
        diarize_model = None
        for wav_path in tqdm(wav_files, unit="file"):
            diar_path = diar_dir / (wav_path.stem + ".json")
            if diar_path.exists() and not config.force:
                continue
            if diarize_model is None:
                print("  Loading diarization pipeline...")
                diarize_model = load_diarize_model(config.hf_token, config.device)
            try:
                result = diarize_file(
                    diarize_model, wav_path, config.min_speakers, config.max_speakers
                )
                save_diarization(result, diar_path)
            except Exception as exc:
                print(f"  [ERROR] Diarization failed for {wav_path.name}: {exc}")
    else:
        print("Stage 3/4: Diarization skipped (--no-diarize).")

    # ── Stages 4+5: Merge and Output ──────────────────────────────────────────
    print("Stage 4/4: Merging and writing transcripts...")
    today = datetime.date.today().isoformat()

    for wav_path in tqdm(wav_files, unit="file"):
        stem = wav_path.stem
        md_path = config.output_path / (stem + ".md")
        txt_path = config.output_path / (stem + ".txt")

        if md_path.exists() and txt_path.exists() and not config.force:
            continue

        asr_path = asr_dir / (stem + ".json")
        if not asr_path.exists():
            print(f"  [SKIP] No ASR output for {stem}.")
            continue

        try:
            asr_data = load_asr(asr_path)
            diar_path = diar_dir / (stem + ".json")

            if config.no_diarize or not diar_path.exists():
                if not config.no_diarize and not diar_path.exists():
                    print(f"  [WARN] No diarization for {stem}; writing without speaker labels.")
                utterances = _fallback_utterances(asr_data)
            else:
                diar_data = load_diarization(diar_path)
                utterances = merge(asr_data, diar_data, config.merge_gap)

            # Apply speakers.txt sidecar substitutions
            speaker_map = _load_speakers(stem, config.ingest_path)
            if speaker_map:
                utterances = [
                    {**utt, "speaker": speaker_map.get(utt["speaker"], utt["speaker"])}
                    for utt in utterances
                ]

            num_speakers = len({utt["speaker"] for utt in utterances})
            duration = utterances[-1]["end"] if utterances else 0.0
            meta = {
                "title": stem,
                "date": today,
                "language": asr_data.get("language", "unknown"),
                "num_speakers": num_speakers,
                "duration": duration,
            }

            write_markdown(utterances, meta, md_path)
            write_plaintext(utterances, meta, txt_path)

        except Exception as exc:
            print(f"  [ERROR] Output failed for {stem}: {exc}")

    print("Done.")
