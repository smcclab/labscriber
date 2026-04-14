# labscriber/pipeline.py
import datetime
import logging
import time
import wave
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

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

console = Console()

_STAGE_NAMES = [
    "Ingest (convert to WAV)",
    "Transcribe (ASR)",
    "Diarize (speaker detection)",
    "Merge & write transcripts",
]


def _make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


class _PipelineDisplay:
    """Manages the rich live display for the pipeline."""

    def __init__(self) -> None:
        self._stage_idx: int = -1
        self._stage_elapsed: list[float | None] = [None, None, None, None]
        self._stage_start: float = 0.0
        self.progress: Progress = _make_progress()
        self._task_id: int | None = None
        self._file_n: int = 0
        self._file_total: int = 0
        self._errors: list[str] = []

    def _stage_text(self) -> Text:
        text = Text()
        for i, name in enumerate(_STAGE_NAMES):
            if i < self._stage_idx:
                elapsed = self._stage_elapsed[i]
                suffix = f"  ({_fmt_duration(elapsed)})" if elapsed is not None else ""
                text.append(f"  ✓  Stage {i + 1}/4: {name}{suffix}\n", style="green")
            elif i == self._stage_idx:
                text.append(f"  ⟳  Stage {i + 1}/4: {name}\n", style="bold yellow")
            else:
                text.append(f"  ·  Stage {i + 1}/4: {name}\n", style="dim")
        return text

    def render(self) -> Group:
        return Group(
            Panel(self._stage_text(), title="[bold]labscriber[/bold]", expand=False),
            self.progress,
        )

    def start_stage(self, idx: int, n_files: int, total_audio: float) -> None:
        self._stage_idx = idx
        self._stage_start = time.monotonic()
        self._file_n = 0
        self._file_total = n_files
        if self._task_id is not None:
            self.progress.remove_task(self._task_id)
        self._task_id = self.progress.add_task(
            f"[0/{n_files}] Starting...",
            total=max(total_audio, 1.0),
        )

    def file_start(self, filename: str) -> None:
        if self._task_id is not None:
            self.progress.update(
                self._task_id,
                description=f"[{self._file_n}/{self._file_total}] {filename}",
            )

    def file_done(self, filename: str, audio_duration: float) -> None:
        self._file_n += 1
        if self._task_id is not None:
            self.progress.advance(self._task_id, audio_duration)
            self.progress.update(
                self._task_id,
                description=f"[{self._file_n}/{self._file_total}] {filename} ✓",
            )

    def complete_stage(self, idx: int) -> None:
        self._stage_elapsed[idx] = time.monotonic() - self._stage_start
        if self._task_id is not None:
            self.progress.remove_task(self._task_id)
            self._task_id = None

    def add_error(self, message: str) -> None:
        self._errors.append(message)

    def print_summary(self, n_files: int, total_elapsed: float) -> None:
        lines = Text()
        lines.append(f"  Files processed : {n_files}\n")
        lines.append(f"  Total time      : {_fmt_duration(total_elapsed)}\n")
        if not self._errors:
            lines.append("  Errors          : 0\n", style="green")
        else:
            lines.append(f"  Errors          : {len(self._errors)}\n", style="red")
        console.print(
            Panel(lines, title="[bold green]labscriber — done[/bold green]", expand=False)
        )
        if self._errors:
            console.print("\n[red]Errors:[/red]")
            for err in self._errors:
                console.print(f"  [red]•[/red] {err}")


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


def _get_audio_duration(wav_path: Path) -> float:
    """Return duration in seconds of a WAV file."""
    with wave.open(str(wav_path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def _fmt_duration(seconds: float) -> str:
    """Format seconds as m:ss or h:mm:ss."""
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


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
