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
    TaskID,
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


def _fmt_duration(seconds: float) -> str:
    """Format seconds as m:ss or h:mm:ss."""
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


class _PipelineDisplay:
    """Manages the rich live display for the pipeline."""

    def __init__(self) -> None:
        self._stage_idx: int = -1
        self._stage_elapsed: list[float | None] = [None, None, None, None]
        self._stage_start: float = 0.0
        self.progress: Progress = _make_progress()
        self._task_id: TaskID | None = None
        self._file_n: int = 0
        self._file_total: int = 0
        self._errors: list[str] = []

    def _stage_text(self) -> Text:
        text = Text()
        for i, name in enumerate(_STAGE_NAMES):
            if i < self._stage_idx:
                elapsed = self._stage_elapsed[i]
                suffix = f"  ({_fmt_duration(elapsed)})" if elapsed is not None else ""
                text.append(f"  ✓  Stage {i + 1}/{len(_STAGE_NAMES)}: {name}{suffix}\n", style="green")
            elif i == self._stage_idx:
                text.append(f"  ⟳  Stage {i + 1}/{len(_STAGE_NAMES)}: {name}\n", style="bold yellow")
            else:
                text.append(f"  ·  Stage {i + 1}/{len(_STAGE_NAMES)}: {name}\n", style="dim")
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


def process(config: Config) -> None:
    """Run the full ingest → transcribe → diarize → merge → output pipeline."""
    if config.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            handlers=[RichHandler(console=console, show_path=False)],
        )

    display = _PipelineDisplay()
    run_start = time.monotonic()

    with Live(display.render(), console=console, refresh_per_second=10) as live:

        def _refresh() -> None:
            live.update(display.render())

        # ── Discover files ────────────────────────────────────────────────────
        audio_files = discover_audio_files(config.ingest_path)
        if not audio_files:
            console.print(
                f"[yellow]No audio files found in {config.ingest_path}. "
                "Place files there and re-run.[/yellow]"
            )
            return

        wav_dir = config.work_path / "wav"
        asr_dir = config.work_path / "asr"
        diar_dir = config.work_path / "diarization"
        for d in (wav_dir, asr_dir, diar_dir, config.output_path):
            d.mkdir(parents=True, exist_ok=True)

        # ── Stage 1: Ingest ───────────────────────────────────────────────────
        display.start_stage(0, len(audio_files), float(len(audio_files)))
        _refresh()

        for audio_path in audio_files:
            wav_path = wav_dir / (audio_path.stem + ".wav")
            display.file_start(audio_path.name)
            _refresh()
            if not (wav_path.exists() and not config.force):
                try:
                    convert_to_wav(audio_path, wav_path)
                except Exception as exc:
                    display.add_error(f"Convert {audio_path.name}: {exc}")
            display.file_done(audio_path.name, 1.0)
            _refresh()

        display.complete_stage(0)
        _refresh()

        wav_files = sorted(wav_dir.glob("*.wav"))
        if not wav_files:
            console.print(
                "[red]No WAV files after conversion. Is ffmpeg installed?[/red]"
            )
            return

        # Build audio duration map (used for ETA weighting in stages 2–4)
        durations: dict[str, float] = {}
        for wav_path in wav_files:
            try:
                durations[wav_path.stem] = _get_audio_duration(wav_path)
            except Exception:
                durations[wav_path.stem] = 0.0
        total_audio = sum(durations.values())

        # ── Stage 2: Transcribe ───────────────────────────────────────────────
        display.start_stage(1, len(wav_files), total_audio)
        _refresh()

        asr_model = None
        for wav_path in wav_files:
            asr_path = asr_dir / (wav_path.stem + ".json")
            if asr_path.exists() and not config.force:
                display.file_done(wav_path.name, durations.get(wav_path.stem, 0.0))
                _refresh()
                continue
            if asr_model is None:
                display.file_start(f"{wav_path.name} (loading model...)")
                _refresh()
                asr_model = load_asr_model(
                    config.model, config.device, config.compute_type, config.language
                )
            display.file_start(wav_path.name)
            _refresh()
            try:
                result = transcribe_file(asr_model, wav_path, config.language, config.device)
                save_asr(result, asr_path)
            except Exception as exc:
                display.add_error(f"ASR {wav_path.name}: {exc}")
            display.file_done(wav_path.name, durations.get(wav_path.stem, 0.0))
            _refresh()

        display.complete_stage(1)
        _refresh()

        # ── Stage 3: Diarize ──────────────────────────────────────────────────
        if not config.no_diarize:
            if not config.hf_token:
                console.print(
                    "[red][ERROR] HF_TOKEN not set. "
                    "Run 'labscriber setup' or use --no-diarize.[/red]"
                )
                return
            display.start_stage(2, len(wav_files), total_audio)
            _refresh()

            diarize_model = None
            for wav_path in wav_files:
                diar_path = diar_dir / (wav_path.stem + ".json")
                if diar_path.exists() and not config.force:
                    display.file_done(wav_path.name, durations.get(wav_path.stem, 0.0))
                    _refresh()
                    continue
                if diarize_model is None:
                    display.file_start(f"{wav_path.name} (loading pipeline...)")
                    _refresh()
                    diarize_model = load_diarize_model(config.hf_token, config.device)
                display.file_start(wav_path.name)
                _refresh()
                try:
                    result = diarize_file(
                        diarize_model, wav_path,
                        config.min_speakers, config.max_speakers
                    )
                    save_diarization(result, diar_path)
                except Exception as exc:
                    display.add_error(f"Diarize {wav_path.name}: {exc}")
                display.file_done(wav_path.name, durations.get(wav_path.stem, 0.0))
                _refresh()

            display.complete_stage(2)
            _refresh()
        else:
            # Mark stage skipped so the stage panel shows it as complete
            display.start_stage(2, 0, 1.0)
            display.complete_stage(2)
            _refresh()

        # ── Stage 4: Merge & Output ───────────────────────────────────────────
        display.start_stage(3, len(wav_files), total_audio)
        _refresh()

        today = datetime.date.today().isoformat()
        for wav_path in wav_files:
            stem = wav_path.stem
            md_path = config.output_path / (stem + ".md")
            txt_path = config.output_path / (stem + ".txt")

            display.file_start(wav_path.name)
            _refresh()

            if md_path.exists() and txt_path.exists() and not config.force:
                display.file_done(wav_path.name, durations.get(stem, 0.0))
                _refresh()
                continue

            asr_path = asr_dir / (stem + ".json")
            if not asr_path.exists():
                display.add_error(f"No ASR output for {stem} — skipped.")
                display.file_done(wav_path.name, durations.get(stem, 0.0))
                _refresh()
                continue

            try:
                asr_data = load_asr(asr_path)
                diar_path = diar_dir / (stem + ".json")

                if config.no_diarize or not diar_path.exists():
                    if not config.no_diarize and not diar_path.exists():
                        display.add_error(
                            f"No diarization for {stem}; writing without speaker labels."
                        )
                    utterances = _fallback_utterances(asr_data)
                else:
                    diar_data = load_diarization(diar_path)
                    utterances = merge(asr_data, diar_data, config.merge_gap)

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
                display.add_error(f"Output {stem}: {exc}")

            display.file_done(wav_path.name, durations.get(stem, 0.0))
            _refresh()

        display.complete_stage(3)

    # Live context closed — print static summary
    total_elapsed = time.monotonic() - run_start
    display.print_summary(len(wav_files), total_elapsed)
