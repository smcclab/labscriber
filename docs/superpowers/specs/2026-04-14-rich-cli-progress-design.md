# Rich CLI Progress Display — Design Spec

**Status:** Approved  
**Date:** 2026-04-14  
**Project:** labscriber

---

## Overview

Replace the bare `print()` and `tqdm` calls in `pipeline.py` with a `rich`-powered live display that makes it obvious whether the process is running, crashed, or finished, and provides per-file time estimates based on audio duration.

---

## Problem

The current `process` command gives no signal that it is still alive during slow operations (model loading, ASR, diarization). A user checking back after several minutes cannot tell whether the process has crashed or is running normally. There is also no estimate of how long remains.

---

## Design

### Live display structure

A `rich.Live` context wraps the entire `process()` function. It renders a vertically stacked layout:

1. **Stage panel** — four labelled stages; the active stage has a spinner, completed stages show a checkmark, pending stages are dimmed.
2. **File progress bar** — a `rich.Progress` bar tracking files through the current stage with columns: filename (truncated), audio duration, elapsed, ETA.

The display is always animating while the process is active (the spinner ticks), which answers the "has it crashed?" question at a glance.

### Model loading state

Before the first file is processed in a stage, the file bar shows a spinner with "Loading model…" text. No ETA is shown until the first file completes, at which point the real-time factor is known.

### Per-file time estimation

Audio duration is read from each WAV file using the `wave` stdlib module (no extra dependency) immediately after ingest. This is stored as a `dict[stem → float]` before the ASR and diarization stages begin.

After each file completes a stage, a real-time factor is computed:

```
rtf = wall_time_seconds / audio_duration_seconds
```

For remaining files, estimated duration is `rtf × audio_duration`. The `rich.Progress` ETA column uses the sum of these estimates as its target. This is more accurate than naive throughput averaging because it accounts for the fact that longer audio files take proportionally longer.

For ingest and merge+output stages (fast), elapsed time is shown but ETA is omitted — these stages are not the bottleneck.

### Summary panel

When `process()` completes, the Live display closes and a static summary panel prints:

```
─────────────────────────────────────
 labscriber — done
 Files processed : 3
 Total time      : 18m 04s
 Errors          : 0
─────────────────────────────────────
```

If any files had errors or were skipped, they are listed below the summary with their reason.

### Verbose mode

When `--verbose` is passed, a `rich.logging.RichHandler` is attached so log output appears inline without scrambling the progress display.

### Non-TTY fallback

`rich` detects non-TTY environments automatically and falls back to plain line output. No special handling required.

---

## Scope

### Changed files

| File | Change |
|---|---|
| `labscriber/pipeline.py` | Replace `print()` + `tqdm` with `rich.Live` display; add `_get_audio_duration()` helper; add timing + RTF logic; add summary panel |
| `pyproject.toml` | Add `rich` dependency |

### Unchanged files

`cli.py`, `transcribe.py`, `diarize.py`, `merge.py`, `output.py`, `ingest.py`, `models.py`, `setup_wizard.py`, `config.py` — no changes. This is a presentation-layer change only.

---

## Out of scope

- Changes to `labscriber status` command
- Per-word or per-segment progress within a file (WhisperX provides no hooks for this)
- Confidence scores or quality metrics in output
