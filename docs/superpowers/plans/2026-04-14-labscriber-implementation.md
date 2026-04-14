# labscriber Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local batch audio transcription CLI tool that processes multi-speaker interview recordings and produces speaker-labelled Markdown and plain-text transcripts.

**Architecture:** Five-stage pipeline (ingest → transcribe → diarize → merge → output) where each stage checks for existing outputs and skips unless `--force` is set. WhisperX provides ASR with word-level timestamps; pyannote via WhisperX handles speaker diarization. A pure merge function assigns speaker labels to words by time overlap, then renders two output formats.

**Tech Stack:** Python 3.11, Poetry, WhisperX (faster-whisper backend), pyannote.audio 3.1, ffmpeg-python, tqdm, python-dotenv, pytest

---

## File Map

| File | Responsibility |
|---|---|
| `pyproject.toml` | Dependencies, scripts entry point |
| `labscriber/__main__.py` | `python -m labscriber` entry |
| `labscriber/config.py` | `Config` dataclass, `detect_device()` |
| `labscriber/ingest.py` | `discover_audio_files()`, `convert_to_wav()` |
| `labscriber/merge.py` | `merge()` — pure word-to-speaker assignment |
| `labscriber/output.py` | `write_markdown()`, `write_plaintext()`, `format_timestamp()` |
| `labscriber/transcribe.py` | `load_asr_model()`, `transcribe_file()`, `save_asr()`, `load_asr()` |
| `labscriber/diarize.py` | `load_diarize_model()`, `diarize_file()`, `save_diarization()`, `load_diarization()` |
| `labscriber/models.py` | `download_all()` — download and cache all ML models |
| `labscriber/setup_wizard.py` | `run_setup()`, `validate_token()` — HuggingFace onboarding |
| `labscriber/pipeline.py` | `process(config)` — orchestrates all stages |
| `labscriber/cli.py` | `main()` — argparse dispatch, `.env` loading |
| `tests/test_merge.py` | Unit tests for merge algorithm |
| `tests/test_output.py` | Unit tests for output rendering |

---

## Task 1: Project Scaffolding

**Files:**
- Modify: `pyproject.toml`
- Create: `labscriber/__init__.py`
- Create: `labscriber/__main__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Update pyproject.toml with dependencies and script entry point**

Replace the contents of `pyproject.toml` with:

```toml
[project]
name = "labscriber"
version = "0.1.0"
description = "Local batch transcription for multi-speaker HCI research recordings"
authors = [
    {name = "Charles Martin", email = "cpm@charlesmartin.com.au"}
]
readme = "README.md"
requires-python = ">=3.11,<3.13"
dependencies = [
    "whisperx",
    "torch>=2.0",
    "torchaudio>=2.0",
    "pyannote.audio>=3.1",
    "faster-whisper",
    "ffmpeg-python",
    "tqdm",
    "python-dotenv",
]

[project.scripts]
labscriber = "labscriber.cli:main"

[build-system]
requires = ["poetry-core>=2.0.0,<3.0.0"]
build-backend = "poetry.core.masonry.api"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Install dependencies**

```bash
poetry install
```

Expected: resolves and installs all dependencies without error. Note: torch download is large (~2 GB on first install).

- [ ] **Step 3: Create package and test stubs**

Create `labscriber/__init__.py` (empty):
```python
```

Create `labscriber/__main__.py`:
```python
from labscriber.cli import main

main()
```

Create `tests/__init__.py` (empty):
```python
```

- [ ] **Step 4: Verify import works**

```bash
poetry run python -c "import labscriber"
```

Expected: no output, no error.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml labscriber/__init__.py labscriber/__main__.py tests/__init__.py
git commit -m "feat: scaffold labscriber package with dependencies"
```

---

## Task 2: config.py — Config dataclass and hardware detection

**Files:**
- Create: `labscriber/config.py`

- [ ] **Step 1: Create config.py**

```python
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

    if torch.backends.mps.is_available():
        return "mps", "float16"
    elif torch.cuda.is_available():
        return "cuda", "float16"
    else:
        print("Notice: No GPU detected. Processing will be slow. Consider --model medium.")
        return "cpu", "int8"
```

- [ ] **Step 2: Verify import**

```bash
poetry run python -c "from labscriber.config import Config, detect_device; print(detect_device())"
```

Expected: prints a tuple like `('mps', 'float16')` or `('cpu', 'int8')`.

- [ ] **Step 3: Commit**

```bash
git add labscriber/config.py
git commit -m "feat: add Config dataclass and hardware detection"
```

---

## Task 3: ingest.py — Audio file discovery and ffmpeg conversion

**Files:**
- Create: `labscriber/ingest.py`

- [ ] **Step 1: Create ingest.py**

```python
# labscriber/ingest.py
from pathlib import Path

import ffmpeg

AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".mp4", ".m4a", ".aac",
    ".flac", ".ogg", ".aiff", ".wma",
}


def discover_audio_files(ingest_path: Path) -> list[Path]:
    """Return sorted list of audio files in ingest_path, skipping duplicate stems."""
    files: list[Path] = []
    seen_stems: dict[str, Path] = {}

    for path in sorted(ingest_path.rglob("*")):
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        stem = path.stem
        if stem in seen_stems:
            print(f"Warning: duplicate stem '{stem}' — skipping {path} (keeping {seen_stems[stem]})")
            continue
        seen_stems[stem] = path
        files.append(path)

    return files


def convert_to_wav(input_path: Path, output_path: Path) -> None:
    """Convert any audio file to 16 kHz mono 16-bit PCM WAV via ffmpeg."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    (
        ffmpeg
        .input(str(input_path))
        .output(str(output_path), ar=16000, ac=1, sample_fmt="s16")
        .overwrite_output()
        .run(quiet=True)
    )
```

- [ ] **Step 2: Smoke-test discovery against empty directory**

```bash
mkdir -p /tmp/labtest/ingest
poetry run python -c "
from pathlib import Path
from labscriber.ingest import discover_audio_files
files = discover_audio_files(Path('/tmp/labtest/ingest'))
print('files found:', files)
"
```

Expected: `files found: []`

- [ ] **Step 3: Smoke-test conversion (requires ffmpeg installed)**

```bash
# Generate a 1-second silent WAV using ffmpeg directly to test the path
ffmpeg -f lavfi -i anullsrc=r=44100:cl=mono -t 1 /tmp/labtest/ingest/test.mp3 -y
poetry run python -c "
from pathlib import Path
from labscriber.ingest import discover_audio_files, convert_to_wav
files = discover_audio_files(Path('/tmp/labtest/ingest'))
print('found:', files)
convert_to_wav(files[0], Path('/tmp/labtest/work/wav/test.wav'))
print('converted OK')
"
```

Expected: `found: [...]` then `converted OK`

- [ ] **Step 4: Commit**

```bash
git add labscriber/ingest.py
git commit -m "feat: add audio file discovery and ffmpeg WAV conversion"
```

---

## Task 4: merge.py (TDD) — Speaker assignment algorithm

**Files:**
- Create: `tests/test_merge.py`
- Create: `labscriber/merge.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_merge.py`:

```python
# tests/test_merge.py
import pytest
from labscriber.merge import merge


@pytest.fixture
def two_speaker_asr():
    return {
        "language": "en",
        "segments": [
            {
                "start": 0.0,
                "end": 5.0,
                "text": "Hello world how are you",
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.5},
                    {"word": "world", "start": 0.6, "end": 1.0},
                    {"word": "how", "start": 2.0, "end": 2.5},
                    {"word": "are", "start": 2.6, "end": 2.9},
                    {"word": "you", "start": 3.0, "end": 3.5},
                ],
            }
        ],
    }


@pytest.fixture
def two_speaker_diar():
    return {
        "segments": [
            {"start": 0.0, "end": 1.2, "speaker": "SPEAKER_00"},
            {"start": 1.8, "end": 3.6, "speaker": "SPEAKER_01"},
        ]
    }


def test_merge_assigns_speakers_by_overlap(two_speaker_asr, two_speaker_diar):
    result = merge(two_speaker_asr, two_speaker_diar, merge_gap=1.5)
    assert len(result) >= 2
    assert result[0]["speaker"] == "SPEAKER_00"
    assert result[-1]["speaker"] == "SPEAKER_01"


def test_merge_returns_required_fields(two_speaker_asr, two_speaker_diar):
    result = merge(two_speaker_asr, two_speaker_diar)
    for utt in result:
        assert "speaker" in utt
        assert "start" in utt
        assert "end" in utt
        assert "text" in utt


def test_merge_gap_joins_same_speaker_segments():
    asr = {
        "segments": [
            {
                "start": 0.0,
                "end": 5.0,
                "text": "a b c d",
                "words": [
                    {"word": "a", "start": 0.0, "end": 0.5},
                    {"word": "b", "start": 1.0, "end": 1.5},
                    # 1.5-second gap (== merge_gap=2.0, so should merge)
                    {"word": "c", "start": 3.0, "end": 3.5},
                    {"word": "d", "start": 4.0, "end": 4.5},
                ],
            }
        ]
    }
    diar = {"segments": [{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"}]}
    result = merge(asr, diar, merge_gap=2.0)
    assert len(result) == 1
    assert result[0]["speaker"] == "SPEAKER_00"
    assert result[0]["start"] == 0.0
    assert result[0]["end"] == 4.5


def test_merge_gap_splits_same_speaker_segments():
    asr = {
        "segments": [
            {
                "start": 0.0,
                "end": 10.0,
                "text": "a b",
                "words": [
                    {"word": "a", "start": 0.0, "end": 0.5},
                    # 5-second gap exceeds merge_gap=1.5, same speaker → split
                    {"word": "b", "start": 5.5, "end": 6.0},
                ],
            }
        ]
    }
    diar = {"segments": [{"start": 0.0, "end": 10.0, "speaker": "SPEAKER_00"}]}
    result = merge(asr, diar, merge_gap=1.5)
    assert len(result) == 2
    assert all(u["speaker"] == "SPEAKER_00" for u in result)


def test_merge_empty_asr():
    asr = {"segments": []}
    diar = {"segments": [{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"}]}
    result = merge(asr, diar)
    assert result == []


def test_merge_no_overlap_uses_nearest_segment():
    asr = {
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "hello",
                "words": [{"word": "hello", "start": 0.2, "end": 0.8}],
            }
        ]
    }
    # Word midpoint at 0.5; nearest diar segment midpoint at 7.5
    diar = {"segments": [{"start": 5.0, "end": 10.0, "speaker": "SPEAKER_00"}]}
    result = merge(asr, diar)
    assert len(result) == 1
    assert result[0]["speaker"] == "SPEAKER_00"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
poetry run pytest tests/test_merge.py -v
```

Expected: ERRORS — `ModuleNotFoundError: No module named 'labscriber.merge'`

- [ ] **Step 3: Implement merge.py**

Create `labscriber/merge.py`:

```python
# labscriber/merge.py


def merge(
    asr_data: dict,
    diarization_data: dict,
    merge_gap: float = 1.5,
) -> list[dict]:
    """Assign speaker labels to ASR words and group into utterances.

    Args:
        asr_data: WhisperX ASR JSON with 'segments' containing 'words'.
        diarization_data: Diarization JSON with 'segments' of {start, end, speaker}.
        merge_gap: Max silence (seconds) between same-speaker segments before splitting.

    Returns:
        List of utterance dicts: {speaker, start, end, text}.
    """
    # Flatten all words with timestamps
    words: list[dict] = []
    for seg in asr_data.get("segments", []):
        for word in seg.get("words", []):
            if "start" in word and "end" in word:
                words.append(dict(word))

    if not words:
        return []

    diar_segs = diarization_data.get("segments", [])

    # Stage 1: assign each word to the speaker with the greatest time overlap
    for word in words:
        best_speaker: str | None = None
        best_overlap: float = -1.0

        for seg in diar_segs:
            overlap = max(0.0, min(word["end"], seg["end"]) - max(word["start"], seg["start"]))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = seg["speaker"]

        if best_speaker is None or best_overlap <= 0.0:
            # No overlap — fall back to nearest segment by midpoint distance
            word_mid = (word["start"] + word["end"]) / 2.0
            if diar_segs:
                best_speaker = min(
                    diar_segs,
                    key=lambda s: abs((s["start"] + s["end"]) / 2.0 - word_mid),
                )["speaker"]
            else:
                best_speaker = "SPEAKER_00"

        word["speaker"] = best_speaker

    # Stage 2: group consecutive same-speaker words into raw utterances
    utterances: list[dict] = []
    for word in words:
        if utterances and utterances[-1]["speaker"] == word["speaker"]:
            utterances[-1]["end"] = word["end"]
            utterances[-1]["text"] += " " + word["word"]
        else:
            utterances.append(
                {
                    "speaker": word["speaker"],
                    "start": word["start"],
                    "end": word["end"],
                    "text": word["word"],
                }
            )

    # Stage 3: merge adjacent same-speaker utterances within merge_gap
    merged: list[dict] = []
    for utt in utterances:
        if (
            merged
            and merged[-1]["speaker"] == utt["speaker"]
            and utt["start"] - merged[-1]["end"] < merge_gap
        ):
            merged[-1]["end"] = utt["end"]
            merged[-1]["text"] += " " + utt["text"]
        else:
            merged.append(dict(utt))

    return merged
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
poetry run pytest tests/test_merge.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_merge.py labscriber/merge.py
git commit -m "feat: implement merge algorithm with TDD (6 tests passing)"
```

---

## Task 5: output.py (TDD) — Markdown and plain-text rendering

**Files:**
- Create: `tests/test_output.py`
- Create: `labscriber/output.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_output.py`:

```python
# tests/test_output.py
import pytest
from pathlib import Path
from labscriber.output import format_timestamp, write_markdown, write_plaintext


# ── format_timestamp ──────────────────────────────────────────────────────────

def test_format_timestamp_short_zero():
    assert format_timestamp(3.0, short=True) == "0:03"


def test_format_timestamp_short_nonzero_minutes():
    assert format_timestamp(63.0, short=True) == "1:03"


def test_format_timestamp_long():
    # 1 hour + 2 min + 3 sec = 3723 seconds
    assert format_timestamp(3723.0, short=False) == "1:02:03"


def test_format_timestamp_long_zero_hours():
    # short=False means H:MM:SS always, even for 0 hours
    assert format_timestamp(63.0, short=False) == "0:01:03"


def test_format_timestamp_defaults_to_long():
    # Default is short=False → H:MM:SS
    assert format_timestamp(63.0) == "0:01:03"


# ── write_markdown ────────────────────────────────────────────────────────────

@pytest.fixture
def sample_utterances():
    return [
        {"speaker": "SPEAKER_00", "start": 3.12, "end": 18.44, "text": "Hello there."},
        {"speaker": "SPEAKER_01", "start": 19.10, "end": 45.22, "text": "Hi back."},
    ]


@pytest.fixture
def sample_meta():
    return {
        "title": "interview-01",
        "date": "2026-04-14",
        "language": "english",
        "num_speakers": 2,
        "duration": 45.22,
    }


def test_write_markdown_creates_file(tmp_path, sample_utterances, sample_meta):
    out = tmp_path / "interview-01.md"
    write_markdown(sample_utterances, sample_meta, out)
    assert out.exists()


def test_write_markdown_contains_title(tmp_path, sample_utterances, sample_meta):
    out = tmp_path / "interview-01.md"
    write_markdown(sample_utterances, sample_meta, out)
    assert "# interview-01" in out.read_text()


def test_write_markdown_contains_speakers(tmp_path, sample_utterances, sample_meta):
    out = tmp_path / "interview-01.md"
    write_markdown(sample_utterances, sample_meta, out)
    content = out.read_text()
    assert "**SPEAKER_00**" in content
    assert "**SPEAKER_01**" in content


def test_write_markdown_contains_text(tmp_path, sample_utterances, sample_meta):
    out = tmp_path / "interview-01.md"
    write_markdown(sample_utterances, sample_meta, out)
    content = out.read_text()
    assert "Hello there." in content
    assert "Hi back." in content


def test_write_markdown_contains_metadata(tmp_path, sample_utterances, sample_meta):
    out = tmp_path / "interview-01.md"
    write_markdown(sample_utterances, sample_meta, out)
    content = out.read_text()
    assert "2026-04-14" in content
    assert "english" in content
    assert "2" in content  # num_speakers


def test_write_markdown_creates_parent_dirs(tmp_path, sample_utterances, sample_meta):
    out = tmp_path / "nested" / "deep" / "interview-01.md"
    write_markdown(sample_utterances, sample_meta, out)
    assert out.exists()


def test_write_markdown_short_timestamps_for_short_recording(tmp_path, sample_utterances):
    # Duration < 3600 → M:SS timestamps
    meta = {
        "title": "t", "date": "2026-04-14", "language": "en",
        "num_speakers": 2, "duration": 45.22,
    }
    out = tmp_path / "t.md"
    write_markdown(sample_utterances, meta, out)
    content = out.read_text()
    # Timestamps like 0:03 not 0:00:03
    assert "0:03" in content
    assert "0:00:03" not in content


# ── write_plaintext ───────────────────────────────────────────────────────────

def test_write_plaintext_creates_file(tmp_path, sample_utterances, sample_meta):
    out = tmp_path / "interview-01.txt"
    write_plaintext(sample_utterances, sample_meta, out)
    assert out.exists()


def test_write_plaintext_header(tmp_path, sample_utterances, sample_meta):
    out = tmp_path / "interview-01.txt"
    write_plaintext(sample_utterances, sample_meta, out)
    lines = out.read_text().splitlines()
    assert "interview-01 — Transcript" in lines[0]
    assert "2026-04-14" in lines[1]
    assert "english" in lines[1]


def test_write_plaintext_utterance_format(tmp_path, sample_utterances, sample_meta):
    out = tmp_path / "interview-01.txt"
    write_plaintext(sample_utterances, sample_meta, out)
    content = out.read_text()
    assert "SPEAKER_00: Hello there." in content
    assert "SPEAKER_01: Hi back." in content


def test_write_plaintext_creates_parent_dirs(tmp_path, sample_utterances, sample_meta):
    out = tmp_path / "nested" / "interview-01.txt"
    write_plaintext(sample_utterances, sample_meta, out)
    assert out.exists()
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
poetry run pytest tests/test_output.py -v
```

Expected: ERRORS — `ModuleNotFoundError: No module named 'labscriber.output'`

- [ ] **Step 3: Implement output.py**

Create `labscriber/output.py`:

```python
# labscriber/output.py
import datetime
from pathlib import Path


def format_timestamp(seconds: float, short: bool = False) -> str:
    """Format seconds as M:SS (short=True) or H:MM:SS (short=False).

    Use short=True when the total recording duration is under one hour,
    so all timestamps are consistently in M:SS format.
    """
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if short:
        return f"{m}:{s:02d}"
    return f"{h}:{m:02d}:{s:02d}"


def write_markdown(utterances: list[dict], meta: dict, output_path: Path) -> None:
    """Write transcript as Markdown.

    meta keys: title, date, language, num_speakers, duration (seconds)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    short = meta["duration"] < 3600

    lines = [
        f"# {meta['title']}",
        "",
        f"**Processed:** {meta['date']}  ",
        f"**Language:** {meta['language']}  ",
        f"**Speakers detected:** {meta['num_speakers']}  ",
        f"**Duration:** {format_timestamp(meta['duration'], short=short)}  ",
        "",
        "---",
        "",
    ]

    for utt in utterances:
        start = format_timestamp(utt["start"], short=short)
        end = format_timestamp(utt["end"], short=short)
        lines.append(f"**{utt['speaker']}** *({start} \u2013 {end})*  ")
        lines.append(utt["text"])
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_plaintext(utterances: list[dict], meta: dict, output_path: Path) -> None:
    """Write transcript as plain text (one utterance per line).

    meta keys: title, date, language, num_speakers, duration (seconds)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    short = meta["duration"] < 3600

    duration_str = format_timestamp(meta["duration"], short=short)
    lines = [
        f"{meta['title']} \u2014 Transcript",
        (
            f"Processed: {meta['date']} | Language: {meta['language']} "
            f"| Speakers: {meta['num_speakers']} | Duration: {duration_str}"
        ),
        "",
    ]

    for utt in utterances:
        start = format_timestamp(utt["start"], short=short)
        lines.append(f"[{start}] {utt['speaker']}: {utt['text']}")

    output_path.write_text("\n".join(lines), encoding="utf-8")
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
poetry run pytest tests/test_output.py -v
```

Expected: all 14 tests PASS.

- [ ] **Step 5: Run full test suite**

```bash
poetry run pytest -v
```

Expected: all 20 tests PASS (6 merge + 14 output).

- [ ] **Step 6: Commit**

```bash
git add tests/test_output.py labscriber/output.py
git commit -m "feat: implement output rendering with TDD (14 tests passing)"
```

---

## Task 6: transcribe.py — WhisperX ASR wrapper

**Files:**
- Create: `labscriber/transcribe.py`

- [ ] **Step 1: Create transcribe.py**

```python
# labscriber/transcribe.py
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
```

- [ ] **Step 2: Verify import (does not run model)**

```bash
poetry run python -c "from labscriber.transcribe import load_asr, save_asr; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add labscriber/transcribe.py
git commit -m "feat: add WhisperX ASR transcription wrapper"
```

---

## Task 7: diarize.py — Speaker diarization wrapper

**Files:**
- Create: `labscriber/diarize.py`

- [ ] **Step 1: Create diarize.py**

```python
# labscriber/diarize.py
import json
from pathlib import Path


def load_diarize_model(hf_token: str, device: str):
    """Load pyannote diarization pipeline via WhisperX. Call once, reuse across files."""
    import whisperx
    return whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)


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
```

- [ ] **Step 2: Verify import**

```bash
poetry run python -c "from labscriber.diarize import load_diarization, save_diarization; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add labscriber/diarize.py
git commit -m "feat: add pyannote diarization wrapper"
```

---

## Task 8: models.py — Model download utility

**Files:**
- Create: `labscriber/models.py`

- [ ] **Step 1: Create models.py**

```python
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
    whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
    print("  ✓ Diarization pipeline ready.")

    print("\nAll models downloaded. You're ready to run: labscriber process")
```

- [ ] **Step 2: Verify import**

```bash
poetry run python -c "from labscriber.models import download_all; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add labscriber/models.py
git commit -m "feat: add model download utility"
```

---

## Task 9: setup_wizard.py — Interactive HuggingFace token setup

**Files:**
- Create: `labscriber/setup_wizard.py`

- [ ] **Step 1: Create setup_wizard.py**

```python
# labscriber/setup_wizard.py
import os
from pathlib import Path


def validate_token(hf_token: str) -> bool:
    """Attempt to load pyannote pipeline to verify the token and model acceptance.

    Returns True if token is valid and model terms have been accepted.
    """
    try:
        import whisperx
        # Use CPU for validation to avoid GPU setup overhead
        whisperx.DiarizationPipeline(use_auth_token=hf_token, device="cpu")
        return True
    except Exception as exc:
        error = str(exc).lower()
        if "401" in error or "unauthorized" in error or "invalid" in error:
            print("  Error: Invalid token or model terms not accepted.")
        elif "403" in error or "forbidden" in error or "gated" in error:
            print("  Error: Model access not granted. Have you accepted the terms at both URLs?")
        else:
            print(f"  Error: {exc}")
        return False


def run_setup(working_dir: Path) -> None:
    """Interactive HuggingFace token setup.

    1. Check if HF_TOKEN is already set.
    2. Print model acceptance URLs.
    3. Prompt for token.
    4. Write to .env.
    5. Validate token by loading pyannote pipeline.
    """
    from dotenv import dotenv_values

    env_path = working_dir / ".env"

    # Check for existing token
    existing = os.environ.get("HF_TOKEN") or dotenv_values(env_path).get("HF_TOKEN")
    if existing:
        print(f"HF_TOKEN is already set in {env_path}.")
        answer = input("Re-enter a new token? [y/N]: ").strip().lower()
        if answer != "y":
            print("Setup skipped. Run 'labscriber download-models' to proceed.")
            return

    print("\n--- labscriber HuggingFace Setup ---\n")
    print("You need a free HuggingFace account with access to two gated models.")
    print("Step 1: Visit each URL below while logged in and click 'Agree':")
    print()
    print("  https://huggingface.co/pyannote/speaker-diarization-3.1")
    print("  https://huggingface.co/pyannote/segmentation-3.0")
    print()
    print("Step 2: Create a read-only access token at:")
    print("  https://huggingface.co/settings/tokens")
    print()
    token = input("Paste your HuggingFace token here: ").strip()

    if not token:
        print("No token entered. Aborting.")
        return

    if not token.startswith("hf_"):
        print("Warning: token usually starts with 'hf_'. Proceeding anyway.")

    print("\nValidating token (downloads ~500 MB on first run — this may take a few minutes)...")
    if not validate_token(token):
        print("Setup failed. Correct the issues above and re-run 'labscriber setup'.")
        return

    # Write to .env
    env_path.write_text(f"HF_TOKEN={token}\n", encoding="utf-8")
    print(f"\n✓ Token saved to {env_path}")
    print("Next step: run 'labscriber download-models' to pre-download all models.")
```

- [ ] **Step 2: Verify import**

```bash
poetry run python -c "from labscriber.setup_wizard import run_setup, validate_token; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add labscriber/setup_wizard.py
git commit -m "feat: add interactive HuggingFace token setup wizard"
```

---

## Task 10: pipeline.py — Pipeline orchestrator

**Files:**
- Create: `labscriber/pipeline.py`

- [ ] **Step 1: Create pipeline.py**

```python
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
            asr_model = load_asr_model(config.model, config.device, config.compute_type)
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
```

- [ ] **Step 2: Verify import**

```bash
poetry run python -c "from labscriber.pipeline import process; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add labscriber/pipeline.py
git commit -m "feat: add pipeline orchestrator with per-stage skip logic"
```

---

## Task 11: cli.py — Argument parsing and command dispatch

**Files:**
- Create: `labscriber/cli.py`

- [ ] **Step 1: Create cli.py**

```python
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
```

- [ ] **Step 2: Verify CLI entry point resolves**

```bash
poetry run labscriber --help
```

Expected: help text listing `setup`, `process`, `download-models`, `status`, `clean` commands.

- [ ] **Step 3: Test status command against empty dir**

```bash
mkdir -p ingest
poetry run labscriber status
```

Expected: `No audio files in ingest`

- [ ] **Step 4: Run full test suite**

```bash
poetry run pytest -v
```

Expected: all 20 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add labscriber/cli.py
git commit -m "feat: add CLI with setup/process/download-models/status/clean commands"
```

---

## Task 12: .env.example and gitignore

**Files:**
- Create: `.env.example`
- Create/modify: `.gitignore`

- [ ] **Step 1: Create .env.example**

```
# HuggingFace access token — required for speaker diarization.
# Get yours at: https://huggingface.co/settings/tokens
# See: labscriber setup
HF_TOKEN=hf_your_token_here
```

- [ ] **Step 2: Create .gitignore**

```
# Python
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.mypy_cache/
dist/
*.egg-info/

# Poetry
.venv/

# labscriber runtime directories
ingest/
work/
output/

# Secrets
.env
```

- [ ] **Step 3: Commit**

```bash
git add .env.example .gitignore
git commit -m "chore: add .env.example and .gitignore"
```

---

## Task 13: Smoke Test — End-to-end with --no-diarize

This verifies the full pipeline executes without ML model errors (ASR model still required).

- [ ] **Step 1: Generate a short test audio file**

```bash
mkdir -p ingest
ffmpeg -f lavfi -i "sine=frequency=440:duration=5" -ar 44100 ingest/test-clip.mp3 -y
```

Expected: `ingest/test-clip.mp3` created (5-second sine wave).

- [ ] **Step 2: Run pipeline with tiny model and --no-diarize**

```bash
poetry run labscriber process --model tiny --no-diarize --language english
```

Expected: stages 1–4 complete, `output/test-clip.md` and `output/test-clip.txt` created.

- [ ] **Step 3: Verify output files exist and have content**

```bash
ls output/
cat output/test-clip.md
cat output/test-clip.txt
```

Expected: both files present; Markdown contains `# test-clip` header; plain text contains `test-clip — Transcript` header.

- [ ] **Step 4: Verify status command shows complete state**

```bash
poetry run labscriber status
```

Expected:
```
ingest/test-clip.mp3  ✓ wav  ✓ asr  ✗ diarize  ✓ output
```

- [ ] **Step 5: Verify idempotency (re-run skips existing)**

```bash
poetry run labscriber process --model tiny --no-diarize --language english
```

Expected: all stages print progress bars but skip all files (already done). No errors.

- [ ] **Step 6: Test --force re-processes**

```bash
poetry run labscriber process --model tiny --no-diarize --language english --force
```

Expected: files are re-processed.

- [ ] **Step 7: Test clean command**

```bash
poetry run labscriber clean
ls
```

Expected: `work/` directory gone; `ingest/` and `output/` unchanged.

- [ ] **Step 8: Final test run**

```bash
poetry run pytest -v
```

Expected: all 20 tests PASS.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "test: smoke test passes — pipeline end-to-end with --no-diarize"
```

---

## Spec Coverage Checklist

| Spec Section | Covered by Task |
|---|---|
| 3.1 WhisperX ASR | Task 6 (transcribe.py) |
| 3.2 pyannote diarization | Task 7 (diarize.py) |
| 3.3 ffmpeg conversion | Task 3 (ingest.py) |
| 3.4 Python/Poetry tooling | Task 1 (pyproject.toml) |
| 4.1 Repository layout | Tasks 1–11 |
| 4.2 Runtime directory layout | Task 10 (pipeline.py) |
| 5.1–5.6 Pipeline stages | Tasks 3, 6, 7, 4, 5, 10 |
| 6.1 Markdown output format | Task 5 (output.py) |
| 6.2 Plain text output format | Task 5 (output.py) |
| 7. Speaker sidecar (speakers.txt) | Task 10 (_load_speakers in pipeline.py) |
| 8.1–8.6 Installation & onboarding | Task 9 (setup_wizard.py), Task 12 |
| 9. CLI reference | Task 11 (cli.py) |
| 10. Dependencies | Task 1 (pyproject.toml) |
| 10. Hardware auto-detection | Task 2 (config.py) |
| 11. Error handling | Tasks 10, 11 (try/except per file) |
| 12. Module responsibilities | All tasks |
| 13. Testing (merge, output) | Tasks 4, 5 |
