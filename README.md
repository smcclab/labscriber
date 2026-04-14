# labscriber

Local batch transcription for multi-speaker HCI research recordings.

labscriber takes a folder of audio files, transcribes them with [WhisperX](https://github.com/m-bain/whisperX), assigns speaker labels via pyannote diarization, and writes clean Markdown and plain-text transcripts. Everything runs on-device — no cloud APIs, no data leaves your machine.

## Requirements

- Python 3.11–3.12
- [Poetry](https://python-poetry.org)
- [ffmpeg](https://ffmpeg.org) installed on your system (`brew install ffmpeg` on macOS)
- A free [HuggingFace](https://huggingface.co) account (required for speaker diarization)
- CUDA GPU recommended; CPU fallback supported (slow on large files)

## Installation

```bash
git clone https://github.com/smcclab/labscriber
cd labscriber
poetry install
```

## Setup

Speaker diarization uses two gated pyannote models. You need to accept their terms once and provide a HuggingFace token.

**Step 1:** Accept model terms (must be logged in to HuggingFace):

- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

**Step 2:** Create a read-only access token at https://huggingface.co/settings/tokens

**Step 3:** Run the setup wizard:

```bash
poetry run labscriber setup
```

This saves your token to a `.env` file in your working directory.

**Step 4:** Pre-download the ML models (optional but recommended before first use):

```bash
poetry run labscriber download-models
```

## Usage

### Directory layout

Create an `ingest/` folder next to your working directory and drop your audio files in:

```
<working-dir>/
    ingest/        ← put audio files here (any format)
    work/          ← intermediate files (auto-created, safe to delete)
    output/        ← transcripts written here
    .env           ← HF_TOKEN (created by labscriber setup)
```

### Transcribe

```bash
poetry run labscriber process
```

Processes all audio files in `ingest/`, writes `.md` and `.txt` transcripts to `output/`. Skips files that already have output — re-run safely at any time.

Common options:

```bash
# Use a smaller/faster model
poetry run labscriber process --model medium

# Force a specific language (skips auto-detection, faster)
poetry run labscriber process --language en

# Skip speaker diarization (no HuggingFace token needed)
poetry run labscriber process --no-diarize

# Re-process everything, ignoring existing outputs
poetry run labscriber process --force

# Hint the number of speakers (improves diarization accuracy)
poetry run labscriber process --max-speakers 2
```

### Check processing state

```bash
poetry run labscriber status
```

Shows per-file progress across all pipeline stages:

```
ingest/interview-01.mp3  ✓ wav  ✓ asr  ✓ diarize  ✓ output
ingest/interview-02.mp3  ✓ wav  ✓ asr  ✗ diarize  ✗ output
```

### Clean intermediate files

```bash
poetry run labscriber clean
```

Deletes the `work/` directory. Prompts for confirmation. Use `--yes` to skip the prompt in scripts.

## Output formats

Each audio file produces two transcript files in `output/`:

**`<stem>.md`** — Markdown with speaker labels and timestamps:

```markdown
# interview-01

**Processed:** 2026-04-14
**Language:** en
**Speakers detected:** 2
**Duration:** 0:42:17

---

**SPEAKER_00** `[0:00]`
Hello, welcome to the study. Can you start by telling me a bit about yourself?

**SPEAKER_01** `[0:08]`
Sure. I'm a second-year PhD student working on...
```

**`<stem>.txt`** — Plain text for import into NVivo, Atlas.ti, or a text editor:

```
interview-01 — Transcript
Processed: 2026-04-14 | Language: en | Speakers: 2 | Duration: 0:42:17

SPEAKER_00 [0:00]
Hello, welcome to the study...

SPEAKER_01 [0:08]
Sure. I'm a second-year PhD student...
```

## Models

| Model | Speed | Accuracy | Recommended for |
|---|---|---|---|
| `tiny` | fastest | lowest | Quick checks, testing |
| `base` | fast | low | Short clips |
| `small` | moderate | good | Drafts |
| `medium` | moderate | very good | General use |
| `large-v2` | slow | excellent | Final transcripts (default) |
| `large-v3` | slow | excellent | Multilingual recordings |

## All options

```
labscriber process [OPTIONS]

  --ingest-path PATH     Input audio directory           [default: ingest]
  --output-path PATH     Output transcript directory     [default: output]
  --work-path PATH       Intermediate files directory    [default: work]
  --model CHOICE         Whisper model size              [default: large-v2]
  --language TEXT        Force language code, e.g. 'en'  [default: auto-detect]
  --min-speakers INT     Minimum speaker count hint      [default: 1]
  --max-speakers INT     Maximum speaker count hint
  --merge-gap FLOAT      Silence (sec) before new segment [default: 1.5]
  --no-diarize           Skip speaker diarization
  --force                Re-process all files
  --verbose              Print detailed debug output
```

## Development

```bash
poetry run pytest
```

## License

MIT
