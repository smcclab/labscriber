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
