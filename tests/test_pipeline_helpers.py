# tests/test_pipeline_helpers.py
import wave
from pathlib import Path

import pytest

from labscriber.pipeline import _fmt_duration, _get_audio_duration


def _write_wav(path: Path, duration_seconds: float, framerate: int = 16000) -> None:
    """Write a silent WAV file of the given duration."""
    n_frames = int(duration_seconds * framerate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(framerate)
        wf.writeframes(b"\x00\x00" * n_frames)


def test_get_audio_duration_10s(tmp_path):
    wav = tmp_path / "test.wav"
    _write_wav(wav, 10.0)
    assert abs(_get_audio_duration(wav) - 10.0) < 0.01


def test_get_audio_duration_90min(tmp_path):
    wav = tmp_path / "long.wav"
    _write_wav(wav, 5400.0)
    assert abs(_get_audio_duration(wav) - 5400.0) < 1.0


def test_fmt_duration_under_one_minute():
    assert _fmt_duration(45) == "0:45"


def test_fmt_duration_minutes_and_seconds():
    assert _fmt_duration(75) == "1:15"


def test_fmt_duration_zero_seconds():
    assert _fmt_duration(0) == "0:00"


def test_fmt_duration_hours():
    assert _fmt_duration(3661) == "1:01:01"


def test_fmt_duration_exactly_one_hour():
    assert _fmt_duration(3600) == "1:00:00"
