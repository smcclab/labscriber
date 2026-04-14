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
