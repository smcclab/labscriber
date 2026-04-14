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
            # Skip words that WhisperX emitted without timestamps (low-confidence words)
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

        if best_speaker is None or best_overlap < 0.0:
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

    # Stage 2: group consecutive same-speaker words into raw utterances,
    # splitting whenever the gap between words exceeds merge_gap.
    utterances: list[dict] = []
    for word in words:
        if (
            utterances
            and utterances[-1]["speaker"] == word["speaker"]
            and word["start"] - utterances[-1]["end"] < merge_gap
        ):
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

    # Stage 3: Defensive merge of adjacent same-speaker utterances within merge_gap
    # (e.g., utterances created by future algorithm changes that may emit adjacent segments)
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
