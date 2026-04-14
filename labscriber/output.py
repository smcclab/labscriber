# labscriber/output.py
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
