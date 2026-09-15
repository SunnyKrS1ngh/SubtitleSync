from __future__ import annotations

from .srt_parser import SubtitleEntry


def _fmt(seconds: float) -> str:
    seconds = max(0.0, seconds)
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def shift_entries(entries: list[SubtitleEntry], offset: float) -> list[SubtitleEntry]:
    shifted: list[SubtitleEntry] = []
    for e in entries:
        new_start = max(0.0, e.start + offset)
        new_end = max(new_start + 0.001, e.end + offset)
        shifted.append(
            SubtitleEntry(index=e.index, start=new_start, end=new_end, text=e.text, words=e.words)
        )
    return shifted


def write_srt(entries: list[SubtitleEntry], path: str) -> None:
    lines: list[str] = []
    for i, e in enumerate(entries, 1):
        lines.append(str(i))
        lines.append(f"{_fmt(e.start)} --> {_fmt(e.end)}")
        lines.append(e.text)
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def default_output_path(srt_path: str) -> str:
    if srt_path.lower().endswith(".srt"):
        return srt_path[:-4] + ".corrected.srt"
    return srt_path + ".corrected.srt"