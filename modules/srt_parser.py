from __future__ import annotations

import re
from dataclasses import dataclass

TAG_RE = re.compile(r"<[^>]*>")
BRACKET_RE = re.compile(r"[\u0000-\u001f\u007f]+|\[[^\]]*\]")


@dataclass
class SubtitleEntry:
    index: int
    start: float
    end: float
    text: str
    words: list[str]

    @property
    def duration(self) -> float:
        return self.end - self.start


def _clean_text(raw: str) -> str:
    text = TAG_RE.sub("", raw)
    text = BRACKET_RE.sub(" ", text)
    return " ".join(text.split())


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _parse_timestamp(ts: str) -> float:
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    if len(parts) != 3:
        raise ValueError(f"invalid timestamp: {ts!r}")
    h, m, s = (float(p) for p in parts)
    return h * 3600 + m * 60 + s


def parse_srt_file(path: str) -> list[SubtitleEntry]:
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        content = f.read().replace("\r\n", "\n").replace("\r", "\n")
    return parse_srt(content)


def parse_srt(content: str) -> list[SubtitleEntry]:
    if content.startswith("\ufeff"):
        content = content.lstrip("\ufeff")
    blocks = re.split(r"\n\s*\n", content.strip())
    entries: list[SubtitleEntry] = []
    for block in blocks:
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        try:
            index = int(lines[0])
        except ValueError:
            continue
        if len(lines) < 2:
            continue
        m = re.match(r"(\d{1,2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,.]\d{3})", lines[1])
        if not m:
            continue
        start = _parse_timestamp(m.group(1))
        end = _parse_timestamp(m.group(2))
        text = " ".join(lines[2:])
        clean = _clean_text(text)
        if not clean:
            continue
        entries.append(SubtitleEntry(index=index, start=start, end=end, text=clean, words=_tokenize(clean)))
    return entries