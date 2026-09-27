from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher


@dataclass
class WordStamp:
    text: str
    start: float
    end: float


@dataclass
class Match:
    matched_words: int
    similarity: float
    actual_time: float
    offset: float
    transcribed_preview: str


def _norm_word(w: str) -> str:
    return "".join(ch for ch in w.lower() if ch.isalnum())


def compute_offset(
    sub_words: list[str],
    stamps: list[WordStamp],
    clip_start: float,
    sub_start: float,
    min_ratio: float = 0.5,
) -> Match | None:
    sub_norm = [_norm_word(w) for w in sub_words if _norm_word(w)]
    kept = [(i, s) for i, s in enumerate(stamps) if _norm_word(s.text)]
    trans_norm = [_norm_word(s.text) for _, s in kept]
    if len(sub_norm) < 2 or len(trans_norm) < 2:
        return None

    sm = SequenceMatcher(None, sub_norm, trans_norm, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size > 0]
    if not blocks:
        return None

    # Filter matching blocks so they form a coherent, compact phrase
    # rather than words scattered across tens of seconds of dialogue.
    max_span = int(len(sub_norm) * 1.5 + 4)
    best_blocks = []
    best_count = 0
    for i in range(len(blocks)):
        for j in range(i, len(blocks)):
            sub_b = blocks[i : j + 1]
            span = (sub_b[-1].b + sub_b[-1].size) - sub_b[0].b
            if span <= max_span:
                matched = sum(b.size for b in sub_b)
                if matched > best_count:
                    best_count = matched
                    best_blocks = sub_b

    if not best_blocks:
        return None

    min_size = min(2, len(sub_norm))
    if best_count < min_size:
        return None

    similarity = best_count / len(sub_norm)
    if similarity < min_ratio:
        return None

    trans_idx = kept[best_blocks[0].b][0]
    actual_time = clip_start + stamps[trans_idx].start
    preview = " ".join(s.text.lower() for s in stamps[trans_idx : trans_idx + 10])
    return Match(
        matched_words=best_count,
        similarity=round(similarity, 3),
        actual_time=round(actual_time, 3),
        offset=round(actual_time - sub_start, 3),
        transcribed_preview=preview,
    )