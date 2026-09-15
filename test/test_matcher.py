from __future__ import annotations

import pytest

from modules.matcher import WordStamp, compute_offset


def _stamps(words_and_times):
    return [WordStamp(text=w, start=t, end=t + 0.5) for w, t in words_and_times]


def test_exact_match_with_leading_filler():
    sub = ["can", "i", "get", "some", "coffee"]
    stamps = _stamps([
        ("well", 0.2), ("can", 1.5), ("i", 1.6), ("get", 1.8),
        ("some", 1.9), ("coffee", 2.1), ("yeah", 2.5),
    ])
    m = compute_offset(sub, stamps, clip_start=100.0, sub_start=108.0)
    assert m is not None
    assert m.offset == pytest.approx(-6.5)
    assert m.similarity == pytest.approx(1.0)


def test_misheard_word_still_matches():
    sub = ["can", "i", "get", "some", "coffee"]
    stamps = _stamps([("can", 1.5), ("i", 1.6), ("git", 1.8), ("some", 1.9), ("coffee", 2.1)])
    m = compute_offset(sub, stamps, clip_start=100.0, sub_start=101.5)
    assert m is not None
    assert m.offset == pytest.approx(0.0)
    assert m.similarity == pytest.approx(0.8)


def test_no_match_returns_none():
    sub = ["the", "dog", "ran", "home"]
    stamps = _stamps([("music", 0.1), ("and", 0.5), ("wind", 0.9)])
    assert compute_offset(sub, stamps, clip_start=0.0, sub_start=5.0) is None


def test_offset_positive_when_subtitle_early():
    sub = ["hello", "there"]
    stamps = _stamps([("hello", 3.0), ("there", 3.4)])
    m = compute_offset(sub, stamps, clip_start=100.0, sub_start=100.0)
    assert m is not None
    assert m.offset == pytest.approx(3.0)


def test_short_line_minimum():
    sub = ["fuck"]
    stamps = _stamps([("fuck", 0.1)])
    assert compute_offset(sub, stamps, clip_start=0.0, sub_start=1.0) is None


def test_punctuation_ignored():
    sub = ["its", "in", "the", "genes"]
    stamps = _stamps([("-", 0.0), ("it's", 1.0), ("in", 1.3), ("the", 1.6), ("genes", 1.9)])
    m = compute_offset(sub, stamps, clip_start=50.0, sub_start=51.0)
    assert m is not None
    assert m.offset == pytest.approx(0.0)