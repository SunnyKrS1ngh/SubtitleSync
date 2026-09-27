from __future__ import annotations

import statistics
import pytest
from unittest.mock import MagicMock, patch

from modules.srt_parser import SubtitleEntry
from modules.syncer import (
    SyncOptions,
    _find_dominant_cluster,
    _select_samples,
    _coarse_search,
)
from modules.ffmpeg import AudioTrack


def _make_entry(idx: int, start: float, duration: float, text: str) -> SubtitleEntry:
    words = text.split()
    return SubtitleEntry(index=idx, start=start, end=start + duration, text=text, words=words)


def test_find_dominant_cluster_empty():
    assert _find_dominant_cluster([]) == []


def test_find_dominant_cluster_single_group():
    votes = [10.1, 10.2, 9.9, 10.5, 45.0]
    cluster = _find_dominant_cluster(votes, tolerance=2.0)
    assert len(cluster) == 4
    assert 45.0 not in cluster


def test_find_dominant_cluster_tolerance():
    votes = [1.0, 2.0, 3.0, 100.0, 101.0]
    cluster = _find_dominant_cluster(votes, tolerance=5.0)
    # The first cluster [1, 2, 3] has 3 items
    assert cluster == [1.0, 2.0, 3.0]


def test_select_samples_filters_brackets_and_short():
    entries = [
        _make_entry(1, 10.0, 2.0, "[MUSIC PLAYING]"),
        _make_entry(2, 20.0, 2.0, "Yeah."),
        _make_entry(3, 30.0, 3.0, "This is a rich dialogue sentence with enough words to match."),
        _make_entry(4, 50.0, 3.0, "Another great complete sentence with good vocabulary in it."),
    ]
    samples = _select_samples(entries, n=2, duration=60.0)
    assert len(samples) == 2
    # Ensure neither the bracketed noise nor the 1-word entry was chosen
    sample_indices = [s.index for s in samples]
    assert 1 not in sample_indices
    assert 3 in sample_indices or 4 in sample_indices


def test_select_samples_spread_across_duration():
    entries = [
        _make_entry(i, float(i * 100), 2.5, f"Sentence number {i} with several distinctive words inside.")
        for i in range(1, 21)
    ]
    samples = _select_samples(entries, n=5, duration=2000.0)
    assert len(samples) == 5
    # Check that samples are in chronological order
    starts = [s.start for s in samples]
    assert starts == sorted(starts)
    # Check that they span early and late times
    assert starts[0] < 500.0
    assert starts[-1] > 1500.0
