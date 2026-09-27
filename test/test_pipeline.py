from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from modules.pipeline import (
    SubtitleSyncError,
    SyncOptions,
    SyncTaskResult,
    format_offset,
    load_entries,
    nvidia_present,
    resolve_device,
    run_task,
)
from modules.syncer import SampleResult


class DummyTrack:
    ordinal = 0
    language = "eng"


def _sample_result(
    index=1, start=100.0, offset=10.5, similarity=0.9, accepted=True, transcribed="hello world"
):
    return SampleResult(
        subtitle_index=index,
        subtitle_start=start,
        clip_start=start - 15,
        clip_end=start + 15,
        offset=offset,
        similarity=similarity,
        matched_words=5,
        transcribed=transcribed,
        accepted=accepted,
    )


class DummyResult:
    offset = 10.5
    mad = 0.12
    accepted_count = 10
    sample_count = 15
    duration = 3200.0
    track = DummyTrack()
    device = "cpu"
    elapsed = 45.2
    per_sample = [
        _sample_result(index=1, start=100.0, offset=10.5, similarity=0.9, accepted=True),
        _sample_result(index=2, start=200.0, offset=10.4, similarity=0.8, accepted=True),
    ]


def test_format_offset():
    assert format_offset(10.5) == "+10.50s (delay subtitles by 10.50s)"
    assert format_offset(-5.25) == "-5.25s (advance subtitles by 5.25s)"
    assert format_offset(0.0) == "+0.00s (delay subtitles by 0.00s)"


def test_load_entries_missing(tmp_path):
    missing = tmp_path / "no.srt"
    with pytest.raises(SubtitleSyncError) as exc:
        load_entries(str(missing))
    assert "not found" in str(exc.value).lower()


def test_load_entries_empty(tmp_path):
    empty = tmp_path / "empty.srt"
    empty.write_text("\n\n", encoding="utf-8")
    with pytest.raises(SubtitleSyncError) as exc:
        load_entries(str(empty))
    assert "no subtitle lines" in str(exc.value).lower()


def test_load_entries_ok(tmp_path):
    srt = tmp_path / "ok.srt"
    srt.write_text(
        "1\n00:00:01,000 --> 00:00:04,000\nHello world\n\n",
        encoding="utf-8",
    )
    entries = load_entries(str(srt))
    assert len(entries) == 1
    assert entries[0].text == "Hello world"


def _make_dummy_video(tmp_path):
    """Create a tiny fake video file for tests that need a real path."""
    v = tmp_path / "dummy.mkv"
    v.write_bytes(b"fake")
    return str(v)


def _make_dummy_srt(tmp_path):
    s = tmp_path / "dummy.srt"
    s.write_text(
        "1\n00:00:01,000 --> 00:00:04,000\nHello world\n\n",
        encoding="utf-8",
    )
    return str(s)


@patch("modules.pipeline.WhisperSession")
@patch("modules.pipeline.run_sync")
@patch("modules.pipeline.probe")
def test_run_task_happy(mock_probe, mock_run_sync, mock_session_class, tmp_path):
    # arrange
    mock_probe.return_value = ([MagicMock(ordinal=1, language="eng")], 3200.0)
    mock_session = MagicMock()
    mock_session.device = "cuda"
    mock_session_class.return_value = mock_session
    mock_run_sync.return_value = DummyResult()

    opts = SyncOptions(model="base", device="auto", track_ordinal=None)
    video = _make_dummy_video(tmp_path)
    srt = _make_dummy_srt(tmp_path)

    # act
    result = run_task(video, srt, opts, progress=None)

    # assert
    assert isinstance(result, SyncTaskResult)
    assert result.offset == 10.5
    assert result.accepted_count == 10
    assert result.sample_count == 15
    assert result.low_confidence is False
    assert result.output_path is not None
    mock_session_class.assert_called_once_with("base", "auto", None)
    mock_run_sync.assert_called_once()


@patch("modules.pipeline.WhisperSession")
@patch("modules.pipeline.run_sync")
@patch("modules.pipeline.probe")
def test_run_task_low_confidence(mock_probe, mock_run_sync, mock_session_class, tmp_path):
    class LowResult(DummyResult):
        accepted_count = 2
        sample_count = 15
        per_sample = [
            _sample_result(index=1, accepted=False),
            _sample_result(index=2, accepted=False),
            _sample_result(index=3, accepted=True),
            _sample_result(index=4, accepted=True),
        ]

    mock_probe.return_value = ([MagicMock(ordinal=1, language="eng")], 3200.0)
    mock_session = MagicMock()
    mock_session.device = "cpu"
    mock_session_class.return_value = mock_session
    mock_run_sync.return_value = LowResult()

    opts = SyncOptions(model="base", device="auto", track_ordinal=None, min_accepted=0.3)
    video = _make_dummy_video(tmp_path)
    srt = _make_dummy_srt(tmp_path)

    result = run_task(video, srt, opts, progress=None)

    assert result.low_confidence is True


@patch("modules.pipeline.WhisperSession")
@patch("modules.pipeline.run_sync")
@patch("modules.pipeline.probe")
def test_run_task_no_match(mock_probe, mock_run_sync, mock_session_class, tmp_path):
    class NoMatchResult:
        offset = None
        mad = float("inf")
        accepted_count = 0
        sample_count = 15
        duration = 3200.0
        track = DummyTrack()
        device = "cpu"
        elapsed = 45.2
        per_sample = []

    mock_probe.return_value = ([MagicMock(ordinal=1, language="eng")], 3200.0)
    mock_session = MagicMock()
    mock_session_class.return_value = mock_session
    mock_run_sync.return_value = NoMatchResult()

    opts = SyncOptions(model="base", device="auto", track_ordinal=None)
    video = _make_dummy_video(tmp_path)
    srt = _make_dummy_srt(tmp_path)

    with pytest.raises(SubtitleSyncError) as exc:
        run_task(video, srt, opts, progress=None)
    assert "no subtitle lines matched" in str(exc.value).lower()


@patch("modules.pipeline.WhisperSession")
@patch("modules.pipeline.probe")
def test_run_task_transcribe_error(mock_probe, mock_session_class, tmp_path):
    from modules.transcribe import TranscribeError

    mock_probe.return_value = ([MagicMock(ordinal=1, language="eng")], 3200.0)
    mock_session_class.side_effect = TranscribeError("model load failed")

    opts = SyncOptions(model="base", device="auto", track_ordinal=None)
    video = _make_dummy_video(tmp_path)
    srt = _make_dummy_srt(tmp_path)
    with pytest.raises(SubtitleSyncError) as exc:
        run_task(video, srt, opts, progress=None)
    assert "could not load" in str(exc.value).lower()


@patch("modules.pipeline.WhisperSession")
@patch("modules.pipeline.probe")
def test_run_task_ffmpeg_error(mock_probe, mock_session_class, tmp_path):
    from modules.ffmpeg import FfmpegError

    mock_probe.side_effect = FfmpegError("no video stream")
    mock_session_class.return_value = MagicMock()

    opts = SyncOptions(model="base", device="auto", track_ordinal=None)
    video = _make_dummy_video(tmp_path)
    srt = _make_dummy_srt(tmp_path)
    with pytest.raises(SubtitleSyncError) as exc:
        run_task(video, srt, opts, progress=None)
    assert "could not read the video file" in str(exc.value).lower()


def test_resolve_device_cpu(monkeypatch):
    # Simulate ctranslate2.get_cuda_device_count returning 0 or raising
    def fake_count():
        return 0

    monkeypatch.setattr("ctranslate2.get_cuda_device_count", fake_count)
    dev = resolve_device()
    assert dev == "cpu"


def test_resolve_device_cuda(monkeypatch):
    def fake_count():
        return 1

    monkeypatch.setattr("ctranslate2.get_cuda_device_count", fake_count)
    # _register_nvidia_dlls should be called but won't affect this test
    dev = resolve_device()
    assert dev == "cuda"


def test_nvidia_present_true(monkeypatch):
    import shutil

    monkeypatch.setattr(shutil, "which", lambda _: "/fake/nvidia-smi")
    assert nvidia_present() is True


def test_nvidia_present_false(monkeypatch):
    import shutil

    monkeypatch.setattr(shutil, "which", lambda _: None)
    assert nvidia_present() is False


def test_progress_callback_fires(tmp_path):
    """run_task should call the progress callback with rising fractions."""
    captured = []

    class DummyTrack:
        ordinal = 0
        language = "eng"

    class DummyResult:
        offset = 5.0
        mad = 0.1
        accepted_count = 5
        sample_count = 5
        duration = 1000.0
        track = DummyTrack()
        device = "cpu"
        elapsed = 10.0
        per_sample = [_sample_result(index=1, start=100.0, offset=5.0, similarity=0.9, accepted=True)]

    with patch("modules.pipeline.probe", return_value=([DummyTrack()], 1000.0)), \
         patch("modules.pipeline.WhisperSession") as mock_ws, \
         patch("modules.pipeline.run_sync", return_value=DummyResult()):

        mock_ws.return_value = MagicMock(device="cpu")

        def on_progress(msg, frac):
            captured.append((msg, round(frac, 2)))

        video = _make_dummy_video(tmp_path)
        srt = _make_dummy_srt(tmp_path)
        run_task(video, srt, SyncOptions(), progress=on_progress)

    assert len(captured) >= 3  # at least start, some progress, done
    fracs = [f for _, f in captured]
    assert fracs[-1] == 1.0  # final "done"
    # fractions should be non-decreasing
    assert all(fracs[i] <= fracs[i + 1] + 1e-9 for i in range(len(fracs) - 1))