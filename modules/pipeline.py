from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from .corrected_srt import default_output_path, shift_entries, write_srt
from .ffmpeg import FfmpegError, probe, select_track
from .srt_parser import SubtitleEntry, parse_srt_file
from .syncer import ProgressFn, SyncOptions, SyncResult, run_sync
from .transcribe import TranscribeError, WhisperSession


class SubtitleSyncError(RuntimeError):
    """A user-facing failure. Carries a message safe to show a non-technical user."""


@dataclass
class SyncTaskResult:
    offset: float | None
    mad: float
    accepted_count: int
    sample_count: int
    duration: float
    track_ordinal: int
    track_language: str | None
    device: str
    elapsed: float
    low_confidence: bool
    output_path: str | None
    video_path: str
    srt_path: str
    per_sample: list[dict] = field(default_factory=list)


def format_offset(seconds: float) -> str:
    """Human-readable offset, e.g. "+10.82s (delay subtitles by 10.82s)"."""
    sign = "+" if seconds >= 0 else ""
    if seconds >= 0:
        return f"{sign}{seconds:.2f}s (delay subtitles by {seconds:.2f}s)"
    return f"{seconds:.2f}s (advance subtitles by {abs(seconds):.2f}s)"


def resolve_device() -> str:
    """Return "cuda" when a usable NVIDIA GPU is available, else "cpu"."""
    try:
        from .transcribe import probe_device

        return probe_device()
    except Exception:
        return "cpu"


def nvidia_present() -> bool:
    """True when the machine has an NVIDIA card, regardless of libraries."""
    import shutil

    return shutil.which("nvidia-smi") is not None


def load_entries(srt_path: str) -> list[SubtitleEntry]:
    if not Path(srt_path).is_file():
        raise SubtitleSyncError(f"Subtitle file not found: {srt_path}")
    try:
        entries = parse_srt_file(srt_path)
    except OSError as exc:
        raise SubtitleSyncError(f"Could not read the subtitle file: {exc}") from exc
    if not entries:
        raise SubtitleSyncError(
            "No subtitle lines were found in that file. Please choose a .srt file."
        )
    return entries


def run_task(
    video_path: str,
    srt_path: str,
    options: SyncOptions,
    session: WhisperSession | None = None,
    progress: ProgressFn | None = None,
    auto_save: bool = True,
    output_path: str | None = None,
) -> SyncTaskResult:
    """Validate inputs, detect the offset, and optionally write a corrected .srt.

    Raises SubtitleSyncError for anything the user can act on. Progress is
    reported as (message, fraction) with fraction rising from 0 to 1.
    """
    def emit(message: str, fraction: float) -> None:
        if progress is not None:
            progress(message, fraction)

    if not Path(video_path).is_file():
        raise SubtitleSyncError(f"Video file not found: {video_path}")

    emit("Reading subtitle file...", 0.01)
    entries = load_entries(srt_path)

    emit("Loading the speech model...", 0.03)
    try:
        session = session or WhisperSession(options.model, options.device, options.language)
    except TranscribeError as exc:
        raise SubtitleSyncError(f"Could not load the speech model: {exc}") from exc

    emit("Reading audio tracks...", 0.06)
    try:
        tracks, _duration = probe(video_path)
    except FfmpegError as exc:
        raise SubtitleSyncError(f"Could not read the video file: {exc}") from exc
    if not tracks:
        raise SubtitleSyncError("No audio track was found in that video file.")

    best = select_track(tracks, options.track_ordinal)
    emit(f"Using audio track {best.ordinal}", 0.08)

    started = time.time()
    try:
        result: SyncResult = run_sync(video_path, entries, options, session, progress=progress)
    except TranscribeError as exc:
        raise SubtitleSyncError(f"Speech recognition failed: {exc}") from exc
    except FfmpegError as exc:
        raise SubtitleSyncError(f"Could not read the video file: {exc}") from exc
    elapsed = time.time() - started

    if result.offset is None:
        raise SubtitleSyncError(
            "No subtitle lines matched the speech in the video. This usually means the "
            "subtitle file is for a different release, or a different audio track is needed."
        )

    fraction = result.accepted_count / max(1, result.sample_count)

    written: str | None = None
    if auto_save or output_path:
        target = output_path or default_output_path(srt_path)
        emit("Saving the corrected subtitle...", 0.97)
        try:
            write_srt(shift_entries(entries, result.offset), target)
        except OSError as exc:
            raise SubtitleSyncError(f"Could not write the corrected subtitle: {exc}") from exc
        written = target

    emit("Done", 1.0)

    return SyncTaskResult(
        offset=result.offset,
        mad=result.mad,
        accepted_count=result.accepted_count,
        sample_count=result.sample_count,
        duration=result.duration,
        track_ordinal=result.track.ordinal,
        track_language=result.track.language,
        device=session.device,
        elapsed=elapsed,
        low_confidence=fraction < options.min_accepted,
        output_path=written,
        video_path=video_path,
        srt_path=srt_path,
        per_sample=[
            {
                "index": s.subtitle_index,
                "start": round(s.subtitle_start, 2),
                "offset": round(s.offset, 3),
                "similarity": round(s.similarity, 2),
                "accepted": s.accepted,
                "transcribed": s.transcribed,
            }
            for s in result.per_sample
        ],
    )
