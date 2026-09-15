from __future__ import annotations

import os
import statistics
import tempfile
from dataclasses import dataclass, field

from .ffmpeg import FfmpegError, AudioTrack, extract_clip, ordered_tracks, probe, select_track
from .matcher import Match, WordStamp, compute_offset
from .srt_parser import SubtitleEntry
from .transcribe import WhisperSession


@dataclass
class SyncOptions:
    model: str = "base"
    device: str = "auto"
    language: str | None = None
    track_ordinal: int | None = None
    n_samples: int = 15
    padding: float = 15.0
    min_ratio: float = 0.5
    min_accepted: float = 0.3


@dataclass
class SampleResult:
    subtitle_index: int
    subtitle_start: float
    clip_start: float
    clip_end: float
    offset: float
    similarity: float
    matched_words: int
    transcribed: str
    accepted: bool


@dataclass
class SyncResult:
    offset: float | None
    mad: float
    accepted_count: int
    sample_count: int
    track: AudioTrack
    duration: float
    per_sample: list[SampleResult] = field(default_factory=list)


def _select_samples(entries: list[SubtitleEntry], n: int, duration: float) -> list[SubtitleEntry]:
    dialogue = [e for e in entries if len(e.words) >= 2 and e.duration >= 0.3]
    if not dialogue:
        return []
    if duration <= 0:
        duration = max(e.end for e in dialogue) + 1.0
    targets = [duration * (i + 0.5) / n for i in range(n)]
    picks = [min(dialogue, key=lambda e: abs(e.start - t)) for t in targets]
    seen: set[int] = set()
    uniq: list[SubtitleEntry] = []
    for e in picks:
        if e.index not in seen:
            seen.add(e.index)
            uniq.append(e)
    return uniq


def _try_track(
    video_path: str,
    entries: list[SubtitleEntry],
    track: AudioTrack,
    options: SyncOptions,
    duration: float,
    tmpdir: str,
    session: WhisperSession,
) -> tuple[list[SampleResult], int]:
    samples = _select_samples(entries, options.n_samples, duration)
    results: list[SampleResult] = []
    accepted = 0

    for i, sub in enumerate(samples):
        pad = options.padding
        clip_start = max(0.0, sub.start - pad)
        clip_end = min(duration if duration > 0 else sub.end + pad, sub.end + pad)
        if clip_end <= clip_start:
            clip_end = clip_start + 0.5

        wav = os.path.join(tmpdir, f"clip_{i:03d}.wav")
        try:
            extract_clip(video_path, track, clip_start, clip_end, wav)
        except FfmpegError:
            continue

        word_stamps, _lang = session.transcribe_clip(wav)
        if not word_stamps:
            continue

        match: Match | None = compute_offset(
            sub.words, word_stamps, clip_start, sub.start, options.min_ratio
        )
        if match is None:
            results.append(
                SampleResult(
                    subtitle_index=sub.index,
                    subtitle_start=sub.start,
                    clip_start=clip_start,
                    clip_end=clip_end,
                    offset=0.0,
                    similarity=0.0,
                    matched_words=0,
                    transcribed=" ".join(s.text for s in word_stamps),
                    accepted=False,
                )
            )
            continue

        results.append(
            SampleResult(
                subtitle_index=sub.index,
                subtitle_start=sub.start,
                clip_start=clip_start,
                clip_end=clip_end,
                offset=match.offset,
                similarity=match.similarity,
                matched_words=match.matched_words,
                transcribed=match.transcribed_preview,
                accepted=True,
            )
        )
        accepted += 1

    return results, accepted


def run_sync(
    video_path: str,
    entries: list[SubtitleEntry],
    options: SyncOptions,
    session: WhisperSession | None = None,
) -> SyncResult:
    tracks, duration = probe(video_path)
    ordered = ordered_tracks(tracks)
    primary = select_track(tracks, options.track_ordinal)
    candidates = [primary]
    if options.track_ordinal is None:
        candidates = [t for t in ordered]

    session = session or WhisperSession(options.model, options.device, options.language)

    fallback_used = False
    last_results: list[SampleResult] = []
    last_accepted = 0
    used_track = primary

    with tempfile.TemporaryDirectory(prefix="subtitlesync_") as tmpdir:
        for attempt, track in enumerate(candidates):
            results, accepted = _try_track(video_path, entries, track, options, duration, tmpdir, session)
            fraction = accepted / max(1, len(results))
            if attempt > 0:
                fallback_used = True
            used_track = track
            last_results, last_accepted = results, accepted
            if fraction >= options.min_accepted:
                break

    accepted_results = [r for r in last_results if r.accepted]
    if not accepted_results:
        return SyncResult(
            offset=None,
            mad=float("inf"),
            accepted_count=0,
            sample_count=len(last_results),
            track=used_track,
            duration=duration,
            per_sample=last_results,
        )

    offsets = [r.offset for r in accepted_results]
    offset = statistics.median(offsets)
    mad = statistics.median(sorted(abs(o - offset) for o in offsets))
    return SyncResult(
        offset=offset,
        mad=mad,
        accepted_count=len(accepted_results),
        sample_count=len(last_results),
        track=used_track,
        duration=duration,
        per_sample=last_results,
    )