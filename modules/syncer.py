from __future__ import annotations

import os
import statistics
import tempfile
from dataclasses import dataclass, field
from typing import Callable

from .ffmpeg import FfmpegError, AudioTrack, extract_clip, ordered_tracks, probe, select_track
from .matcher import Match, WordStamp, compute_offset
from .srt_parser import SubtitleEntry
from .transcribe import WhisperSession

# Called as progress(message, fraction) where fraction is 0..1 and never
# decreases. Fraction is capped below 1.0 so the caller can show a final step.
ProgressFn = Callable[[str, float], None]


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
    """Select n well-spaced dialogue entries across the video duration.

    Prefers rich, distinctive sentences (6-14 words) over short exclamations,
    ignoring sound effect brackets, to maximize Whisper recognition accuracy.
    """
    clean: list[SubtitleEntry] = []
    for e in entries:
        txt = e.text.strip()
        if (txt.startswith("[") and txt.endswith("]")) or (txt.startswith("(") and txt.endswith(")")):
            continue
        if len(e.words) >= 3 and e.duration >= 0.5:
            clean.append(e)
    if not clean:
        clean = [e for e in entries if len(e.words) >= 2]
    if not clean:
        return []
    if duration <= 0:
        duration = max(e.end for e in clean) + 1.0

    targets = [duration * (i + 0.5) / n for i in range(n)]
    window_radius = duration / (n * 1.5)

    def score_entry(e: SubtitleEntry, target: float) -> float:
        w_len = len(e.words)
        if w_len < 3:
            w_score = -50.0
        elif w_len < 5:
            w_score = float(w_len)
        else:
            w_score = 10.0 + min(w_len, 14)
        dist_sec = abs(e.start - target)
        dist_penalty = dist_sec / 30.0
        return w_score - dist_penalty

    picks: list[SubtitleEntry] = []
    seen: set[int] = set()
    for t in targets:
        candidates = [e for e in clean if abs(e.start - t) <= window_radius and e.index not in seen]
        if not candidates:
            candidates = [e for e in clean if e.index not in seen]
        if not candidates:
            break
        best = max(candidates, key=lambda e: score_entry(e, t))
        picks.append(best)
        seen.add(best.index)
    return picks


def _find_dominant_cluster(votes: list[float], tolerance: float = 6.0) -> list[float]:
    """Return all votes that belong to the largest cluster within +-tolerance seconds."""
    if not votes:
        return []
    best_cluster: list[float] = []
    for candidate in votes:
        cluster = [v for v in votes if abs(v - candidate) <= tolerance]
        if len(cluster) > len(best_cluster):
            best_cluster = cluster
    return best_cluster


def _try_track(
    video_path: str,
    samples: list[SubtitleEntry],
    track: AudioTrack,
    options: SyncOptions,
    duration: float,
    tmpdir: str,
    session: WhisperSession,
    progress: ProgressFn | None = None,
    base: int = 0,
    span: int = 0,
    coarse_offset: float = 0.0,
) -> tuple[list[SampleResult], int]:
    results: list[SampleResult] = []
    accepted = 0

    for i, sub in enumerate(samples):
        if progress is not None:
            if base:
                progress(
                    f"track {track.ordinal}: sample {i + 1} of {span}",
                    (base + i + 1) / max(1, base + span),
                )
            else:
                progress(f"sample {i + 1} of {span}", (i + 1) / max(1, span))

        pad = options.padding
        # Apply coarse offset bias: shift the clip window to where the audio
        # is likely to be.
        biased_start = sub.start + coarse_offset
        clip_start = max(0.0, biased_start - pad)
        clip_end = min(duration if duration > 0 else biased_start + pad, biased_start + pad)
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


# ---------------------------------------------------------------------------
# Coarse offset search for severe out-of-sync
# ---------------------------------------------------------------------------

_COARSE_SCAN_STEP = 60
_COARSE_CLIP_LEN = 75
_COARSE_VERIFY_PAD = 15.0
_COARSE_VERIFY_TOL = 6.0


def _coarse_search(
    video_path: str,
    entries: list[SubtitleEntry],
    track: AudioTrack,
    duration: float,
    tmpdir: str,
    session: WhisperSession,
    progress: ProgressFn | None = None,
) -> float | None:
    """Find the approximate global offset when subtitles are severely out of sync.

    Picks 4-6 high-quality probe subtitles from early dialogue (first ~20 min).
    Scans the video timeline in 60s steps. When a probe hits, it immediately
    verifies the candidate offset against a second probe subtitle. Once verified,
    it exits early, resolving coarse offset in seconds.
    """
    limit_time = min(1200.0, max(300.0, duration * 0.4))
    candidates = [
        e for e in entries
        if 20.0 <= e.start <= limit_time
        and len(e.words) >= 7
        and e.duration >= 1.0
        and not (e.text.strip().startswith("[") or e.text.strip().startswith("("))
    ]
    candidates.sort(key=lambda e: -len(e.words))
    probes = candidates[:5]
    if not probes:
        all_cands = sorted(
            [e for e in entries if len(e.words) >= 5 and e.duration >= 0.8],
            key=lambda e: -len(e.words)
        )
        probes = all_cands[:5]
    if not probes:
        return None

    scan_limit = min(int(duration), 1800)
    step = _COARSE_SCAN_STEP
    clip_len = _COARSE_CLIP_LEN

    all_votes: list[float] = []

    for scan_idx, vid_pos in enumerate(range(0, scan_limit, step)):
        if progress is not None:
            progress(
                f"Searching for dialogue ({vid_pos}s / {scan_limit}s)...",
                0.05 + 0.35 * (scan_idx / max(1, scan_limit // step)),
            )

        clip_end = min(vid_pos + clip_len, duration)
        wav = os.path.join(tmpdir, f"coarse_scan_{scan_idx}.wav")
        try:
            extract_clip(video_path, track, float(vid_pos), clip_end, wav)
        except FfmpegError:
            continue

        stamps, _ = session.transcribe_clip(wav)
        if not stamps:
            continue

        for p in probes:
            m = compute_offset(p.words, stamps, float(vid_pos), p.start, min_ratio=0.55)
            if m is not None:
                all_votes.append(m.offset)
                # Verify candidate offset with other probes
                other_probes = [o for o in probes if o.index != p.index]
                verified_offsets = [m.offset]
                for other in other_probes:
                    v_start = max(0.0, other.start + m.offset - _COARSE_VERIFY_PAD)
                    v_end = min(duration, other.start + m.offset + _COARSE_VERIFY_PAD)
                    if v_end <= v_start:
                        continue
                    v_wav = os.path.join(tmpdir, f"verify_{other.index}.wav")
                    try:
                        extract_clip(video_path, track, v_start, v_end, v_wav)
                        v_stamps, _ = session.transcribe_clip(v_wav)
                    except FfmpegError:
                        continue
                    v_m = compute_offset(other.words, v_stamps, v_start, other.start, min_ratio=0.50)
                    if v_m is not None and abs(v_m.offset - m.offset) <= _COARSE_VERIFY_TOL:
                        verified_offsets.append(v_m.offset)
                        break

                if len(verified_offsets) >= 2:
                    return statistics.median(verified_offsets)

    if all_votes:
        cluster = _find_dominant_cluster(all_votes, _COARSE_VERIFY_TOL)
        if cluster:
            return statistics.median(cluster)
        return statistics.median(all_votes)

    return None


def run_sync(
    video_path: str,
    entries: list[SubtitleEntry],
    options: SyncOptions,
    session: WhisperSession | None = None,
    progress: ProgressFn | None = None,
) -> SyncResult:
    tracks, duration = probe(video_path)
    ordered = ordered_tracks(tracks)
    primary = select_track(tracks, options.track_ordinal)
    candidates = [primary]
    if options.track_ordinal is None:
        candidates = [t for t in ordered]

    session = session or WhisperSession(options.model, options.device, options.language)

    samples = _select_samples(entries, options.n_samples, duration)

    high_water = 0.0

    def emit(message: str, fraction: float) -> None:
        nonlocal high_water
        fraction = min(0.95, max(high_water, fraction))
        high_water = fraction
        if progress is not None:
            progress(message, fraction)

    last_results: list[SampleResult] = []
    last_accepted = 0
    used_track = primary

    with tempfile.TemporaryDirectory(prefix="subtitlesync_") as tmpdir:
        for attempt, track in enumerate(candidates):
            # --- Phase 1: Try normal fine-grained pass (coarse_offset = 0) ---
            results, accepted = _try_track(
                video_path, samples, track, options, duration, tmpdir, session,
                progress=emit, base=attempt * len(samples), span=len(samples),
                coarse_offset=0.0,
            )
            fraction = accepted / max(1, len(results))
            used_track = track
            last_results, last_accepted = results, accepted

            if fraction >= options.min_accepted:
                # Normal sync succeeded (within options.padding)
                break

            # --- Phase 2: Coarse global-offset search for severe out-of-sync ---
            emit("Normal sync failed — searching for offset...", high_water)
            coarse_offset = _coarse_search(
                video_path, entries, track, duration, tmpdir, session, progress=emit
            )

            if coarse_offset is not None:
                emit(
                    f"Coarse offset found: {coarse_offset:+.1f}s — refining...",
                    high_water,
                )
                refine_padding = max(options.padding, 25.0)
                refine_options = SyncOptions(
                    model=options.model,
                    device=options.device,
                    language=options.language,
                    track_ordinal=options.track_ordinal,
                    n_samples=options.n_samples,
                    padding=refine_padding,
                    min_ratio=options.min_ratio,
                    min_accepted=options.min_accepted,
                )
                results2, accepted2 = _try_track(
                    video_path, samples, track, refine_options, duration, tmpdir, session,
                    progress=emit,
                    base=(attempt + 1) * len(samples),
                    span=len(samples),
                    coarse_offset=coarse_offset,
                )
                fraction2 = accepted2 / max(1, len(results2))
                if fraction2 >= fraction:
                    last_results, last_accepted = results2, accepted2
                    fraction = fraction2

            if fraction >= options.min_accepted:
                break

    if progress is not None:
        progress("done", 1.0)

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
    # Use dominant cluster to eliminate isolated outliers / cuts
    cluster = _find_dominant_cluster(offsets, tolerance=6.0)
    if cluster and len(cluster) >= 3:
        offset = statistics.median(cluster)
        mad = statistics.median(sorted(abs(o - offset) for o in cluster))
    else:
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