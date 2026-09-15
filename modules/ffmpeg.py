from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

import imageio_ffmpeg

_DURATION_RE = re.compile(r"Duration:\s*(\d{2}):(\d{2}):(\d{2}\.\d+)")
_AUDIO_STREAM_RE = re.compile(
    r"^\s*Stream #0:(\d+)(?:\(([^)]*)\))?\s*:\s*Audio:", re.IGNORECASE
)


class FfmpegError(RuntimeError):
    pass


@dataclass
class AudioTrack:
    stream_index: int
    ordinal: int
    language: str | None
    codec: str
    layout: str
    title: str
    is_default: bool


def get_ffmpeg_binary() -> str:
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise FfmpegError(f"could not locate bundled ffmpeg: {exc}")


def probe(video_path: str) -> tuple[list[AudioTrack], float]:
    ffmpeg = get_ffmpeg_binary()
    cmd = [ffmpeg, "-hide_banner", "-i", str(video_path), "-t", "0", "-f", "null", "-"]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=180, creationflags=0x08000000
        )
    except subprocess.TimeoutExpired as exc:
        raise FfmpegError(f"ffmpeg probe timed out: {exc}")
    except OSError as exc:
        raise FfmpegError(f"could not run ffmpeg: {exc}")

    stderr = proc.stderr or ""
    if "Unknown format" in stderr or "Invalid data" in stderr:
        raise FfmpegError(f"ffmpeg could not read video: {video_path}")

    if "No such file" in stderr:
        raise FfmpegError(f"video file not found: {video_path}")

    duration = 0.0
    dm = _DURATION_RE.search(stderr)
    if dm:
        duration = int(dm.group(1)) * 3600 + int(dm.group(2)) * 60 + float(dm.group(3))

    tracks: list[AudioTrack] = []
    audio_ordinal = 0
    for line in stderr.splitlines():
        if line.strip().startswith("Stream mapping:"):
            break
        sm = _AUDIO_STREAM_RE.search(line)
        if not sm:
            continue
        stream_index = int(sm.group(1))
        lang = sm.group(2).strip() if sm.group(2) else None
        codec_m = re.search(r"Audio:\s*([^,\s]+)", line)
        codec = codec_m.group(1) if codec_m else "?"
        layout_m = re.search(
            r"(\d+ channels|mono|stereo|5\.1\(side\)|5\.1|7\.1|4\.0|2\.0|quad|surround)", line
        )
        layout = layout_m.group(1) if layout_m else "?"
        title_m = re.search(r"title\s*:\s*([^,]+)", line)
        title = title_m.group(1).strip() if title_m else ""
        is_default = "(default)" in line
        tracks.append(
            AudioTrack(
                stream_index=stream_index,
                ordinal=audio_ordinal,
                language=lang,
                codec=codec,
                layout=layout,
                title=title,
                is_default=is_default,
            )
        )
        audio_ordinal += 1

    return tracks, duration


def _eng_score(track: AudioTrack) -> tuple[int, int, bool]:
    lang = (track.language or "").lower()
    title = (track.title or "").lower()
    if lang in ("eng", "english", "en"):
        return 3, -track.ordinal, track.is_default
    if "eng" in lang or "english" in lang or "en" in lang:
        return 2, -track.ordinal, track.is_default
    if "english" in title or " eng" in title or title.startswith("eng"):
        return 2, -track.ordinal, track.is_default
    return 1, -track.ordinal, track.is_default


def ordered_tracks(tracks: list[AudioTrack]) -> list[AudioTrack]:
    return sorted(tracks, key=_eng_score, reverse=True)


def select_track(tracks: list[AudioTrack], force_ordinal: int | None = None) -> AudioTrack:
    if not tracks:
        raise FfmpegError("no audio tracks found in the video")
    if len(tracks) == 1:
        return tracks[0]
    if force_ordinal is not None:
        if not 0 <= force_ordinal < len(tracks):
            raise FfmpegError(f"--track {force_ordinal} out of range (0..{len(tracks)-1})")
        return tracks[force_ordinal]
    return ordered_tracks(tracks)[0]


def extract_clip(
    video_path: str,
    track: AudioTrack,
    start: float,
    end: float,
    out_wav: str,
) -> None:
    ffmpeg = get_ffmpeg_binary()
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        str(video_path),
        "-map",
        f"0:{track.stream_index}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        out_wav,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180, creationflags=0x08000000)
    except subprocess.TimeoutExpired as exc:
        raise FfmpegError(f"ffmpeg clip extraction timed out: {exc}")
    except OSError as exc:
        raise FfmpegError(f"could not run ffmpeg: {exc}")
    if proc.returncode != 0:
        raise FfmpegError(f"ffmpeg failed to extract clip {start:.1f}-{end:.1f}s: {proc.stderr.strip()}")