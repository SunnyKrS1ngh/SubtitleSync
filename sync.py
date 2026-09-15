from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from modules.corrected_srt import default_output_path, shift_entries, write_srt
from modules.ffmpeg import FfmpegError, probe, select_track, ordered_tracks
from modules.srt_parser import parse_srt_file
from modules.syncer import SyncOptions, SyncResult, run_sync
from modules.transcribe import WhisperSession


def _fmt_offset(seconds: float) -> str:
    sign = "+" if seconds >= 0 else ""
    m, s = divmod(abs(seconds), 60)
    return f"{sign}{seconds:.2f}s (delay subtitles by {seconds:.2f}s)" if seconds >= 0 else f"{seconds:.2f}s (advance subtitles by {abs(seconds):.2f}s)"


def cmd_list_tracks(video: str) -> None:
    try:
        tracks, duration = probe(video)
    except FfmpegError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Audio tracks in {Path(video).name}:")
    if not tracks:
        print("  (none)")
        sys.exit(0)
    ordered = ordered_tracks(tracks)
    for rank, t in enumerate(ordered):
        default = " (default)" if t.is_default else ""
        print(
            f"  [{t.ordinal}] stream #{t.stream_index} | "
            f"lang={t.language or '(none)':>5} | codec={t.codec} | "
            f"layout={t.layout}{default}"
        )
    print(f"\nTrack ordinal is the 0-based index in the audio-track list above.")
    print(f"Use --track <ordinal> to select a specific audio track.")
    if duration:
        m, s = divmod(duration, 60)
        h, m = divmod(int(m), 60)
        print(f"Duration: {int(h)}h {int(m)}m {s:.1f}s")


def cmd_sync(args: argparse.Namespace) -> None:
    video = args.video
    srt = args.subtitle

    if not Path(video).is_file():
        print(f"error: video file not found: {video}", file=sys.stderr)
        sys.exit(1)
    if not Path(srt).is_file():
        print(f"error: subtitle file not found: {srt}", file=sys.stderr)
        sys.exit(1)

    print("parsing subtitles ...")
    entries = parse_srt_file(srt)
    if not entries:
        print("error: no subtitle entries parsed", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(entries)} subtitle entries loaded")

    options = SyncOptions(
        model=args.model,
        device=args.device,
        language=args.language,
        track_ordinal=args.track,
        n_samples=args.samples,
        padding=args.padding,
        min_accepted=args.min_accepted,
    )

    print("loading whisper model ...")
    t0 = time.time()
    session = WhisperSession(options.model, options.device, options.language)
    print(f"  model '{options.model}' loaded in {time.time()-t0:.1f}s on {session.device}")

    print("analyzing audio tracks ...")
    try:
        tracks, duration = probe(video)
    except FfmpegError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    best = select_track(tracks, args.track)
    print(f"  selected track: ordinal={best.ordinal} lang={best.language or '(none)'} "
          f"{'(default)' if best.is_default else ''}")
    if args.track is None and len(tracks) > 1:
        print(f"  (auto-selected; use --track <0..{len(tracks)-1}> to override)")

    print(f"sampling up to {options.n_samples} subtitle segments ...")
    t0 = time.time()
    result = run_sync(video, entries, options, session)
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s")

    print(f"\n{'='*60}")
    if result.offset is None:
        print("no matching subtitle lines found - wrong subtitle file or wrong audio track?")
        print(f"tried track: ordinal={result.track.ordinal} lang={result.track.language or '(none)'}")
        print("try --list-tracks to see all audio tracks, then --track <ordinal> to try another.")
        sys.exit(1)

    print(f"detected offset: {_fmt_offset(result.offset)}")
    print(f"confidence:     {result.accepted_count}/{result.sample_count} samples matched "
          f"(median absolute deviation: {result.mad:.2f}s)")
    print(f"track used:      ordinal={result.track.ordinal} lang={result.track.language or '(none)'}")
    print(f"duration:        {int(result.duration//3600)}h {int((result.duration%3600)//60)}m "
          f"{result.duration%60:.1f}s")
    print(f"{'='*60}")

    fraction = result.accepted_count / max(1, result.sample_count)
    if fraction < options.min_accepted:
        print(f"\nWARNING: low match rate ({fraction*100:.0f}% < "
              f"{options.min_accepted*100:.0f}%). The detected offset is likely unreliable.")
        print("This usually means the wrong audio track, a different subtitle release,")
        print("or the subtitles match a different episode.")
        print("Check available tracks with `sync.py list-tracks <video>`, then retry with")
        print("`--track <ordinal>`. To accept the result anyway, use `--min-accepted`.")
        sys.exit(1)

    if args.verbose:
        print("\nsample details:")
        for r in result.per_sample:
            tag = "OK" if r.accepted else "  "
            print(
                f"  [{tag}] sub#{r.subtitle_index:4d} @ {r.subtitle_start:8.2f}s | "
                f"offset={r.offset:+8.3f}s | sim={r.similarity:.2f} | "
                f"transcribed: {r.transcribed[:60]}"
            )

    if not args.no_corrected and args.output is None:
        out_path = default_output_path(srt)
    else:
        out_path = args.output

    if not args.no_corrected and out_path:
        shifted = shift_entries(entries, result.offset)
        write_srt(shifted, out_path)
        print(f"\ncorrected subtitle saved to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="subtitlesync",
        description="Detect subtitle desync by sampling audio and matching against subtitle text.",
    )
    sub = parser.add_subparsers(dest="command")

    p_tracks = sub.add_parser("list-tracks", help="list audio tracks in a video file")
    p_tracks.add_argument("video", help="path to the video file")

    p_sync = sub.add_parser("sync", help="compute offset and generate corrected subtitle")
    p_sync.add_argument("video", help="path to the video file")
    p_sync.add_argument("subtitle", help="path to the .srt subtitle file")
    p_sync.add_argument("--model", default="base", help="whisper model size (default: base)")
    p_sync.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                        help="compute device (default: auto)")
    p_sync.add_argument("--language", default=None, help="force whisper language code (default: auto)")
    p_sync.add_argument("--track", default=None, type=int, help="audio track ordinal 0-based")
    p_sync.add_argument("--samples", default=15, type=int, help="number of subtitle samples (default: 15)")
    p_sync.add_argument("--padding", default=15.0, type=float, help="clip padding in seconds (default: 15)")
    p_sync.add_argument("--min-accepted", default=0.3, type=float,
                        help="min fraction of matched samples to trust the offset (default: 0.3)")
    p_sync.add_argument("--output", "-o", default=None, help="output corrected .srt path")
    p_sync.add_argument("--no-corrected", action="store_true", help="skip writing corrected .srt")
    p_sync.add_argument("--verbose", "-v", action="store_true", help="print per-sample results")

    args = parser.parse_args()
    if args.command == "list-tracks":
        cmd_list_tracks(args.video)
    elif args.command == "sync":
        cmd_sync(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()