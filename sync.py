from __future__ import annotations

import argparse
import sys
from pathlib import Path

from modules.ffmpeg import FfmpegError, probe, ordered_tracks
from modules.pipeline import SubtitleSyncError, format_offset, run_task
from modules.syncer import SyncOptions


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
    options = SyncOptions(
        model=args.model,
        device=args.device,
        language=args.language,
        track_ordinal=args.track,
        n_samples=args.samples,
        padding=args.padding,
        min_accepted=args.min_accepted,
    )

    def on_progress(message: str, fraction: float) -> None:
        sys.stderr.write(f"\r  {fraction * 100:5.1f}%  {message:<52}")
        sys.stderr.flush()

    auto_save = not args.no_corrected
    try:
        result = run_task(
            args.video,
            args.subtitle,
            options,
            progress=on_progress,
            auto_save=auto_save,
            output_path=args.output,
        )
    except SubtitleSyncError as exc:
        sys.stderr.write("\r" + " " * 70 + "\r")
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    sys.stderr.write("\r" + " " * 70 + "\r")

    print(f"\n{'='*60}")
    print(f"detected offset: {format_offset(result.offset)}")
    print(f"confidence:     {result.accepted_count}/{result.sample_count} samples matched "
          f"(median absolute deviation: {result.mad:.2f}s)")
    print(f"track used:      ordinal={result.track_ordinal} "
          f"lang={result.track_language or '(none)'}")
    print(f"device:          {result.device}")
    print(f"duration:        {int(result.duration//3600)}h {int((result.duration%3600)//60)}m "
          f"{result.duration%60:.1f}s")
    print(f"{'='*60}")

    if result.low_confidence:
        print(f"\nWARNING: low match rate ({result.accepted_count}/{result.sample_count}). "
              f"The detected offset is likely unreliable.")
        print("This usually means the wrong audio track, a different subtitle release,")
        print("or the subtitles match a different episode.")
        print("Check available tracks with `sync.py list-tracks <video>`, then retry with")
        print("`--track <ordinal>`. To accept the result anyway, use `--min-accepted`.")
        sys.exit(1)

    if args.verbose:
        print("\nsample details:")
        for s in result.per_sample:
            tag = "OK" if s["accepted"] else "  "
            print(
                f"  [{tag}] sub#{s['index']:4d} @ {s['start']:8.2f}s | "
                f"offset={s['offset']:+8.3f}s | sim={s['similarity']:.2f} | "
                f"transcribed: {s['transcribed'][:60]}"
            )

    if result.output_path:
        print(f"\ncorrected subtitle saved to: {result.output_path}")


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