# SubtitleSync

Detect and fix subtitle desync by **sampling** short audio clips from a video and
matching the spoken words against the subtitle text — no need to transcribe the
whole movie.

Given a video and a `.srt` file, it reports the offset in seconds (the exact
number to use in VLC's subtitle sync) and optionally writes a corrected `.srt`.

## What it does

1. Parses the subtitle file.
2. Picks ~15 subtitle lines spread evenly across the movie.
3. Extracts a ~30 s audio clip centered on each line (15 s padding each side).
4. Transcribes each clip with Whisper (`faster-whisper`), which is very fast on
   an NVIDIA GPU — a 53-minute episode syncs in ~8 seconds.
5. Fuzzy-matches the transcription back to the subtitle text and computes the
   offset (spoken time − subtitle time) per sample.
6. Reports the **median** offset (robust against a few bad samples, e.g. music
   or silence) and a confidence score.

The magic is that it only ever transcribes a few minutes of audio in total,
not the whole movie. Because each clip has 15 s of padding, offsets up to
**±15 s** are detectable (widen with `--padding` for more).

## Usage example

```
detected offset: +10.82s (delay subtitles by 10.82s)
confidence:     6/12 samples matched (median absolute deviation: 0.12s)
```

- `+X s` → subtitles appear **early** → **delay** them by X s in VLC (key `h`, or
  via *Subtitle sync* in the tools menu, enter a positive delay in ms).
- `−X s` → subtitles appear **late** → **advance** them by X s in VLC (key `g`).

A corrected `.srt` is written next to the input file by default.

## Requirements

- Python 3.10+ (tested on 3.11)
- ~1.5 GB of free disk for the GPU runtime libraries (optional but recommended)
- NVIDIA GPU with CUDA (works on CPU too, just slower)

## Setup

```powershell
# 1. create and activate the virtual environment
python -m venv venv
venv\Scripts\Activate.ps1     # PowerShell

# 2. install dependencies (includes the bundled ffmpeg binary)
pip install -r requirements.txt
```

The `nvidia-cublas/cudnn/runtime` packages in `requirements.txt` provide the
CUDA libraries needed for GPU transcription. If you don't have an NVIDIA GPU,
remove those three lines (or leave them — they're just unused DLLs) and run with
`--device cpu`. The tool automatically falls back to CPU if CUDA libraries are
missing.

## Usage

```bash
# list audio tracks in a video (useful for multi-language files)
python sync.py list-tracks movie.mkv

# detect the offset and write a corrected .srt
python sync.py movie.mkv subtitles.srt

# force a specific audio track (ordinal from list-tracks)
python sync.py movie.mkv subtitles.srt --track 1

# speed / accuracy knobs
python sync.py movie.mkv subtitles.srt --model tiny   # faster, less accurate
python sync.py movie.mkv subtitles.srt --model small  # slower, more accurate
python sync.py movie.mkv subtitles.srt --samples 20   # more samples = more robust
python sync.py movie.mkv subtitles.srt --padding 30   # detect offsets up to ±30 s
python sync.py movie.mkv subtitles.srt --output fixed.srt

# see per-sample details
python sync.py movie.mkv subtitles.srt --verbose
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `base` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large` |
| `--device` | `auto` | `auto`, `cuda`, or `cpu` |
| `--language` | auto | Force Whisper language code (e.g. `en`, `hi`) |
| `--track` | auto | Audio track ordinal (see `list-tracks`), 0-based |
| `--samples` | 15 | Number of subtitle lines to sample |
| `--padding` | 15 | Clip padding in seconds around each line |
| `--min-accepted` | 0.3 | Min fraction of matched samples to trust the offset |
| `--output`, `-o` | auto | Output corrected `.srt` path |
| `--no-corrected` | — | Skip writing the corrected file (offset only) |
| `--verbose`, `-v` | — | Print per-sample transcription details |

## Multi-language videos

Many releases contain several audio tracks. SubtitleSync auto-selects the track
that *seems* English (language tag → track name → non-default track), and if
the match rate is too low it automatically retries the other tracks.

For a dual-audio file, pre-check with:

```bash
python sync.py list-tracks movie.mkv
# Audio tracks in movie.mkv:
#   [0] lang=hin   (default)
#   [1] lang=eng
```

…then force the English one with `--track 1` if needed.

## When things go wrong

- **`low match rate (8% < 30%)`** — almost always one of:
  - the wrong audio track was transcribed (try `--track`),
  - the subtitle file is for a different release/episode,
  - the subtitles barely match the spoken dialogue.
  The tool refuses to write a corrected file in this case. Use
  `--min-accepted` only if you really know what you're doing.
- **Offsets larger than 15 s** — raise `--padding`.
- **Variable/drifting offsets** — the tool reports a single *global* offset
  (what VLC supports). Use `--verbose` to see per-sample offsets and spot drift.

## Project layout

```
sync.py               # CLI entry point
modules/
  srt_parser.py       # SRT parsing, tag/bracket stripping, tokenization
  ffmpeg.py           # bundled ffmpeg discovery, track probing, clip extraction
  transcribe.py       # faster-whisper wrapper (CUDA with CPU fallback)
  matcher.py          # fuzzy text alignment + per-sample offset
  syncer.py           # sampling, track fallback, median/MAD orchestration
  corrected_srt.py    # offset shifting and .srt writing
test/                 # unit tests (pytest)
```

## Tests

```bash
python -m pytest test -q
```

## Notes / limitations

- Designed for effectively constant (per-episode) desync of up to ~±15 s.
- Best results with subtitles that closely match the spoken English dialogue.
- Uses `faster-whisper` under the hood; the first run downloads the model
  (~140 MB for `base`) to the HuggingFace cache.
- ffmpeg is bundled (`imageio-ffmpeg`), no system install required.