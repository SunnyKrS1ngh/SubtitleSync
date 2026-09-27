# SubtitleSync

Detect and fix subtitle desync by **sampling** short audio clips from a video and matching the spoken words against the subtitle text — no need to transcribe the whole movie.

Given a video and a `.srt` file, it reports the offset in seconds (the exact number to use in VLC's subtitle sync) and writes a corrected `.srt` next to the original.

---

## Quick start (GUI — for everyone)

**Requires:** Windows 10/11, Python 3.10+ installed.

1. **Download** the latest release zip from the Releases page and unzip it.
2. **Double-click `run.bat`**.
   - The first run creates a Python environment, downloads ~180 MB of components, and fetches the speech model (141 MB). This takes a few minutes and only happens once.
   - A browser window opens automatically at `http://127.0.0.1:8765/`.
3. **Pick your video** (Browse…) and **pick your `.srt` subtitle**.
4. Click **Find the desync**.
   - A progress bar shows each sample being checked. On a typical laptop this takes 2–5 minutes (CPU) or ~10 seconds (GPU — see below).
5. **Read the verdict** — e.g. *"Your subtitles were 10.8s too early — delayed by 10.8s."*
6. The corrected file is **already saved** next to your subtitle as `movie.corrected.srt`. Use **Save as…** if you want it elsewhere.

That's it. The corrected `.srt` works in VLC, MPC-BE, MPV, Plex, Jellyfin, etc.

### Speed it up with your graphics card (optional)

If you have an NVIDIA GPU, a banner appears: **"Speed up with your graphics card"**. Click **Enable GPU** — it downloads ~2 GB of CUDA libraries once, then every run uses the GPU (10–20× faster). AMD/Intel GPUs are not supported and will use the CPU.

---

## Command line (for power users)

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\Activate.ps1     # PowerShell

# 2. Install dependencies (CPU only — no 2 GB CUDA download)
pip install -r requirements.txt

# 3. List audio tracks in a video
python sync.py list-tracks movie.mkv

# 4. Detect offset and write corrected .srt
python sync.py sync movie.mkv subtitles.srt

# 5. Force a specific audio track (ordinal from list-tracks)
python sync.py sync movie.mkv subtitles.srt --track 1

# 6. Use GPU if you have the CUDA wheels
pip install -r requirements-gpu.txt
python sync.py sync movie.mkv subtitles.srt --device cuda
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `base` | Whisper model: `tiny`, `base`, `small`, `medium` |
| `--device` | `auto` | `auto`, `cuda`, or `cpu` |
| `--language` | auto | Force Whisper language code (e.g. `en`, `hi`) |
| `--track` | auto | Audio track ordinal (see `list-tracks`) |
| `--samples` | 15 | Subtitle lines to sample |
| `--padding` | 15 | Clip padding in seconds around each line |
| `--min-accepted` | 0.3 | Min fraction of matched samples to trust the offset |
| `--output`, `-o` | auto | Output corrected `.srt` path |
| `--no-corrected` | — | Skip writing the corrected file |
| `--verbose`, `-v` | — | Print per-sample details |

---

## How it works (in 30 seconds)

1. Parses the `.srt` file.
2. Picks ~15 subtitle lines spread evenly across the movie.
3. Extracts a ~30 s audio clip centered on each line (15 s padding each side).
4. Transcribes each clip with Whisper (`faster-whisper`), which is very fast on an NVIDIA GPU.
5. Fuzzy-matches the transcription back to the subtitle text and computes the offset (spoken time − subtitle time) per sample.
6. Reports the **median** offset (robust against a few bad samples, e.g. music or silence) and a confidence score.

The magic is that it only ever transcribes a few minutes of audio in total, not the whole movie. Because each clip has 15 s of padding, offsets up to **±15 s** are detectable (widen with `--padding` for more).

---

## Multi-language videos

Many releases contain several audio tracks. SubtitleSync auto-selects the track that *seems* English (language tag → track name → non-default track), and if the match rate is too low it automatically retries the other tracks.

For a dual-audio file, pre-check with:

```bash
python sync.py list-tracks movie.mkv
# Audio tracks in movie.mkv:
#   [0] lang=hin   (default)
#   [1] lang=eng
```

…then force the English one with `--track 1` if needed.

---

## When things go wrong

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `low match rate (8% < 30%)` | Wrong audio track, different subtitle release, or subtitles don't match the dialogue | Run `list-tracks`, then retry with `--track <ordinal>` |
| Offset larger than 15 s | Desync exceeds padding | Increase `--padding` |
| Variable/drifting offsets | The tool reports a single *global* offset | Use `--verbose` to see per-sample offsets and spot drift |
| "Video file not found" in the GUI | Path typo or the file moved | Use the Browse button to pick it again |
| GPU button doesn't appear | No NVIDIA GPU detected, or drivers missing | CPU is used automatically; install NVIDIA drivers for GPU support |

The CLI refuses to write a corrected file when confidence is low. The GUI warns but lets you save anyway.

---

## Project layout

```
run.bat              # Windows bootstrap: venv + install + launch
app.py               # Local web server (ThreadingHTTPServer)
modules/
  pipeline.py        # Shared sync orchestration (no sys.exit)
  syncer.py          # Sampling, track fallback, median/MAD
  transcribe.py      # faster-whisper wrapper (CUDA with CPU fallback)
  ffmpeg.py          # bundled ffmpeg discovery, track probing, clip extraction
  srt_parser.py      # SRT parsing, tag/bracket stripping, tokenization
  matcher.py         # fuzzy text alignment + per-sample offset
  corrected_srt.py   # offset shifting and .srt writing
  webui/
    index.html       # UI markup
    app.css          # styling
    app.js           # front-end logic (file picker, progress polling)
test/                # unit tests (pytest)
requirements.txt     # CPU-only dependencies (~180 MB)
requirements-gpu.txt # + NVIDIA CUDA wheels (~2 GB extra)
```

---

## Requirements

- Python 3.10+ (tested on 3.11)
- Windows 10/11 (the bootstrap is a `.bat` file)
- ~400 MB disk for the first run (CPU only)
- NVIDIA GPU + drivers for GPU acceleration (optional; adds ~2 GB)

The `nvidia-cublas/cudnn/runtime` packages in `requirements-gpu.txt` provide the CUDA libraries needed for GPU transcription. If you don't have an NVIDIA GPU, you don't need them — the tool automatically falls back to CPU.

---

## Tests

```bash
python -m pytest test -q
```

---

## Notes / limitations

- Designed for effectively constant (per-episode) desync of up to ~±15 s.
- Best results with subtitles that closely match the spoken English dialogue.
- Uses `faster-whisper` under the hood; the first run downloads the model (~140 MB for `base`) to the HuggingFace cache.
- ffmpeg is bundled (`imageio-ffmpeg`), no system install required.
- The GUI runs a local HTTP server on `127.0.0.1` only. A random token in the URL prevents other websites from reaching it. Nothing is uploaded anywhere — video never leaves your computer.
- Cues that become empty after tag/bracket stripping are dropped from the output, so the corrected file may have fewer entries than the input. This is pre-existing behavior.

---

## License

MIT