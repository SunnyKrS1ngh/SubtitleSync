from __future__ import annotations

import json
import mimetypes
import os
import secrets
import subprocess
import sys
import threading
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from modules.corrected_srt import default_output_path, shift_entries, write_srt
from modules.ffmpeg import FfmpegError, ordered_tracks, probe
from modules.pipeline import (
    SubtitleSyncError,
    format_offset,
    nvidia_present,
    resolve_device,
    run_task,
)
from modules.srt_parser import parse_srt_file
from modules.syncer import SyncOptions

WEBUI_DIR = Path(__file__).parent / "modules" / "webui"
VIDEO_EXTENSIONS = {".mkv", ".mp4", ".avi", ".m4v", ".mov", ".webm", ".wmv", ".ts", ".mpg", ".mpeg", ".flv"}
SUBTITLE_EXTENSIONS = {".srt"}
MODELS = ["base", "tiny", "small", "medium"]
MAX_LIST_ENTRIES = 2000


# --------------------------------------------------------------------------
# filesystem helpers (pure functions, unit tested in test/test_webui.py)
# --------------------------------------------------------------------------

def is_safe_path(candidate: str) -> bool:
    if not candidate or not candidate.strip():
        return False
    try:
        return Path(candidate).expanduser().is_absolute()
    except (OSError, ValueError):
        return False


def list_drives() -> list[str]:
    if os.name != "nt":
        return ["/"]
    import string as _string

    drives: list[str] = []
    for letter in _string.ascii_uppercase:
        root = f"{letter}:\\"
        if os.path.exists(root):
            drives.append(root)
    return drives


def list_directory(path: str, extensions: set[str] | None = None) -> dict:
    target = Path(path).expanduser()
    if not target.is_dir():
        raise FileNotFoundError(f"Not a folder: {path}")

    entries: list[dict] = []
    with os.scandir(target) as it:
        for item in it:
            try:
                is_dir = item.is_dir()
            except OSError:
                continue
            if not is_dir and extensions:
                suffix = Path(item.path).suffix.lower()
                if suffix not in extensions:
                    continue
            try:
                size = 0 if is_dir else item.stat().st_size
            except OSError:
                size = 0
            entries.append({
                "name": item.name,
                "path": item.path,
                "is_dir": is_dir,
                "size": size,
            })

    entries.sort(key=lambda e: (not e["is_dir"], e["name"].lower()))
    if len(entries) > MAX_LIST_ENTRIES:
        entries = entries[:MAX_LIST_ENTRIES]

    parent = str(target.parent) if target.parent != target else ""
    return {
        "path": str(target),
        "parent": parent,
        "entries": entries,
        "truncated": len(entries) >= MAX_LIST_ENTRIES,
    }


def build_srt_filename(srt_path: str) -> str:
    return Path(default_output_path(srt_path)).name


# --------------------------------------------------------------------------
# job bookkeeping
# --------------------------------------------------------------------------

@dataclass
class Job:
    job_id: str
    status: str = "running"          # running | done | error
    message: str = "Starting..."
    fraction: float = 0.0
    result: dict | None = None
    error: str | None = None
    per_sample: list[dict] = field(default_factory=list)


@dataclass
class GpuJob:
    status: str = "idle"             # idle | running | done | error
    message: str = ""
    log: list[str] = field(default_factory=list)


class Store:
    """Single-user local state. A lock is enough; there is no concurrency to speak of."""

    def __init__(self) -> None:
        self.token = secrets.token_urlsafe(24)
        self.jobs: dict[str, Job] = {}
        self.gpu = GpuJob()
        self.lock = threading.Lock()
        self.gpu_thread: threading.Thread | None = None

    def put(self, job: Job) -> None:
        with self.lock:
            self.jobs[job.job_id] = job
            if len(self.jobs) > 20:
                for key in list(self.jobs)[:-20]:
                    if key != job.job_id:
                        del self.jobs[key]


STORE = Store()


def _start_sync(payload: dict) -> Job:
    video = payload["video"]
    srt = payload["subtitle"]
    options = SyncOptions(
        model=payload.get("model", "base"),
        device=payload.get("device", "auto"),
        track_ordinal=payload.get("track"),
        n_samples=int(payload.get("samples", 15)),
        padding=float(payload.get("padding", 15.0)),
        min_accepted=float(payload.get("min_accepted", 0.3)),
    )

    job = Job(job_id=secrets.token_urlsafe(12))

    def on_progress(message: str, fraction: float) -> None:
        job.message = message
        job.fraction = fraction

    def worker() -> None:
        try:
            result = run_task(video, srt, options, progress=on_progress, auto_save=True)
            job.result = {
                "offset": result.offset,
                "offset_text": format_offset(result.offset),
                "mad": round(result.mad, 3),
                "accepted": result.accepted_count,
                "sample_count": result.sample_count,
                "duration": result.duration,
                "track_ordinal": result.track_ordinal,
                "track_language": result.track_language,
                "device": result.device,
                "elapsed": round(result.elapsed, 1),
                "low_confidence": result.low_confidence,
                "output_path": result.output_path,
                "video_path": result.video_path,
                "srt_path": result.srt_path,
                "per_sample": result.per_sample,
            }
            job.status = "done"
            job.fraction = 1.0
            job.message = "Done"
        except SubtitleSyncError as exc:
            job.status = "error"
            job.error = str(exc)
        except Exception as exc:                      # noqa: BLE001 - last resort for the UI
            job.status = "error"
            job.error = f"Unexpected problem: {exc}"

    STORE.put(job)
    threading.Thread(target=worker, daemon=True).start()
    return job


def _start_gpu_install() -> GpuJob:
    gpu = STORE.gpu
    if gpu.status == "running":
        return gpu

    req = Path(__file__).parent / "requirements-gpu.txt"
    exe = sys.executable
    gpu.status = "running"
    gpu.message = "Downloading GPU support (about 2 GB)..."
    gpu.log = ["Starting download..."]

    def worker() -> None:
        try:
            proc = subprocess.Popen(
                [exe, "-m", "pip", "install", "-r", str(req)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=0x08000000 if os.name == "nt" else 0,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                text = line.rstrip()
                if text:
                    gpu.log.append(text)
                    if len(gpu.log) > 300:
                        del gpu.log[0]
            code = proc.wait()
            if code == 0:
                gpu.status = "done"
                gpu.message = "Installed. Restart SubtitleSync to use the GPU."
            else:
                gpu.status = "error"
                gpu.message = f"pip exited with code {code}. See the log below."
        except Exception as exc:                      # noqa: BLE001
            gpu.status = "error"
            gpu.message = f"Could not run pip: {exc}"

    STORE.gpu_thread = threading.Thread(target=worker, daemon=True)
    STORE.gpu_thread.start()
    return gpu


# --------------------------------------------------------------------------
# HTTP layer
# --------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "SubtitleSync"
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:              # keep the console quiet
        return

    # -- helpers ----------------------------------------------------------
    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, payload: dict, code: int = 200) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def _authorised(self) -> bool:
        supplied = self.headers.get("X-Auth-Token", "")
        return bool(supplied) and secrets.compare_digest(supplied, STORE.token)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > 4_000_000:
            raise ValueError("Request body too large")
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc
        return data if isinstance(data, dict) else {}

    # -- routing ----------------------------------------------------------
    def do_GET(self) -> None:                            # noqa: N802
        parsed = urlparse(self.path)
        route = parsed.path
        query = parse_qs(parsed.query)

        if route in ("/", "/index.html"):
            self._serve_static("index.html")
            return
        if route.startswith("/static/"):
            self._serve_static(route[len("/static/"):])
            return

        if not self._authorised():
            self._json({"error": "Not authorised"}, 401)
            return

        try:
            if route == "/api/state":
                self._json(_api_state())
            elif route == "/api/browse":
                self._json(_api_browse(query))
            elif route == "/api/drives":
                self._json({"drives": list_drives()})
            elif route == "/api/tracks":
                self._json(_api_tracks(query))
            elif route == "/api/progress":
                self._json(_api_progress(query))
            elif route == "/api/gpu":
                self._json(_gpu_state())
            else:
                self._json({"error": "Not found"}, 404)
        except SubtitleSyncError as exc:
            self._json({"error": str(exc)}, 400)
        except FileNotFoundError as exc:
            self._json({"error": str(exc)}, 404)
        except Exception as exc:                        # noqa: BLE001
            self._json({"error": f"Unexpected problem: {exc}"}, 500)

    def do_POST(self) -> None:                           # noqa: N802
        route = urlparse(self.path).path
        if not self._authorised():
            self._json({"error": "Not authorised"}, 401)
            return

        try:
            payload = self._read_json()
            if route == "/api/sync":
                job = _start_sync(payload)
                self._json({"job_id": job.job_id})
            elif route == "/api/save":
                self._json(_api_save(payload))
            elif route == "/api/gpu":
                self._json({"status": _start_gpu_install().status})
            elif route == "/api/shutdown":
                self._json({"ok": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            else:
                self._json({"error": "Not found"}, 404)
        except SubtitleSyncError as exc:
            self._json({"error": str(exc)}, 400)
        except FileNotFoundError as exc:
            self._json({"error": str(exc)}, 404)
        except ValueError as exc:
            self._json({"error": str(exc)}, 400)
        except Exception as exc:                        # noqa: BLE001
            self._json({"error": f"Unexpected problem: {exc}"}, 500)

    def _serve_static(self, name: str) -> None:
        if "/" in name or "\\" in name or ".." in name:
            self._json({"error": "Bad path"}, 400)
            return
        target = WEBUI_DIR / name
        if not target.is_file():
            self._json({"error": "Not found"}, 404)
            return
        ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript",):
            ctype += "; charset=utf-8"
        self._send(200, target.read_bytes(), ctype)


# --------------------------------------------------------------------------
# API handlers
# --------------------------------------------------------------------------

def _api_state() -> dict:
    device = resolve_device()
    has_nvidia = nvidia_present()
    return {
        "device": device,
        "has_nvidia": has_nvidia,
        "gpu_enabled": device == "cuda",
        "gpu_status": STORE.gpu.status,
        "gpu_message": STORE.gpu.message,
        "models": MODELS,
        "python": sys.version.split()[0],
    }


def _api_browse(query: dict) -> dict:
    kind = (query.get("kind") or ["video"])[0]
    path = (query.get("path") or [""])[0]
    exts = VIDEO_EXTENSIONS if kind == "video" else SUBTITLE_EXTENSIONS
    if not is_safe_path(path):
        path = str(Path.home())
    data = list_directory(path, exts)
    data["kind"] = kind
    return data


def _api_tracks(query: dict) -> dict:
    video = (query.get("path") or [""])[0]
    if not is_safe_path(video) or not Path(video).is_file():
        raise SubtitleSyncError("Choose a video file first.")
    try:
        tracks, duration = probe(video)
    except FfmpegError as exc:
        raise SubtitleSyncError(f"Could not read the video file: {exc}") from exc
    return {
        "duration": duration,
        "tracks": [
            {
                "ordinal": t.ordinal,
                "language": t.language,
                "codec": t.codec,
                "layout": t.layout,
                "is_default": t.is_default,
            }
            for t in ordered_tracks(tracks)
        ],
    }


def _api_progress(query: dict) -> dict:
    job_id = (query.get("job") or [""])[0]
    with STORE.lock:
        job = STORE.jobs.get(job_id)
        if job is None:
            raise SubtitleSyncError("That job has expired. Please run it again.")
        return {
            "status": job.status,
            "message": job.message,
            "fraction": job.fraction,
            "result": job.result,
            "error": job.error,
            "per_sample": job.result["per_sample"] if job.result else [],
        }


def _api_save(payload: dict) -> dict:
    video = payload.get("video", "")
    srt = payload.get("subtitle", "")
    dest = payload.get("dest", "")
    if not dest:
        raise SubtitleSyncError("Choose where to save the file.")

    job_id = payload.get("job", "")
    with STORE.lock:
        job = STORE.jobs.get(job_id)
        offset = job.result["offset"] if job and job.result else None
    if offset is None:
        raise SubtitleSyncError("Run the check before saving.")

    entries = parse_srt_file(srt)
    if not entries:
        raise SubtitleSyncError("Could not read the subtitle file.")
    try:
        write_srt(shift_entries(entries, offset), dest)
    except OSError as exc:
        raise SubtitleSyncError(f"Could not write the file: {exc}") from exc
    return {"path": dest, "name": Path(dest).name}


def _gpu_state() -> dict:
    return {"status": STORE.gpu.status, "message": STORE.gpu.message, "log": STORE.gpu.log[-40:]}


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def free_port(preferred: int = 8765) -> int:
    import socket

    for port in range(preferred, preferred + 40):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> int:
    port = free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/?t={STORE.token}"

    print("SubtitleSync is running.")
    print(f"  {url}")
    print("  Leave this window open while you work. Press Ctrl+C to stop.")
    print()

    threading.Timer(0.7, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
