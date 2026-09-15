from __future__ import annotations

import os
import site
import sys
from functools import lru_cache

from .matcher import WordStamp


class TranscribeError(RuntimeError):
    pass


def _register_nvidia_dlls() -> None:
    if sys.platform != "win32":
        return
    bin_dirs: list[str] = []
    registered: set[str] = set()
    for site_path in site.getsitepackages():
        base = os.path.join(site_path, "nvidia")
        if not os.path.isdir(base):
            continue
        for name in os.listdir(base):
            bindir = os.path.join(base, name, "bin")
            if os.path.isdir(bindir):
                try:
                    os.add_dll_directory(bindir)
                except OSError:
                    pass
                if bindir not in registered:
                    registered.add(bindir)
                    bin_dirs.append(bindir)
    if bin_dirs:
        path = os.environ.get("PATH", "")
        os.environ["PATH"] = ";".join(bin_dirs) + (";" + path if path else "")


@lru_cache(maxsize=1)
def _load_model(model_size: str, device: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscribeError(f"faster-whisper is not installed: {exc}")

    _register_nvidia_dlls()

    compute_type = "int8_float16" if device == "cuda" else "int8"
    try:
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=None,
        )
    except Exception as exc:
        if device == "cuda":
            raise TranscribeError(
                f"could not load whisper model on cuda (fall back with --device cpu): {exc}"
            )
        raise TranscribeError(f"could not load whisper model on cpu: {exc}")
    return model


class WhisperSession:
    def __init__(self, model_size: str = "base", device: str = "auto", language: str | None = None):
        self.model_size = model_size
        self.language = language
        resolved = device
        if resolved == "auto":
            resolved = _pick_device(model_size)
        self.device = resolved
        self._model = _load_model(model_size, resolved)

    def transcribe_clip(self, wav_path: str) -> tuple[list[WordStamp], str | None]:
        try:
            return self._transcribe(wav_path)
        except TranscribeError as exc:
            if self.device == "cuda" and _is_cuda_lib_missing(exc):
                print("warning: cuda unavailable, falling back to cpu", file=sys.stderr)
                self._model = _load_model(self.model_size, "cpu")
                self.device = "cpu"
                return self._transcribe(wav_path)
            raise

    def _transcribe(self, wav_path: str) -> tuple[list[WordStamp], str | None]:
        try:
            segments, info = self._model.transcribe(
                wav_path,
                language=self.language,
                beam_size=1,
                word_timestamps=True,
                vad_filter=True,
                condition_on_previous_text=False,
                without_timestamps=False,
            )
        except Exception as exc:
            raise TranscribeError(f"transcription failed: {exc}")

        stamps: list[WordStamp] = []
        for seg in segments:
            for w in seg.words or []:
                stamps.append(WordStamp(text=w.word.strip(), start=float(w.start), end=float(w.end)))
        return stamps, (info.language if info is not None else None)


def _is_cuda_lib_missing(exc: TranscribeError) -> bool:
    msg = str(exc).lower()
    return "cublas" in msg or "cudnn" in msg or "not found or cannot be loaded" in msg


def _pick_device(model_size: str) -> str:
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"