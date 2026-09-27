from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

from app import (
    build_srt_filename,
    is_safe_path,
    list_directory,
    list_drives,
)


def test_is_safe_path():
    # Windows absolute paths
    assert is_safe_path("C:\\Users\\x\\video.mkv")
    assert is_safe_path("D:/Movies/video.mkv")
    assert is_safe_path(r"\\server\share\video.mkv")

    # Unix paths only work on Unix
    if os.name != "nt":
        assert is_safe_path("/home/user/video.mkv")

    # Invalid
    assert not is_safe_path("")
    assert not is_safe_path("   ")
    assert not is_safe_path("relative/path.mkv")
    assert not is_safe_path("..\\video.mkv")
    assert not is_safe_path("video.mkv")


def test_list_drives():
    drives = list_drives()
    assert isinstance(drives, list)
    assert len(drives) >= 1
    if os.name == "nt":
        for d in drives:
            assert d.endswith(":\\")
    else:
        assert "/" in drives


def test_list_directory(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.mkv").write_bytes(b"x" * 1000)
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "c.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n\n")

    data = list_directory(str(tmp_path), {".mkv", ".srt"})
    assert data["path"] == str(tmp_path)
    assert data["parent"] == str(tmp_path.parent)
    names = {e["name"] for e in data["entries"]}
    assert "b.mkv" in names
    assert "a.txt" not in names  # filtered out
    assert "subdir" in names
    dirs = [e for e in data["entries"] if e["is_dir"]]
    files = [e for e in data["entries"] if not e["is_dir"]]
    assert len(dirs) == 1
    assert len(files) == 1
    assert files[0]["size"] == 1000


def test_list_directory_parent(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    data = list_directory(str(sub), None)
    assert data["parent"] == str(tmp_path)


def test_list_directory_root_windows():
    if os.name != "nt":
        pytest.skip("Windows only")
    root = "C:\\"
    data = list_directory(root, None)
    assert data["parent"] == ""


def test_list_directory_not_found():
    with pytest.raises(FileNotFoundError):
        list_directory("C:\\this_folder_does_not_exist_12345")


def test_list_directory_truncation(tmp_path):
    for i in range(2100):
        (tmp_path / f"f{i}.mkv").write_bytes(b"x")
    data = list_directory(str(tmp_path), {".mkv"})
    assert data["truncated"] is True
    assert len(data["entries"]) == 2000


def test_build_srt_filename():
    assert build_srt_filename("movie.srt") == "movie.corrected.srt"
    assert build_srt_filename("movie.SRT") == "movie.corrected.srt"
    assert build_srt_filename("movie") == "movie.corrected.srt"
    assert build_srt_filename("/path/to/movie.srt") == "movie.corrected.srt"