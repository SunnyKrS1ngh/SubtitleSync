from __future__ import annotations

import pytest

from modules.srt_parser import parse_srt, parse_srt_file

SAMPLE = """1
00:00:34,451 --> 00:00:35,702
Fuck.

2
00:00:52,344 --> 00:00:53,428
<i>Fuck!</i>

3
00:01:27,003 --> 00:01:28,463
Can I get some coffee?
Cash up front.

4
00:01:53,613 --> 00:01:55,949
[music] Nice trick.
"""


def test_parse_basic():
    entries = parse_srt(SAMPLE)
    assert len(entries) == 4
    assert entries[0].index == 1
    assert entries[0].start == pytest.approx(34.451)
    assert entries[0].end == pytest.approx(35.702)


def test_strips_tags():
    entries = parse_srt(SAMPLE)
    assert entries[1].text == "Fuck!"


def test_multiline_merge():
    entries = parse_srt(SAMPLE)
    assert entries[2].text == "Can I get some coffee? Cash up front."


def test_tokenize():
    entries = parse_srt(SAMPLE)
    assert entries[2].words == ["can", "i", "get", "some", "coffee", "cash", "up", "front"]


def test_bracket_cue_removed():
    entries = parse_srt(SAMPLE)
    assert entries[3].text == "Nice trick."


def test_crlf_and_bom():
    crlf = SAMPLE.replace("\n", "\r\n")
    entries = parse_srt("\ufeff" + crlf)
    assert len(entries) == 4


def test_malformed_blocks_skipped():
    bad = "9\n00:01:57,200 --> 00:01:58,742\nI'm not a thief.\n\n" + SAMPLE
    entries = parse_srt(bad)
    assert entries[0].index == 9


def test_real_file(tmp_path):
    p = tmp_path / "s.srt"
    p.write_text(SAMPLE, encoding="utf-8")
    assert len(parse_srt_file(str(p))) == 4