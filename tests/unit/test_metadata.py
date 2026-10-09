"""Tests for the audio metadata writer."""

import os

import pytest


@pytest.fixture()
def mp3_file(tmp_path):
    # Mutagen's ID3 can save to an empty file; no valid audio payload required
    # because write_metadata only touches tags, not audio data.
    dst = os.path.join(str(tmp_path), "sample.mp3")
    with open(dst, "wb") as f:
        f.write(b"\xff\xfb\x90\x00" * 100)
    from mutagen.id3 import ID3

    audio = ID3()
    audio.save(dst)
    return dst


def _cues():
    return [
        {"slot": 1, "type": "INTRO", "time": 0.07, "label": "START"},
        {"slot": 5, "type": "DROP", "time": 31.9, "label": "DROP"},
    ]


from core.metadata import CUE_TAG, write_metadata


def test_dry_run_writes_nothing(mp3_file):
    before = os.path.getmtime(mp3_file)
    result = write_metadata(
        mp3_file, bpm=130.0, key="12A", cues=_cues(), analysis={"bpm": 130.0, "key": "12A"}, dry_run=True
    )
    assert result["ok"] is True
    assert result["written"] is False
    assert result["dryRun"] is True
    assert result["tags"] == ["TBPM", "TKEY", f"TXXX:{CUE_TAG}"]
    assert os.path.getmtime(mp3_file) == before


def test_writes_id3_tags(mp3_file):
    from mutagen.id3 import ID3

    result = write_metadata(
        mp3_file, bpm=130.0, key="12A", cues=_cues(), analysis={"bpm": 130.0, "key": "12A"}
    )
    assert result["ok"] is True
    assert result["written"] is True
    assert result["cueCount"] == 2
    assert result["backup"] and os.path.isfile(result["backup"])

    tags = ID3(mp3_file)
    assert float(tags["TBPM"].text[0]) == pytest.approx(130.0)
    assert tags["TKEY"].text[0] == "12A"
    cue_tags = tags.getall(f"TXXX:{CUE_TAG}")
    assert len(cue_tags) == 1
    assert '"version": 1' in cue_tags[0].text[0]
    assert "DROP" in cue_tags[0].text[0]


def test_missing_file():
    result = write_metadata("Z:/does/not/exist.mp3", 128.0, "8A", [], None)
    assert result["ok"] is False
    assert "file not found" in result["error"]


def test_unsupported_extension(tmp_path):
    bogus = os.path.join(str(tmp_path), "track.txt")
    with open(bogus, "w") as f:
        f.write("hello")
    result = write_metadata(bogus, 128.0, "8A", [], None)
    assert result["ok"] is False
    assert "unsupported" in result["error"]


def test_writes_wav_without_existing_id3(tmp_path):
    import numpy as np
    import soundfile as sf

    wav = os.path.join(str(tmp_path), "sample.wav")
    sf.write(wav, np.zeros(22050, dtype=np.float32), 22050, subtype="PCM_16")

    result = write_metadata(
        wav, bpm=128.0, key="8A", cues=_cues(), analysis={"bpm": 128.0, "key": "8A"}
    )
    assert result["ok"] is True
    assert result["written"] is True
    assert result["cueCount"] == 2
    assert result["backup"] and os.path.isfile(result["backup"])

    from mutagen.wave import WAVE

    tags = WAVE(wav).tags
    assert tags is not None
    assert float(tags["TBPM"].text[0]) == pytest.approx(128.0)
    assert tags["TKEY"].text[0] == "8A"
    assert "DROP" in tags.getall(f"TXXX:{CUE_TAG}")[0].text[0]


def test_wav_dry_run_writes_nothing(tmp_path):
    import numpy as np
    import soundfile as sf

    wav = os.path.join(str(tmp_path), "sample.wav")
    sf.write(wav, np.zeros(22050, dtype=np.float32), 22050, subtype="PCM_16")

    result = write_metadata(
        wav, bpm=128.0, key="8A", cues=_cues(), analysis={"bpm": 128.0, "key": "8A"}, dry_run=True
    )
    assert result["ok"] is True
    assert result["written"] is False
    assert result["dryRun"] is True
