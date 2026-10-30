"""Tests for folder scanner."""

import os

from core.scanner import file_hash, is_supported, scan_folder
from tests.fixtures.synth import write_synth_track

# Short tracks for scanner tests only exercise hashing/metadata, not the full
# structure, so use a minimal bar-aligned structure.
_MIN = [("INTRO", 0.25), ("DROP", 1.0), ("OUTRO", 0.2)]


def test_is_supported_extensions():
    assert is_supported("a.mp3")
    assert is_supported("b.wav")
    assert is_supported("c.flac")
    assert not is_supported("d.txt")


def test_scan_single(tmp_path):
    wav = write_synth_track(str(tmp_path / "track1.wav"), seconds=8.0, structure=_MIN)
    result = scan_folder(str(tmp_path))
    assert result.total == 1
    track = result.tracks[0]
    assert track.path == wav
    assert track.status.value == "QUEUED"
    assert len(track.file_hash) == 64


def test_scan_duplicate_detection(tmp_path):
    write_synth_track(str(tmp_path / "a.wav"), seconds=8.0, structure=_MIN)
    write_synth_track(str(tmp_path / "b.wav"), seconds=8.0, structure=_MIN)
    result = scan_folder(str(tmp_path))
    statuses = {t.filename: t.status.value for t in result.tracks}
    assert result.duplicates == 1
    assert statuses["a.wav"] == "QUEUED"
    assert statuses["b.wav"] == "SKIPPED"


def test_file_hash_stable(tmp_path):
    wav = write_synth_track(str(tmp_path / "h.wav"), seconds=8.0, structure=_MIN)
    assert file_hash(wav) == file_hash(wav)
    assert len(file_hash(wav)) == 64


def test_track_id_case_sensitive_and_separator_normalised():
    from core.scanner import track_id_for_path

    # On a case-sensitive filesystem these are different files: their IDs
    # must differ (the old path.lower() hash collided here).
    assert track_id_for_path("/music/Track.mp3") != track_id_for_path("/music/track.mp3")
    # Separator normalisation only: Windows vs POSIX spelling of the same
    # path keeps the same ID.
    assert track_id_for_path("C:\\Music\\Track.mp3") == track_id_for_path("C:/Music/Track.mp3")
    assert len(track_id_for_path("/music/Track.mp3")) == 16
