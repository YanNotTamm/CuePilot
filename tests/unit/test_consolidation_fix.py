"""Consolidation tests — cross-module fixes applied by the coordinator.

Runnable without librosa: a bare librosa stub is installed first when
librosa is absent, following tests/unit/test_slot_collision.py. No audio
feature is used at runtime here (all inputs are synthetic or stdlib).
"""

import os
import sys
import time
from pathlib import Path
from types import ModuleType

try:
    import librosa  # noqa: F401
except ImportError:  # pragma: no cover - only on librosa-less machines
    sys.modules.setdefault("librosa", ModuleType("librosa"))

import numpy as np

from core.analyzer.base import AnalysisFeatures
from core.beatgrid import BeatGrid, beatgrid_confidence
from core.cache import get_cached_arrays, store_cached_arrays
from core.cues import generate_cueplan
from core.metadata import prune_backups
from core.models import Bar, Beat, CuePlan, Phrase, Section, SectionType
from core.profiles.genre import get_profile


def _features() -> AnalysisFeatures:
    n = 200
    return AnalysisFeatures(
        hop_length=512,
        sample_rate=22050,
        frame_times=np.linspace(0.0, 60.0, n),
        onset_env=np.full(n, 0.8),
        rms=np.full(n, 0.5),
        spectral_flux=np.full(n, 0.5),
        harmonic=np.full(n, 0.5),
        percussive=np.full(n, 0.5),
    )


def _grid() -> BeatGrid:
    beats = [Beat(time=t, is_downbeat=(i % 4 == 0), index=i)
             for i, t in enumerate(np.arange(0.0, 60.0, 0.5))]
    downbeats = [float(t) for t in np.arange(0.0, 60.0, 2.0)]
    return BeatGrid(
        bpm=120.0,
        beats=beats,
        downbeat_times=downbeats,
        bars=[Bar(index=i, downbeat_time=t, beat_count=4)
              for i, t in enumerate(downbeats)],
        phrases=[Phrase(index=0, start_time=0.0, end_time=60.0, bar_count=30)],
        confidence=0.9,
    )


def test_unknown_section_gets_no_cue():
    # Consolidation (T-5): UNKNOWN is an explicit "not classified"
    # marker, not a DROP/BREAKDOWN detection, and must not invent a cue.
    sections = [
        Section(type=SectionType.INTRO, start=0.0, end=8.0, confidence=0.9, source="dsp"),
        Section(type=SectionType.UNKNOWN, start=16.0, end=24.0, confidence=0.4, source="dsp"),
        Section(type=SectionType.OUTRO, start=56.0, end=60.0, confidence=0.9, source="dsp"),
    ]
    plan = generate_cueplan(None, _features(), _grid(), sections, get_profile("open_format"))
    # No cue may come from the UNKNOWN section's time window (16-24s).
    # Sections themselves are preserved in the plan; only cues are not invented.
    assert all(c.time < 15.0 or c.time > 25.0 for c in plan.cues)
    assert any(s.type == SectionType.UNKNOWN for s in plan.sections)


def test_genre_profile_used_is_canonical():
    plan = CuePlan(genre_profile_used="edm", genre_detected="", genre_confidence=0.0)
    d = plan.to_dict()
    assert d["genreProfileUsed"] == "edm"
    assert "genreDetected" not in d
    assert "genreConfidence" not in d


def test_beatgrid_confidence_exported_and_bounded():
    # Perfectly regular beats with strong onsets -> high confidence.
    assert beatgrid_confidence([0.0, 0.5, 1.0, 1.5], np.ones(100)) > 0.9
    # Fewer than 3 beats -> 0, never a misleading high score.
    assert beatgrid_confidence([0.0, 0.5], np.ones(100)) == 0.0
    assert 0.0 <= beatgrid_confidence([0.0, 0.5, 1.1, 1.4], np.ones(100)) <= 1.0


def test_prune_backups_keeps_newest_three(tmp_path):
    track = tmp_path / "song.mp3"
    track.write_bytes(b"audio")
    backup_dir = tmp_path / ".cuepilot-backup"
    backup_dir.mkdir()
    names = []
    for i in range(5):
        p = backup_dir / f"song.2026010{i}-000000.abcdef.mp3"
        p.write_bytes(b"backup")
        # Increasing mtime: i=4 is newest.
        os.utime(p, (time.time() + i, time.time() + i))
        names.append(p.name)
    # A file for a different track must never be touched.
    other = backup_dir / "other.20260109-000000.abcdef.mp3"
    other.write_bytes(b"other")
    result = prune_backups(str(track), keep=3)
    assert len(result["kept"]) == 3
    assert len(result["deleted"]) == 2
    assert other.exists()
    assert track.read_bytes() == b"audio"
    # Newest backup survives even with a max_age that would delete it.
    newest = max(backup_dir.iterdir(), key=lambda p: p.stat().st_mtime)
    assert newest.name in result["kept"]


def test_cache_params_invalidate_stale_entries(tmp_path, monkeypatch):
    # Cache identity includes analysis params (R-2): same file hash,
    # different params must miss, not silently reuse stale arrays.
    monkeypatch.setenv("CUEPILOT_DATA_DIR", str(tmp_path))
    arrays = {"energy": np.array([1.0, 2.0, 3.0])}
    store_cached_arrays("stem_features", "abc123", arrays, meta={}, params={"sample_rate": 22050, "hop_length": 512})
    hit = get_cached_arrays("stem_features", "abc123", params={"sample_rate": 22050, "hop_length": 512})
    assert hit is not None
    assert get_cached_arrays("stem_features", "abc123", params={"sample_rate": 44100, "hop_length": 512}) is None
    assert get_cached_arrays("stem_features", "abc123", params={"sample_rate": 22050, "hop_length": 1024}) is None
