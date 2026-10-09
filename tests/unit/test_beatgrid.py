"""Unit tests for beatgrid detection."""

import numpy as np
import pytest

from core.analyzer import LibrosaEngine
from core.beatgrid import detect_beatgrid, snap_to_beat, snap_to_downbeat
from tests.fixtures.synth import synth_track


@pytest.fixture(scope="module")
def grid():
    samples, sr = synth_track(bpm=128.0, seconds=96.0)
    from core.analyzer.base import AudioData

    audio = AudioData(samples=samples[np.newaxis, :], sample_rate=int(sr), channels=1, duration=96.0)
    features = LibrosaEngine().extract(audio)
    return detect_beatgrid(features)


def test_bpm_detected(grid):
    # Synthetic 128 BPM click track should be detected within tolerance.
    assert 120.0 <= grid.bpm <= 136.0, f"bpm {grid.bpm} outside tolerance"


def test_beats_present(grid):
    assert len(grid.beats) > 50


def test_downbeats_subset_of_beats(grid):
    down = set(round(t, 3) for t in grid.downbeat_times)
    beats = set(round(b.time, 3) for b in grid.beats)
    assert down <= beats


def test_bars_aligned_to_downbeats(grid):
    rounded = set(round(t, 3) for t in grid.downbeat_times)
    for bar in grid.bars:
        assert round(bar.downbeat_time, 3) in rounded


def test_phrases_span_track(grid):
    assert len(grid.phrases) >= 1
    for p in grid.phrases:
        assert p.end_time > p.start_time


def test_snap_to_beat():
    from core.beatgrid import BeatGrid
    from core.models import Beat

    g = BeatGrid(
        bpm=120.0,
        beats=[Beat(time=t, index=i) for i, t in enumerate([0.0, 0.5, 1.0, 1.5, 2.0])],
        downbeat_times=[0.0, 1.0, 2.0],
        bars=[],
        phrases=[],
        confidence=0.9,
    )
    assert snap_to_beat(0.49, g) == 0.5
    assert snap_to_beat(1.0, g) == 1.0
    assert snap_to_beat(0.1, g, before=True) == 0.0


def test_snap_to_downbeat():
    from core.beatgrid import BeatGrid
    from core.models import Beat

    g = BeatGrid(
        bpm=120.0,
        beats=[Beat(time=t, index=i) for i, t in enumerate([0.0, 0.5, 1.0, 1.5, 2.0])],
        downbeat_times=[0.0, 1.0, 2.0],
        bars=[],
        phrases=[],
        confidence=0.9,
    )
    assert snap_to_downbeat(0.9, g) == 1.0
