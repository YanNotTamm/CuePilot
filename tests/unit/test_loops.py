"""Tests for Loop-Region Intelligence."""

import numpy as np

from core.beatgrid import BeatGrid
from core.loops import LOOP_BARS, detect_loops
from core.loops.detector import SEAMLESS_MIN, _loop_duration, _seamless_score
from core.models import Bar, Beat, EnergyPoint, Phrase


def _beatgrid(bpm: float = 128.0, bars: int = 32) -> BeatGrid:
    beat_sec = 60.0 / bpm
    beats: list[Beat] = []
    downbeats: list[float] = []
    bar_list: list[Bar] = []
    phrase_list: list[Phrase] = []
    t = 0.0
    for bi in range(bars * 4):
        beats.append(Beat(time=t, is_downbeat=(bi % 4 == 0), index=bi))
        if bi % 4 == 0:
            downbeats.append(t)
            bar_list.append(Bar(index=bi // 4, downbeat_time=t, beat_count=4))
        if bi % 16 == 0:
            phrase_list.append(Phrase(index=bi // 16, start_time=t, end_time=t + 16 * beat_sec, bar_count=4))
        t += beat_sec
    return BeatGrid(
        bpm=bpm,
        beats=beats,
        downbeat_times=downbeats,
        bars=bar_list,
        phrases=phrase_list,
        confidence=0.95,
    )


def _energy_curve(duration: float, seconds_per_point: float = 0.5) -> list[EnergyPoint]:
    points = []
    t = 0.0
    while t <= duration:
        # Smooth energy that repeats every 8 bars so seams are clean.
        cycle = 60.0 / 128.0 * 16  # 8 bars
        e = 0.4 + 0.3 * (0.5 + 0.5 * np.sin(2 * np.pi * t / cycle))
        points.append(EnergyPoint(time=t, energy=float(e)))
        t += seconds_per_point
    return points


def test_loop_duration_calculation():
    assert _loop_duration(120.0, 4) == 8.0
    assert _loop_duration(120.0, 8) == 16.0
    assert _loop_duration(0, 4) == 0.0


def test_seamless_score_bounds():
    curve = _energy_curve(40.0)
    score = _seamless_score(curve, 8.0, 16.0, 8.0)
    assert 0.0 <= score <= 1.0


def test_detect_loops_returns_candidates():
    grid = _beatgrid()
    duration = grid.beats[-1].time
    loops = detect_loops(grid, _energy_curve(duration), duration=duration)
    assert len(loops) > 0
    for loop in loops:
        assert loop.bars in LOOP_BARS
        assert loop.start >= 0.0
        assert loop.end > loop.start
        assert loop.end <= duration + 1e-6
        assert 0.0 <= loop.seamless_score <= 1.0
        assert loop.use_case in ("intro-extension", "outro-extension", "drop-loop", "build-extension")


def test_detect_loops_dedupes_and_caps():
    grid = _beatgrid()
    duration = grid.beats[-1].time
    loops = detect_loops(grid, _energy_curve(duration), duration=duration, max_results=3)
    assert len(loops) <= 3
    spans = [(l.start, l.end) for l in loops]
    assert len(set(spans)) == len(spans)


def test_detect_loops_no_phrases():
    grid = _beatgrid()
    grid.phrases = []
    assert detect_loops(grid, [], duration=30.0) == []


def test_loop_to_dict_serializable():
    grid = _beatgrid()
    duration = grid.beats[-1].time
    loops = detect_loops(grid, _energy_curve(duration), duration=duration)
    assert loops
    data = loops[0].to_dict()
    assert data["type"] == "SAVED_LOOP"
    assert data["bars"] in LOOP_BARS
    assert isinstance(data["start"], float)
    assert "seamlessScore" in data


def test_seamless_min_threshold_used():
    assert 0.0 < SEAMLESS_MIN < 1.0