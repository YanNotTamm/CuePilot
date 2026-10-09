"""Tests for Intelligent Set Planner."""

from core.planner import plan_set
from core.planner.planner import _target_energy


def _track(tid, bpm=128.0, key="Am", duration=240.0, energy=0.5, genre="Tech House"):
    return {
        "id": tid,
        "bpm": bpm,
        "key": key,
        "duration": duration,
        "energy": energy,
        "genre": genre,
    }


_LIB = [
    _track("t1", bpm=128, key="Am", energy=0.8),
    _track("t2", bpm=124, key="C", energy=0.3),
    _track("t3", bpm=130, key="G", energy=0.9),
    _track("t4", bpm=126, key="Em", energy=0.5),
]


def test_plan_set_produces_ordered_draft():
    draft = plan_set(_LIB, target_minutes=10.0)
    assert len(draft.tracks) >= 1
    positions = [t.position for t in draft.tracks]
    assert positions == sorted(positions)
    assert draft.total_duration > 0
    assert len(draft.energy_arc) == len(draft.tracks)


def test_plan_set_honors_must_include():
    draft = plan_set(_LIB, target_minutes=5.0, must_include=["t3"])
    ids = [t.track_id for t in draft.tracks]
    assert "t3" in ids


def test_plan_set_genre_pool_filter():
    lib = _LIB + [_track("x1", genre="Hip-Hop")]
    draft = plan_set(lib, target_minutes=5.0, genre_pool=["Tech House"])
    assert all(t.track_id != "x1" for t in draft.tracks)


def test_plan_set_starts_warm():
    draft = plan_set(_LIB, target_minutes=10.0)
    assert draft.tracks[0].track_id == "t2"  # lowest energy opener


def test_plan_set_empty_library():
    draft = plan_set([], target_minutes=5.0)
    assert draft.tracks == []
    assert draft.total_duration == 0.0


def test_plan_set_transitions_present():
    draft = plan_set(_LIB, target_minutes=15.0)
    for t in draft.tracks[1:]:
        assert t.transition is not None


def test_target_energy_arc_shape():
    assert _target_energy(0.0, ["warm-up", "peak", "cool-down"]) < 0.5
    assert _target_energy(0.5, ["warm-up", "peak", "cool-down"]) > 0.5
    assert _target_energy(0.9, ["warm-up", "peak", "cool-down"]) < 0.5


def test_plan_set_to_dict():
    draft = plan_set(_LIB, target_minutes=5.0)
    data = draft.to_dict()
    assert "tracks" in data and "totalDuration" in data and "energyArc" in data
    assert isinstance(data["tracks"], list)