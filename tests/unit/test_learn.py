"""Tests for the learning store and learned-pattern application."""

import os

import pytest

from core.learn import apply_learned_patterns, get_store
from core.learn.store import MIN_SAMPLES, LearningStore
from core.models import Cue, CueType


@pytest.fixture()
def store(tmp_path):
    db = LearningStore(tmp_path / "learn.db")
    yield db
    db.clear()


def _cue(slot: int, time: float) -> Cue:
    return Cue(
        slot=slot,
        type=CueType.DROP,
        time=time,
        label="DROP",
        confidence=0.8,
        color="#E74C3C",
        source="dsp",
    )


def test_upsert_same_track_replaces(store):
    store.record_cue_feedback(r"D:\m\t1.mp3", "baile", 5, 33.0, 110.0, "DROP", "DROP")
    store.record_cue_feedback(r"D:\m\t1.mp3", "baile", 5, 40.0, 110.0, "DROP", "DROP")
    stats = store.stats()
    assert stats["total"] == 1
    assert stats["tracks"] == 1


def test_patterns_trusted_only_after_min_samples(store):
    for i in range(MIN_SAMPLES):
        store.record_cue_feedback(fr"D:\m\t{i}.mp3", "baile", 5, 33.0, 110.0, "DROP", "DROP")
    patterns = store.patterns_for("baile")
    assert patterns[0]["samples"] == MIN_SAMPLES
    assert patterns[0]["trusted"] is True
    assert patterns[0]["slot"] == 5
    assert abs(patterns[0]["avgPosition"] - 0.3) < 1e-6


def test_untrusted_pattern_not_applied(store):
    # single sample -> below MIN_SAMPLES -> cue unchanged
    store.record_cue_feedback(r"D:\m\t1.mp3", "baile", 5, 33.0, 110.0, "DROP", "DROP")
    cues = [_cue(5, 28.0)]
    apply_learned_patterns(cues, 110.0, "baile", store)
    assert cues[0].time == 28.0


def test_learned_pattern_moves_cue(store):
    for i in range(MIN_SAMPLES):
        store.record_cue_feedback(fr"D:\m\t{i}.mp3", "baile", 5, 33.0, 110.0, "DROP", "DROP")
    cues = [_cue(5, 28.0)]
    apply_learned_patterns(cues, 110.0, "baile", store)
    # avg position 0.3 * 110 = 33.0s; shift clamped at 0.15*110 = 16.5s (ok)
    assert abs(cues[0].time - 33.0) < 1e-6
    assert any("learned" in r for r in cues[0].reason)


def test_shift_is_clamped(store):
    # learned 0.95 of track, auto cue at 0.1 -> clamp prevents wild jump
    for i in range(MIN_SAMPLES):
        store.record_cue_feedback(fr"D:\m\t{i}.mp3", "baile", 5, 104.5, 110.0, "DROP", "DROP")
    cues = [_cue(5, 11.0)]
    apply_learned_patterns(cues, 110.0, "baile", store)
    # max allowed shift = 0.15 * 110 = 16.5s -> 11.0 -> 27.5
    assert abs(cues[0].time - 27.5) < 1e-6


def test_locked_and_user_cues_not_moved(store):
    for i in range(MIN_SAMPLES):
        store.record_cue_feedback(fr"D:\m\t{i}.mp3", "baile", 5, 33.0, 110.0, "DROP", "DROP")
    locked = _cue(5, 28.0)
    locked.locked = True
    user = _cue(4, 50.0)
    user.source = "user"
    apply_learned_patterns([locked, user], 110.0, "baile", store)
    assert locked.time == 28.0
    assert user.time == 50.0


def test_default_store_singleton(tmp_path, monkeypatch):
    monkeypatch.setenv("CUEPILOT_DATA_DIR", str(tmp_path))
    s1 = get_store()
    s2 = get_store()
    assert s1 is s2
    s1.clear()
