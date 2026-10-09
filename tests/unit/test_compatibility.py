"""Tests for Harmonic Mixing Advisor."""

from core.compatibility import (
    best_next_tracks,
    bpm_compatibility,
    key_compatibility,
    key_to_camelot,
    score_pair,
)
from core.compatibility.advisor import WEIGHTS


def _track(tid, key, bpm, energy=None, cues=None, duration=240.0):
    return {
        "id": tid,
        "key": key,
        "bpm": bpm,
        "duration": duration,
        "energyCurve": energy or [{"time": t / 10, "energy": 0.5} for t in range(10)],
        "cues": cues or [],
    }


# -- key -------------------------------------------------------------------


def test_key_to_camelot_basic():
    assert key_to_camelot("Am") == 15
    assert key_to_camelot("A minor") == 15
    assert key_to_camelot("C") == 16
    assert key_to_camelot("F#m") == 21
    assert key_to_camelot("") is None


def test_key_compatibility_relationships():
    assert key_compatibility("Am", "Am") == ("same", 1.0)
    label, score = key_compatibility("Am", "C")
    assert label == "relative" and score == 0.95
    label, score = key_compatibility("Am", "Em")
    assert label == "adjacent" and score == 0.85
    label, score = key_compatibility("Am", "G")
    assert label == "energy_boost" and score == 0.8
    label, score = key_compatibility("Am", "F#m")
    assert label == "unrelated" and score == 0.0


def test_key_compatibility_unknown():
    label, score = key_compatibility("", "Am")
    assert label == "unknown" and score == 0.5


# -- bpm -------------------------------------------------------------------


def test_bpm_compatibility():
    assert bpm_compatibility(128, 128)[0] == "exact"
    assert bpm_compatibility(128, 64)[0] == "half_time"
    assert bpm_compatibility(128, 256)[0] == "double_time"
    assert bpm_compatibility(128, 134)[0] == "pitch"
    assert bpm_compatibility(128, 150)[0] == "incompatible"
    assert bpm_compatibility(0, 128)[0] == "unknown"


def test_bpm_exact_score():
    label, score, delta = bpm_compatibility(128, 128)
    assert score == 1.0 and delta == 0.0


# -- pair scoring ----------------------------------------------------------


def test_score_pair_best_match():
    a = _track("A", "Am", 128)
    b = _track("B", "Am", 128)
    r = score_pair(a, b)
    assert r.score >= 0.9
    assert r.key_compatibility == "same"
    assert r.bpm_compatibility == "exact"
    assert r.suggested_transition is not None


def test_score_pair_bad_match():
    a = _track("A", "Am", 128)
    b = _track("B", "F#m", 170)
    r = score_pair(a, b)
    assert r.score < 0.5


def test_score_pair_suggests_cue_pairing():
    a = _track(
        "A", "Am", 128,
        cues=[{"slot": 1, "type": "OUTRO", "time": 230.0, "confidence": 0.9}],
    )
    b = _track(
        "B", "C", 128,
        cues=[{"slot": 1, "type": "INTRO", "time": 2.0, "confidence": 0.9}],
    )
    r = score_pair(a, b)
    assert r.suggested_transition == {"fromCue": "OUTRO", "toCue": "INTRO"}


def test_best_next_tracks_ranks_and_filters():
    lib = [
        _track("A", "Am", 128),
        _track("B", "C", 128),    # relative — best
        _track("C", "F#m", 170),  # incompatible
        _track("D", "Em", 120),   # adjacent
    ]
    best = best_next_tracks(lib[0], lib, top_n=5, min_score=0.5)
    ids = [r.track_id for r in best]
    assert ids[0] == "B"
    assert "C" not in ids
    assert len(best) >= 1


def test_weights_sum():
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9