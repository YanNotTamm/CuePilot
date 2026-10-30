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
    # Energy boost (standard definition): one Camelot number higher,
    # same letter — 8A (Am) -> 9A (Em).
    label, score = key_compatibility("Am", "Em")
    assert label == "energy_boost" and score == 0.8
    # One number lower, same letter, is adjacent, not a boost.
    label, score = key_compatibility("Am", "Dm")
    assert label == "adjacent" and score == 0.85
    label, score = key_compatibility("Am", "F#m")
    assert label == "unrelated" and score == 0.0


def test_key_to_camelot_accepts_camelot_codes():
    # Keys already detected as Camelot codes (e.g. by detect_key) must parse.
    assert key_to_camelot("8A") == 15
    assert key_to_camelot("8a") == 15
    assert key_to_camelot("12B") == 24
    assert key_to_camelot("1A") == 1
    # Note-name and Camelot forms agree.
    assert key_to_camelot("8A") == key_to_camelot("Am")
    assert key_to_camelot("8B") == key_to_camelot("C")


def test_key_compatibility_circular_wrap():
    # Circular wheel: 12B and 1A are neighbours (distance 1, not 23).
    label, score = key_compatibility("12B", "1A")
    assert label == "adjacent" and score == 0.85
    label, score = key_compatibility("1A", "12B")
    assert label == "adjacent" and score == 0.85
    # Energy boost also wraps: 12A -> 1A is one number higher.
    label, score = key_compatibility("12A", "1A")
    assert label == "energy_boost" and score == 0.8


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
    # No OUTRO/INTRO cues on either track, so no transition may be claimed.
    assert r.suggested_transition is None


def test_score_pair_no_transition_without_both_cues():
    # Only an OUTRO on A (no INTRO on B): previously both branches of the
    # if/else returned the same dict, claiming a transition anyway.
    a = _track("A", "Am", 128, cues=[{"slot": 1, "type": "OUTRO", "time": 230.0}])
    b = _track("B", "C", 128, cues=[])
    assert score_pair(a, b).suggested_transition is None
    a2 = _track("A", "Am", 128, cues=[])
    b2 = _track("B", "C", 128, cues=[{"slot": 1, "type": "INTRO", "time": 2.0}])
    assert score_pair(a2, b2).suggested_transition is None


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