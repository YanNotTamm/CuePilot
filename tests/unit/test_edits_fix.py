"""Regression tests for the Edits/Learn fix batch (S-1).

Covers: corrupt sidecars are reported (not silently treated as "no
edits"), slot 1..8 validation and time clamping in apply_edits_to_cues
with notifications, and the beatgrid snap helper used by learned
patterns (see also test_learn.py).
"""

import json

from core.edits import (
    CueEdit,
    EditsFile,
    apply_edits_to_cues,
    load_edits,
    load_edits_status,
)
from core.learn import snap_to_nearest_grid
from core.models import Cue, CueType


def _cue(slot, time):
    return Cue(slot=slot, type=CueType.DROP, time=time, label="D", confidence=0.8, color="#fff")


def test_load_edits_corrupt_is_reported_not_none(tmp_path):
    track = tmp_path / "track.mp3"
    track.write_bytes(b"fake")
    sidecar = tmp_path / "track.mp3.cuepilot.json"
    sidecar.write_text("{not valid json", encoding="utf-8")
    edits = load_edits(str(track))
    assert edits is not None  # must NOT look like "no edits"
    assert edits.corrupt is True
    assert edits.cues == []
    assert edits.load_error
    status = load_edits_status(str(track))
    assert status["status"] == "corrupt"
    assert status["error"]


def test_load_edits_missing_is_still_none(tmp_path):
    track = tmp_path / "track.mp3"
    assert load_edits(str(track)) is None
    assert load_edits_status(str(track))["status"] == "missing"


def test_apply_edits_rejects_bad_slot_with_notification():
    cues = [_cue(1, 10.0)]
    edits = EditsFile(cues=[CueEdit(slot=9, time=20.0, locked=True, label="BAD")])
    notes: list = []
    out = apply_edits_to_cues(cues, edits, duration=100.0, notifications=notes)
    # Slot 9 is outside schema 1..8: edit ignored, auto cue untouched.
    assert out[0].time == 10.0
    assert any(n["action"] == "rejected" and n["slot"] == 9 for n in notes)


def test_apply_edits_clamps_time_beyond_duration_with_notification():
    cues = [_cue(2, 10.0)]
    edits = EditsFile(cues=[CueEdit(slot=2, time=999.0, locked=True)])
    notes: list = []
    out = apply_edits_to_cues(cues, edits, duration=100.0, notifications=notes)
    assert out[0].time == 100.0  # clamped, never exported beyond duration
    assert any(
        n["action"] == "clamped" and n["originalTime"] == 999.0 and n["clampedTime"] == 100.0
        for n in notes
    )


def test_apply_edits_rejects_extra_cue_with_bad_slot():
    cues = [_cue(1, 10.0)]
    edits = EditsFile(cues=[CueEdit(slot=0, time=5.0)])
    notes: list = []
    out = apply_edits_to_cues(cues, edits, duration=100.0, notifications=notes)
    assert [c.slot for c in out] == [1]  # no slot-0 cue smuggled in
    assert any(n["action"] == "rejected" for n in notes)


def test_apply_edits_corrupt_sidecar_applies_nothing():
    cues = [_cue(1, 10.0)]
    edits = EditsFile(cues=[CueEdit(slot=1, time=50.0, locked=True)], corrupt=True, load_error="boom")
    notes: list = []
    out = apply_edits_to_cues(cues, edits, duration=100.0, notifications=notes)
    assert out[0].time == 10.0
    assert any(n["action"] == "rejected" for n in notes)


def test_snap_to_nearest_grid_edges():
    assert snap_to_nearest_grid(0.1, beat_times=[1.0, 2.0, 3.0]) == 1.0
    assert snap_to_nearest_grid(99.0, beat_times=[1.0, 2.0, 3.0]) == 3.0
    assert snap_to_nearest_grid(1.6, beat_times=[1.0, 2.0, 3.0]) == 2.0
    assert snap_to_nearest_grid(1.4, beat_times=[1.0, 2.0, 3.0]) == 1.0
    assert snap_to_nearest_grid(5.0, beat_times=[], downbeat_times=[]) == 5.0
