"""Tests for canonical export and schema validation."""

import json

import pytest

from core.export import build_track_block, cue_plan_to_dict, validate_cue_plan_dict
from core.models import Cue, CuePlan, Track
from tests.fixtures.synth import synth_track


def _sample_plan() -> CuePlan:
    return CuePlan(
        schema_version="1.0",
        cues=[
            Cue(slot=1, type="INTRO", time=0.0, label="INTRO", confidence=0.97, color="green"),
            Cue(slot=4, type="DROP", time=64.0, label="DROP", confidence=0.94, color="red"),
        ],
    )


def test_canonical_shape():
    plan = _sample_plan()
    payload = cue_plan_to_dict(plan)
    assert payload["schemaVersion"] == "1.0"
    assert len(payload["cues"]) == 2
    assert payload["cues"][1]["time"] == 64.0


def _sample_track() -> dict:
    return {
        "path": "D:/Music/track.mp3",
        "artist": "Artist",
        "title": "Track",
        "duration": 234.5,
        "bpm": 128,
        "key": "8A",
    }


def test_schema_valid_for_valid_plan():
    plan = _sample_plan()
    payload = cue_plan_to_dict(plan, track=_sample_track())
    errors = validate_cue_plan_dict(payload)
    assert errors == []


def test_schema_rejects_bad_plan():
    plan = _sample_plan()
    payload = cue_plan_to_dict(plan)
    payload["cues"][0]["slot"] = 99
    payload["cues"][0]["confidence"] = 2.0
    errors = validate_cue_plan_dict(payload)
    assert len(errors) > 0


def test_track_block_build():
    track = Track(id="1", path="D:\\Music\\track.mp3", filename="track.mp3", artist="A", title="T", duration=234.5)
    block = build_track_block(track)
    assert block["path"] == "D:/Music/track.mp3"
    assert block["artist"] == "A"
    assert block["title"] == "T"
    assert block["duration"] == 234.5


def test_synth_fixture_runs():
    samples, sr = synth_track(
        bpm=128.0,
        seconds=8.0,
        structure=[("INTRO", 0.25), ("DROP", 1.0), ("OUTRO", 0.2)],
    )
    assert samples.shape[0] == 8 * sr
