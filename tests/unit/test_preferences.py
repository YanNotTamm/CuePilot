"""Tests for DJ preference profiles and adaptive scoring."""

import json

import pytest

from core.learn import (
    PreferenceProfile,
    PreferenceStore,
    adaptive_weights,
    apply_preference,
    get_preference_store,
    slot_strategy_for,
)
from core.learn.preferences import ADAPT_MAX_DELTA
from core.learn.store import MIN_SAMPLES, LearningStore

BASE_WEIGHTS = {"vocal_onset": 0.5, "energy_change": 0.3, "spectral_change": 0.2}


@pytest.fixture()
def pstore(tmp_path):
    return PreferenceStore(tmp_path / "preferences.json")


@pytest.fixture()
def lstore(tmp_path):
    db = LearningStore(tmp_path / "learn.db")
    yield db
    db.clear()


def _teach_slot_as_vocal(store, n=MIN_SAMPLES):
    for i in range(n):
        store.record_cue_feedback(fr"D:\m\t{i}.mp3", "house", 2, 44.0, 110.0, "VOCAL", "VOCAL")


def test_preference_store_roundtrip(pstore):
    profile = PreferenceProfile(name="test_dj", weights={"house": {"vocal_onset": 0.9}})
    pstore.save(profile)
    loaded = pstore.get("test_dj")
    assert loaded is not None
    assert loaded.weights["house"]["vocal_onset"] == 0.9
    assert pstore.delete("test_dj") is True
    assert pstore.get("test_dj") is None


def test_preference_store_list(pstore):
    pstore.save(PreferenceProfile(name="a", weights={}))
    pstore.save(PreferenceProfile(name="b", weights={}))
    names = {p.name for p in pstore.list_profiles()}
    assert names == {"a", "b"}


def test_adaptive_weights_no_patterns(lstore):
    weights = adaptive_weights("house", base_weights=dict(BASE_WEIGHTS), store=lstore)
    assert weights == BASE_WEIGHTS


def test_adaptive_weights_nudges_toward_vocal(lstore):
    _teach_slot_as_vocal(lstore)
    weights = adaptive_weights("house", base_weights=dict(BASE_WEIGHTS), store=lstore)
    assert weights["vocal_onset"] > BASE_WEIGHTS["vocal_onset"]
    assert weights["spectral_change"] < BASE_WEIGHTS["spectral_change"]
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_adaptive_delta_bounded(lstore):
    _teach_slot_as_vocal(lstore)
    weights = adaptive_weights("house", base_weights=dict(BASE_WEIGHTS), store=lstore)
    assert abs(weights["vocal_onset"] - BASE_WEIGHTS["vocal_onset"]) <= ADAPT_MAX_DELTA


def test_adaptive_ignores_untrusted(lstore):
    # single untrusted sample -> no pressure
    lstore.record_cue_feedback(r"D:\m\t1.mp3", "house", 2, 44.0, 110.0, "VOCAL", "VOCAL")
    weights = adaptive_weights("house", base_weights=dict(BASE_WEIGHTS), store=lstore)
    assert weights == BASE_WEIGHTS


def test_apply_preference_overrides(lstore):
    profile = PreferenceProfile(name="p", weights={"house": {"vocal_onset": 0.9, "energy_change": 0.05}})
    blended = adaptive_weights("house", base_weights=dict(BASE_WEIGHTS), store=lstore)
    out = apply_preference(profile, "house", blended)
    assert out["vocal_onset"] == 0.9
    assert out["energy_change"] == 0.05
    assert out["spectral_change"] == BASE_WEIGHTS["spectral_change"]


def test_slot_strategy_override():
    base = {1: "INTRO", 2: "DROP", 3: "VOCAL"}
    profile = PreferenceProfile(name="p", slot_strategy={"house": {3: "BREAKDOWN"}})
    out = slot_strategy_for(profile, "house", base)
    assert out[3] == "BREAKDOWN"
    assert out[1] == "INTRO"
    assert slot_strategy_for(None, "house", base) == base


def test_get_preference_store_singleton(tmp_path, monkeypatch):
    monkeypatch.setenv("CUEPILOT_DATA_DIR", str(tmp_path))
    s1 = get_preference_store()
    s2 = get_preference_store()
    assert s1 is s2


def test_preferences_file_json(tmp_path):
    pstore = PreferenceStore(tmp_path / "preferences.json")
    pstore.save(PreferenceProfile(name="dj", weights={"tech": {"energy_change": 0.8}}))
    raw = json.loads((tmp_path / "preferences.json").read_text(encoding="utf-8"))
    assert "dj" in raw["profiles"]