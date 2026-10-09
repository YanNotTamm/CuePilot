"""Tests for subdivision detection, tempo verification, sub-bass and cache."""

import os
import tempfile

import numpy as np
import pytest

from core.analyzer import LibrosaEngine
from core.analyzer.base import AudioData
from core.bands import reverse_bass_score, subbass_novelty, subbass_per_beat
from core.beatgrid import (
    STRAIGHT,
    classify_subdivision,
    detect_beatgrid,
    verify_kick_snare_tempo,
)
from core.structure.detector import StructureDetector
from core.structure.router import STRATEGY_ENERGY
from core.profiles.genre import get_profile
from tests.fixtures.synth import synth_track, write_synth_track


def _features_for(samples, sr=22050):
    audio = AudioData(samples=samples[np.newaxis, :], sample_rate=int(sr), channels=1, duration=samples.shape[-1] / float(sr))
    return LibrosaEngine().extract(audio)


def test_beatgrid_exposes_subdivision_and_tempo():
    samples, sr = synth_track(bpm=128.0, seconds=96.0)
    features = _features_for(samples, sr)
    grid = detect_beatgrid(features)
    assert grid.subdivision in ("straight", "triplet", "tresillo", "two_step", "five_kick", "unknown")
    assert 0.0 <= grid.subdivision_confidence <= 1.0
    assert "feel" in grid.tempo_verification
    assert grid.tempo_verification["feel"] in ("straight", "half_time", "double_time")


def test_classify_subdivision_straight_synth():
    samples, sr = synth_track(bpm=128.0, seconds=96.0)
    features = _features_for(samples, sr)
    grid = detect_beatgrid(features)
    sub, conf = classify_subdivision(features, grid.beat_times)
    # Four-on-the-floor synth with offbeat hats should read as straight.
    assert sub == STRAIGHT
    assert conf > 0.0


def test_classify_subdivision_hint_tiebreak():
    samples, sr = synth_track(bpm=128.0, seconds=96.0)
    features = _features_for(samples, sr)
    grid = detect_beatgrid(features)
    # Borderline cases must follow the genre hint toward documented grids.
    sub, _ = classify_subdivision(features, grid.beat_times, hint="baile")
    assert sub == "triplet"
    sub2, _ = classify_subdivision(features, grid.beat_times, hint="jersey_club")
    assert sub2 == "five_kick"
    sub3, _ = classify_subdivision(features, grid.beat_times, hint="reggaeton")
    assert sub3 == "tresillo"
    sub4, _ = classify_subdivision(features, grid.beat_times, hint="uk_garage")
    # 2-step is a *broken* kick: on a real four-on-the-floor track the hint
    # must not force it (Genre_pattern.md §717: don't use a 4x4 prior for
    # 2-step, but also don't mislabel actual 4x4 audio).
    assert sub4 in ("two_step", "straight")


def test_verify_kick_snare_straight_synth():
    samples, sr = synth_track(bpm=128.0, seconds=96.0)
    features = _features_for(samples, sr)
    grid = detect_beatgrid(features)
    result = verify_kick_snare_tempo(features, grid.beat_times, grid.bpm)
    assert result["feel"] == "straight"


def test_subbass_helpers_shapes():
    samples, sr = synth_track(bpm=128.0, seconds=96.0)
    features = _features_for(samples, sr)
    grid = detect_beatgrid(features)
    assert features.subbass_energy.size == features.onset_env.size
    per_beat = subbass_per_beat(features, grid)
    assert per_beat.size == len(grid.beats)
    nov = subbass_novelty(features, grid)
    assert nov.size == len(grid.beats)
    assert 0.0 <= float(np.max(nov)) <= 1.0


def test_reverse_bass_score_bounds():
    samples, sr = synth_track(bpm=128.0, seconds=96.0)
    features = _features_for(samples, sr)
    grid = detect_beatgrid(features)
    score = reverse_bass_score(features, grid)
    assert 0.0 <= score <= 1.0


def test_detector_energy_uses_subbass():
    samples, sr = synth_track(bpm=128.0, seconds=96.0)
    features = _features_for(samples, sr)
    grid = detect_beatgrid(features)
    profile = get_profile("dubstep")
    sections = StructureDetector().detect(features, grid, profile)
    assert len(sections) >= 1


def test_stem_features_roundtrip_cache():
    from core.stems.features import StemFeatures

    n = 32
    feats = StemFeatures(
        frame_times=np.linspace(0, 16, n),
        hop_length=512,
        sample_rate=22050,
        vocal_presence=np.full(n, 0.5),
        drum_energy=np.full(n, 0.5),
        bass_energy=np.full(n, 0.5),
        melodic_energy=np.full(n, 0.5),
    )
    arrays = feats.to_arrays()
    rebuilt = StemFeatures.from_arrays(arrays, hop_length=512, sample_rate=22050)
    assert np.allclose(rebuilt.vocal_presence, feats.vocal_presence)
    assert np.allclose(rebuilt.frame_times, feats.frame_times)
    assert rebuilt.hop_length == 512


def test_cache_store_and_load(tmp_path, monkeypatch):
    import core.cache as cache

    monkeypatch.setenv("CUEPILOT_DATA_DIR", str(tmp_path))
    h = cache.file_sha256  # sanity: function exists
    file_hash = "abcd1234"
    arrays = {"x": np.array([1.0, 2.0, 3.0])}
    cache.store_cached_arrays("stem_features", file_hash, arrays, meta={"hop_length": 512})
    loaded = cache.get_cached_arrays("stem_features", file_hash)
    assert loaded is not None
    out_arrays, meta = loaded
    assert np.allclose(out_arrays["x"], [1.0, 2.0, 3.0])
    assert meta["hop_length"] == 512
    # Different hash -> miss.
    assert cache.get_cached_arrays("stem_features", "zzzz") is None


def test_file_sha256_deterministic():
    import core.cache as cache

    with tempfile.NamedTemporaryFile("wb", suffix=".wav", delete=False) as f:
        f.write(b"0123456789" * 100)
        path = f.name
    try:
        h1 = cache.file_sha256(path)
        h2 = cache.file_sha256(path)
        assert h1 == h2
        assert len(h1) == 64
    finally:
        os.unlink(path)


def test_cache_version_invalidation(tmp_path, monkeypatch):
    import core.cache as cache

    monkeypatch.setenv("CUEPILOT_DATA_DIR", str(tmp_path))
    h = "hash1"
    cache.store_cached_arrays("stem_features", h, {"x": np.array([1.0])})
    # Simulate a bump of the analysis version -> stale entry invalidated.
    cache.ANALYSIS_VERSION = "9.9.9"
    try:
        assert cache.get_cached_arrays("stem_features", h) is None
    finally:
        cache.ANALYSIS_VERSION = "1.0.0"


def test_cueplan_provenance_export():
    from core.export import cue_plan_to_dict, validate_cue_plan_dict
    from core.models import Cue, CuePlan

    plan = CuePlan(
        schema_version="1.0",
        cues=[Cue(slot=1, type="INTRO", time=0.0, label="INTRO", confidence=0.9, color="green")],
        strategy_used="grid",
        genre_detected="house",
        genre_confidence=0.87,
    )
    payload = cue_plan_to_dict(plan, track={"path": "x.mp3", "artist": "", "title": "", "duration": 10.0})
    assert payload["strategyUsed"] == "grid"
    assert payload["genreDetected"] == "house"
    assert payload["genreConfidence"] == 0.87
    assert validate_cue_plan_dict(payload) == []


def test_cueplan_provenance_omitted_when_unset():
    from core.export import cue_plan_to_dict
    from core.models import CuePlan

    plan = CuePlan(schema_version="1.0", cues=[])
    payload = cue_plan_to_dict(plan)
    assert "strategyUsed" not in payload
    assert "genreDetected" not in payload