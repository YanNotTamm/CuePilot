"""Tests for the genre routing layer (logic_new.md §6.3)."""

import os
import tempfile

from core.profiles.genre import get_profile
from core.structure import StructureDetector
from core.structure.router import (
    STRATEGY_ENERGY,
    STRATEGY_GRID,
    STRATEGY_VOCAL,
    confidence_cap,
    strategy_for_genre,
    strategy_for_profile,
)
from tests.fixtures.synth import write_synth_track
from core.analyzer import get_engine
from core.beatgrid import detect_beatgrid
from core.structure import detect_structure


def _synth_file() -> str:
    tmp = tempfile.mkdtemp()
    return write_synth_track(os.path.join(tmp, "t.wav"))


def test_strategy_for_genre_grid():
    assert strategy_for_genre("house") == STRATEGY_GRID
    assert strategy_for_genre("techno") == STRATEGY_GRID
    assert strategy_for_genre("progressive_house") == STRATEGY_GRID


def test_strategy_for_genre_energy():
    assert strategy_for_genre("drum_and_bass") == STRATEGY_ENERGY
    assert strategy_for_genre("dubstep") == STRATEGY_ENERGY
    assert strategy_for_genre("trap") == STRATEGY_ENERGY
    assert strategy_for_genre("baile") == STRATEGY_ENERGY
    assert strategy_for_genre("jersey_club") == STRATEGY_ENERGY


def test_strategy_for_genre_grid_new():
    assert strategy_for_genre("edm") == STRATEGY_GRID
    assert strategy_for_genre("indobounce") == STRATEGY_GRID


def test_strategy_for_genre_vocal():
    assert strategy_for_genre("hip_hop") == STRATEGY_VOCAL
    assert strategy_for_genre("rnb") == STRATEGY_VOCAL
    assert strategy_for_genre("open_format") == STRATEGY_VOCAL


def test_strategy_for_genre_local():
    assert strategy_for_genre("bassline_bounce") == "local_custom"
    assert strategy_for_genre("hipdut") == "local_custom"


def test_strategy_for_genre_unknown_fallback():
    # Unknown genre falls back deterministically.
    assert strategy_for_genre("weird") in (STRATEGY_GRID, STRATEGY_VOCAL)


def test_strategy_for_profile_uses_energy_hint():
    profile = get_profile("rnb")  # energy_profile="vocal_based"
    assert strategy_for_profile(profile) == STRATEGY_VOCAL


def test_strategy_for_profile_local_overrides_hint():
    # bassline_bounce / hipdut are Lokal-Custom even though their
    # energy_profile hint ("bass_bounce"/"vocal_based") names other strategies.
    assert strategy_for_profile(get_profile("bassline_bounce")) == "local_custom"
    assert strategy_for_profile(get_profile("hipdut")) == "local_custom"


def test_strategy_for_profile_energy_hint():
    assert strategy_for_profile(get_profile("jersey_club")) == STRATEGY_ENERGY
    assert strategy_for_profile(get_profile("baile")) == STRATEGY_ENERGY


def test_strategy_for_profile_none():
    assert strategy_for_profile(None) == STRATEGY_GRID


def test_confidence_cap_local():
    assert confidence_cap("local_custom") == 0.7
    assert confidence_cap(STRATEGY_GRID) == 0.99


def test_detector_accepts_strategy_override():
    file = _synth_file()
    engine = get_engine()
    audio = engine.load(file)
    features = engine.extract(audio)
    grid = detect_beatgrid(features)
    profile = get_profile("open_format")
    for strategy in (STRATEGY_GRID, STRATEGY_ENERGY, STRATEGY_VOCAL):
        sections = StructureDetector().detect(features, grid, profile, strategy=strategy)
        assert isinstance(sections, list) and len(sections) >= 1


def test_detector_auto_routes_from_profile():
    file = _synth_file()
    engine = get_engine()
    audio = engine.load(file)
    features = engine.extract(audio)
    grid = detect_beatgrid(features)
    sections = detect_structure(features, grid, get_profile("house"))
    assert len(sections) >= 1