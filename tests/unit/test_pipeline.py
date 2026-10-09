"""Integration-style tests for the full pipeline."""

import numpy as np
import pytest

from core.analyzer import LibrosaEngine
from core.analyzer.base import AudioData
from core.beatgrid import detect_beatgrid
from core.cues import generate_cueplan
from core.models import SectionType
from core.pipeline import _apply_stem_selection, _stem_curves
from core.profiles.genre import get_profile
from core.stems.features import StemFeatures
from core.structure import detect_structure
from tests.fixtures.synth import synth_track


@pytest.fixture(scope="module")
def features():
    samples, sr = synth_track(bpm=128.0, seconds=96.0)
    audio = AudioData(samples=samples[np.newaxis, :], sample_rate=int(sr), channels=1, duration=96.0)
    return LibrosaEngine().extract(audio)


@pytest.fixture(scope="module")
def grid(features):
    return detect_beatgrid(features)


def test_structure_has_sections(features, grid):
    profile = get_profile("open_format")
    sections = detect_structure(features, grid, profile)
    assert len(sections) >= 2
    for sec in sections:
        assert sec.start >= 0.0
        assert sec.end > sec.start
        assert 0.0 <= sec.confidence <= 1.0
        assert isinstance(sec.type, SectionType)


def test_structure_covers_track(features, grid):
    profile = get_profile("open_format")
    sections = detect_structure(features, grid, profile)
    assert sections[0].start == pytest.approx(0.0, abs=0.5)
    assert sections[-1].end >= 90.0


def test_cue_plan_generation(features, grid):
    profile = get_profile("open_format")
    sections = detect_structure(features, grid, profile)
    plan = generate_cueplan(None, features, grid, sections, profile)
    assert len(plan.cues) > 0
    slots = [c.slot for c in plan.cues]
    assert 1 <= min(slots) <= 8
    assert max(slots) <= 8
    for cue in plan.cues:
        assert cue.time >= 0.0
        assert 0.0 <= cue.confidence <= 1.0
        assert len(cue.reason) > 0


def _stem_feats() -> StemFeatures:
    n = 64
    import numpy as np

    return StemFeatures(
        frame_times=np.linspace(0.0, 32.0, n),
        hop_length=512,
        sample_rate=22050,
        vocal_presence=np.full(n, 0.5),
        vocal_energy=np.full(n, 0.5),
        vocal_density=np.full(n, 0.5),
        vocal_onset=np.full(n, 0.5),
        drum_energy=np.full(n, 0.5),
        kick_activity=np.full(n, 0.5),
        snare_activity=np.full(n, 0.5),
        hat_activity=np.full(n, 0.5),
        percussion_density=np.full(n, 0.5),
        onset_density=np.full(n, 0.5),
        bass_energy=np.full(n, 0.5),
        bass_presence=np.full(n, 0.5),
        bass_onset=np.full(n, 0.5),
        melodic_energy=np.full(n, 0.5),
        harmonic_density=np.full(n, 0.5),
        backend="test",
        status="ok",
    )


def test_apply_stem_selection_disables_chosen_stems():
    import numpy as np

    feats = _stem_feats()
    out = _apply_stem_selection(feats, ["vocal", "bass"])
    assert np.all(out.drum_energy == 0.0)
    assert np.all(out.melodic_energy == 0.0)
    assert np.all(out.vocal_energy == 0.5)
    assert np.all(out.bass_energy == 0.5)


def test_apply_stem_selection_all_keeps_unchanged():
    import numpy as np

    feats = _stem_feats()
    out = _apply_stem_selection(feats, ["vocal", "drums", "bass", "other"])
    assert np.all(out.drum_energy == 0.5)


def test_apply_stem_selection_none_keeps_unchanged():
    import numpy as np

    feats = _stem_feats()
    out = _apply_stem_selection(feats, None)
    assert np.all(out.drum_energy == 0.5)


def test_stem_curves_shapes():
    curves = _stem_curves(_stem_feats())
    assert "vocal" in curves and "drums" in curves
    assert "bass" in curves and "other" in curves
    assert len(curves["times"]) == len(curves["vocal"]) == len(curves["drums"])
    assert len(curves["vocal"]) > 0
    assert 0.0 <= curves["vocal"][0]["v"] <= 1.0


def test_stem_curves_empty():
    curves = _stem_curves(StemFeatures())
    assert curves["times"] == []
    assert curves["vocal"] == []
