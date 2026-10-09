"""Unit tests for cue scoring."""

from core.beatgrid import BeatGrid
from core.models import Beat, GenreProfile, SectionType
from core.scoring.scorer import (
    compute_cue_score,
    genre_prior_score,
    phrase_boundary_score,
)


def _grid():
    return BeatGrid(
        bpm=120.0,
        beats=[Beat(time=t, index=i) for i, t in enumerate([0.0, 0.5, 1.0, 1.5, 2.0])],
        downbeat_times=[0.0, 1.0, 2.0],
        bars=[],
        phrases=[],
        confidence=0.9,
    )


def test_phrase_boundary_score():
    g = _grid()
    assert phrase_boundary_score(1.0, g) == 1.0
    assert phrase_boundary_score(0.5, g) == 0.0


def test_genre_prior_score():
    profile = GenreProfile(
        name="edm",
        bpm_range=(120.0, 150.0),
        section_order=[SectionType.INTRO, SectionType.BUILD, SectionType.DROP],
        slot_map={},
        cue_weights={},
    )
    assert genre_prior_score(SectionType.DROP, profile) == 1.0
    assert genre_prior_score(SectionType.CHORUS, profile) == 0.3


def test_compute_cue_score_total_bounded():
    g = _grid()
    profile = GenreProfile(
        name="edm",
        bpm_range=(120.0, 150.0),
        section_order=[SectionType.DROP],
        slot_map={},
        cue_weights={},
    )
    result = compute_cue_score(
        features=None,
        grid=g,
        profile=profile,
        section_type=SectionType.DROP,
        time=1.0,
    )
    assert 0.0 <= result.total <= 1.0
    assert set(result.components) == {
        "phrase_boundary",
        "energy_change",
        "spectral_change",
        "rhythm_change",
        "vocal_onset",
        "beat_alignment",
        "genre_prior",
    }


def test_weights_configurable():
    g = _grid()
    profile = GenreProfile(
        name="edm",
        bpm_range=(120.0, 150.0),
        section_order=[SectionType.DROP],
        slot_map={},
        cue_weights={"phrase_boundary": 1.0},
    )
    result = compute_cue_score(None, g, profile, SectionType.DROP, 1.0)
    assert result.total == 1.0
