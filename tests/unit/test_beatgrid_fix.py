"""Tests for the BEATGRID/CUE/GENRE/LOOP fixes (T-4, T-5, S-2, R-1, R-4).

Runnable without librosa or pytest's audio fixtures: every input here is
synthetic, and a bare librosa stub is installed only when the real
package is absent (the core.analyzer package imports librosa at top
level, but none of these tests call it at runtime).
"""

import sys
from types import ModuleType

try:
    import librosa  # noqa: F401
except ImportError:  # pragma: no cover - only on librosa-less machines
    sys.modules.setdefault("librosa", ModuleType("librosa"))

import numpy as np

from core.analyzer.base import AnalysisFeatures
from core.beatgrid import BeatGrid
from core.beatgrid.detector import beatgrid_confidence
from core.beatgrid.subdivision import HINT_MAP, classify_subdivision
from core.loops.detector import _loop_duration, _loop_end_from_grid, detect_loops
from core.models import (
    Bar,
    Beat,
    Cue,
    CueType,
    EnergyPoint,
    Phrase,
    Section,
    SectionType,
)
from core.quality.readiness import _correction_stability, compute_readiness
from core.structure.detector import StructureDetector
from core.structure.router import STRATEGY_GRID, STRATEGY_LOCAL, confidence_cap


def _features(onset: np.ndarray) -> AnalysisFeatures:
    n = len(onset)
    return AnalysisFeatures(
        hop_length=512,
        sample_rate=22050,
        frame_times=np.arange(n) * 0.023,
        onset_env=onset,
        rms=np.full(n, 0.5),
        spectral_flux=np.full(n, 0.5),
        harmonic=np.full(n, 0.5),
        percussive=np.full(n, 0.5),
    )


# -- T-5: beatgrid confidence -------------------------------------------

def test_beatgrid_confidence_regular_grid_scores_one():
    beats = np.arange(0.0, 20.0, 0.5)
    assert beatgrid_confidence(beats, np.ones(len(beats)), onset_max=1.0) == 1.0


def test_beatgrid_confidence_jitter_scores_below_regular():
    regular = beatgrid_confidence(np.arange(0.0, 20.0, 0.5),
                                  np.ones(40), onset_max=1.0)
    jittered_times = np.cumsum([0.0] + [0.4, 0.6] * 20)[:40]
    jittered = beatgrid_confidence(jittered_times, np.ones(40), onset_max=1.0)
    assert 0.0 < jittered < regular


def test_beatgrid_confidence_weak_onsets_lower_score():
    beats = np.arange(0.0, 20.0, 0.5)
    strong = beatgrid_confidence(beats, np.ones(len(beats)), onset_max=1.0)
    weak = beatgrid_confidence(beats, np.full(len(beats), 0.2), onset_max=1.0)
    assert weak < strong
    assert abs(weak - (0.7 * 1.0 + 0.3 * 0.2)) < 1e-9  # documented 0.7/0.3 blend


def test_beatgrid_confidence_too_few_beats_is_zero():
    assert beatgrid_confidence([0.0, 0.5]) == 0.0
    assert beatgrid_confidence([]) == 0.0


def test_beatgrid_confidence_without_onsets_is_consistency_only():
    jittered_times = np.cumsum([0.0] + [0.4, 0.6] * 20)[:40]
    # No onset data: consistency alone (CV of 0.4/0.6 intervals ~ 0.2).
    assert abs(beatgrid_confidence(jittered_times) - 0.8) < 0.02


# -- T-4: meter / tempo scope is explicit -------------------------------

def test_beatgrid_scope_fields_default_to_assumed_global():
    grid = BeatGrid(bpm=120.0, beats=[], downbeat_times=[], bars=[],
                    phrases=[], confidence=0.5)
    assert grid.meter_detected is False
    assert grid.tempo_model == "global-single-tempo"
    assert "4/4" in grid.meter and "not detected" in grid.meter


# -- R-1: subdivision hints ----------------------------------------------

def test_hint_map_covers_indobounce_and_bassline_bounce():
    assert HINT_MAP["indobounce"] == "straight"
    assert HINT_MAP["bassline_bounce"] == "straight"
    assert "weird_genre" not in HINT_MAP


def test_hint_not_forced_with_zero_evidence():
    # Flat zero onset envelope + baile hint: must NOT return triplet
    # with a floored 0.35 confidence (the old behavior).
    feats = _features(np.zeros(1000))
    sub, conf = classify_subdivision(feats, np.arange(0.0, 20.0, 0.5),
                                     hint="baile")
    assert sub == "unknown"
    assert conf < 0.35


def test_straight_hint_with_straight_evidence():
    # Onsets exactly on beats, weak elsewhere -> straight, for both
    # newly mapped straight-grid profiles.
    frame_times = np.arange(0, 2000) * 0.01
    onset = np.full(2000, 0.1)
    for t in np.arange(0.0, 20.0, 0.5):
        onset[int(round(t / 0.01))] = 1.0
    feats = AnalysisFeatures(
        hop_length=512, sample_rate=22050, frame_times=frame_times,
        onset_env=onset, rms=np.full(2000, 0.5),
        spectral_flux=np.full(2000, 0.5), harmonic=np.full(2000, 0.5),
        percussive=np.full(2000, 0.5),
    )
    for hint in ("indobounce", "bassline_bounce"):
        sub, conf = classify_subdivision(feats, np.arange(0.0, 20.0, 0.5),
                                         hint=hint)
        assert sub == "straight" and conf > 0.0


# -- S-2: loop duration + grid downbeat ends ------------------------------

def test_loop_duration_uses_beats_per_bar_not_phrase_length():
    assert _loop_duration(120.0, 4) == 8.0
    assert _loop_duration(120.0, 8) == 16.0
    assert _loop_duration(120.0, 4, beats_per_bar=3) == 6.0
    assert _loop_duration(0, 4) == 0.0


def _loop_grid(downbeats, phrases):
    beats = [Beat(time=t, index=i) for i, t in enumerate(
        np.arange(0.0, 64.0, 0.5))]
    return BeatGrid(
        bpm=120.0, beats=beats, downbeat_times=downbeats,
        bars=[Bar(index=i, downbeat_time=t, beat_count=4)
              for i, t in enumerate(downbeats)],
        phrases=phrases, confidence=0.9,
    )


def _flat_curve(duration: float) -> list[EnergyPoint]:
    return [EnergyPoint(time=t, energy=0.5)
            for t in np.arange(0.0, duration + 0.5, 0.5)]


def test_loop_end_comes_from_grid_downbeat_not_nominal_duration():
    # Drifted downbeats: nominal 4-bar loop at 120 BPM is 8.0s, but the
    # grid's 4th downbeat after 0 is at 8.2s -- the loop must end there.
    downbeats = [0.0, 2.1, 4.1, 6.0, 8.2, 10.3, 12.4, 14.4, 16.5]
    grid = _loop_grid(downbeats, [Phrase(index=0, start_time=0.0,
                                         end_time=8.2, bar_count=4)])
    loops = detect_loops(grid, _flat_curve(64.0), duration=64.0)
    four_bar = [l for l in loops if l.bars == 4 and l.start == 0.0]
    assert four_bar, [ (l.bars, l.start, l.end) for l in loops ]
    assert four_bar[0].end == 8.2
    assert four_bar[0].end_from_grid is True
    assert "grid downbeat end" in four_bar[0].reasons
    assert four_bar[0].to_dict()["endFromGrid"] is True


def test_loop_end_fallback_is_flagged_when_no_grid_downbeat():
    grid = _loop_grid([], [Phrase(index=0, start_time=0.0,
                                  end_time=8.0, bar_count=4)])
    loops = detect_loops(grid, _flat_curve(64.0), duration=64.0)
    four_bar = [l for l in loops if l.bars == 4 and l.start == 0.0]
    assert four_bar
    assert four_bar[0].end == 8.0  # start + nominal duration
    assert four_bar[0].end_from_grid is False
    assert any("fallback" in r for r in four_bar[0].reasons)


def test_loop_end_from_grid_helper():
    grid = _loop_grid([0.0, 2.0, 4.0, 6.0, 8.0], [])
    end, ok = _loop_end_from_grid(grid, 0.0, 4, 999.0)
    assert (end, ok) == (8.0, True)
    end, ok = _loop_end_from_grid(grid, 0.0, 99, 999.0)
    assert (end, ok) == (999.0, False)


# -- T-5: readiness corrections raise trust -------------------------------

def _cue(conf: float) -> Cue:
    return Cue(slot=1, type=CueType.DROP, time=5.0, label="D",
               confidence=conf, color="#fff")


def _sections() -> list[Section]:
    return [
        Section(type=SectionType.INTRO, start=0.0, end=2.0, confidence=0.9),
        Section(type=SectionType.DROP, start=4.0, end=6.0, confidence=0.9),
        Section(type=SectionType.OUTRO, start=8.0, end=10.0, confidence=0.9),
    ]


def test_corrections_never_lower_readiness():
    assert _correction_stability(0) == 0.8
    assert _correction_stability(4) == 1.0
    base = compute_readiness(cues=[_cue(0.9)], sections=_sections(),
                             correction_count=0)
    corrected = compute_readiness(cues=[_cue(0.9)], sections=_sections(),
                                  correction_count=4)
    assert corrected.score >= base.score
    assert any("human-verified" in r for r in corrected.reasons)


# -- T-5: structure fallback + second-drop evidence ------------------------

def test_classify_single_fallback_is_not_drop():
    det = StructureDetector()
    det._strategy = STRATEGY_GRID
    det._flux = np.full(20, 0.5)
    f = {"rel_energy": 0.4, "slope": 0.0, "mean_flux": 0.1,
         "mean_harmonic": 0.3, "position": 0.5}
    label, is_fallback = det._classify_single(1, f, 4, 0.8, 0.2, 1.0, None)
    assert label != SectionType.DROP
    assert is_fallback is True


def test_second_drop_requires_intervening_contrast():
    det = StructureDetector()
    det._strategy = STRATEGY_GRID
    det._beat_times = np.arange(0.0, 64.0, 0.5)
    # _section_features masks beats in [start, end): sections below are
    # 0-16s (high), 16-32s (high), 32-48s (low), 48-64s (high).
    det._energy = np.concatenate([np.full(32, 0.9), np.full(32, 0.9),
                                  np.full(32, 0.1), np.full(32, 0.9)])
    det._flux = np.concatenate([np.full(64, 0.1), np.full(32, 1.0),
                                np.full(32, 0.1)])
    det._harmonic_ratio = np.full(128, 0.2)
    sections = [
        Section(type=SectionType.INTRO, start=0.0, end=16.0, confidence=0.0),
        Section(type=SectionType.INTRO, start=16.0, end=32.0, confidence=0.0),
        Section(type=SectionType.INTRO, start=32.0, end=48.0, confidence=0.0),
        Section(type=SectionType.INTRO, start=48.0, end=64.0, confidence=0.0),
    ]
    out = det._classify(sections, None)
    types = [s.type for s in out]
    # First two high sections merge into one DROP (no contrast between
    # them, so the second is NOT promoted); the high section after the
    # low BREAKDOWN is a genuine SECOND_DROP.
    assert types.count(SectionType.SECOND_DROP) == 1
    assert types[-1] == SectionType.SECOND_DROP
    assert SectionType.DROP in types


# -- R-4: router cap values pinned ------------------------------------------

def test_router_caps_local_below_other_strategies():
    assert confidence_cap(STRATEGY_LOCAL) == 0.7
    assert confidence_cap(STRATEGY_GRID) == 0.99
    # The stem path (core/stems/sections.py, out of scope) clips at 0.98
    # with no strategy cap -- documented in router.confidence_cap, not
    # silently equalized here. This pins the classic-path values so any
    # future harmonization is a deliberate change, not drift.
    assert confidence_cap(STRATEGY_LOCAL) < 0.98
