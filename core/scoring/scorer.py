"""Cue scoring based on structural importance, energy, and downbeat alignment.

Implements the configurable weighted CueScore formula:

    CueScore =
      0.25 * PhraseBoundary
    + 0.20 * EnergyChange
    + 0.15 * SpectralChange
    + 0.15 * RhythmChange
    + 0.10 * VocalOnset
    + 0.10 * BeatAlignment
    + 0.05 * GenrePrior

Each sub-score is normalized to [0, 1]. Weights are configurable and can be
tuned per dataset (Section 9: "Bobot harus configurable dan dapat di-tuning").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..analyzer import AnalysisFeatures
from ..beatgrid import BeatGrid
from ..models import GenreProfile


@dataclass
class CueScoreResult:
    total: float
    components: dict[str, float]


def _value_at(features: AnalysisFeatures, time: float) -> int:
    """Nearest feature-frame index for a time in seconds."""
    times = features.frame_times
    if times.size == 0:
        return 0
    idx = int(np.searchsorted(times, time))
    idx = min(max(idx, 0), times.size - 1)
    return idx


def phrase_boundary_score(time: float, grid: BeatGrid) -> float:
    """1.0 if the time lands on a downbeat, else 0.0 (phrase boundary)."""
    for dt in grid.downbeat_times:
        if abs(dt - time) < 0.02:
            return 1.0
    return 0.0


def energy_change_score(features: AnalysisFeatures, time: float) -> float:
    """Normalized onset/RMS change at the time point."""
    idx = _value_at(features, time)
    rms = features.rms
    if rms.size < 3:
        return 0.0
    lo = max(0, idx - 2)
    hi = min(rms.size, idx + 3)
    window = rms[lo:hi]
    span = float(np.max(window) - np.min(window))
    base = float(np.max(rms) - np.min(rms)) or 1.0
    return float(np.clip(span / base, 0.0, 1.0))


def spectral_change_score(features: AnalysisFeatures, time: float) -> float:
    """Normalized spectral flux around the time point."""
    idx = _value_at(features, time)
    flux = features.spectral_flux
    if flux.size == 0:
        return 0.0
    lo = max(0, idx - 2)
    hi = min(flux.size, idx + 3)
    window = flux[lo:hi]
    peak = float(np.max(window))
    base = float(np.percentile(flux, 95)) or 1.0
    return float(np.clip(peak / base, 0.0, 1.0))


def rhythm_change_score(features: AnalysisFeatures, time: float) -> float:
    """Normalized onset-envelope change (rhythmic density shift)."""
    idx = _value_at(features, time)
    onset = features.onset_env
    if onset.size < 3:
        return 0.0
    lo = max(0, idx - 2)
    hi = min(onset.size, idx + 3)
    window = onset[lo:hi]
    span = float(np.max(window) - np.min(window))
    base = float(np.max(onset) - np.min(onset)) or 1.0
    return float(np.clip(span / base, 0.0, 1.0))


def vocal_onset_score(features: AnalysisFeatures, time: float) -> float:
    """Approximate vocal presence via harmonic/percussive ratio.

    High harmonic content relative to percussive content at a phrase boundary
    is a weak proxy for a vocal entry (MVP Phase-1 DSP baseline; refined later
    by stem-aware detection in Section 45.2 and the LLM semantic layer).
    """
    idx = _value_at(features, time)
    harm = features.harmonic
    perc = features.percussive
    if harm.size == 0 or perc.size == 0:
        return 0.0
    h = float(harm[idx]) if 0 <= idx < harm.size else 0.0
    p = float(perc[idx]) if 0 <= idx < perc.size else 0.0
    if (h + p) <= 1e-9:
        return 0.0
    ratio = h / (h + p)
    # Harmonic-dominant regions score higher.
    return float(np.clip((ratio - 0.5) * 2.0, 0.0, 1.0))


def beat_alignment_score(time: float, grid: BeatGrid) -> float:
    """1.0 if the time is on a downbeat, 0.7 on any beat, else 0.0."""
    for dt in grid.downbeat_times:
        if abs(dt - time) < 0.02:
            return 1.0
    for b in grid.beats:
        if abs(b.time - time) < 0.02:
            return 0.7
    return 0.0


def genre_prior_score(section_type, profile: GenreProfile) -> float:
    """1.0 if the section type appears in the profile's expected order."""
    if profile is None:
        return 0.5
    expected = [s.value for s in profile.section_order]
    return 1.0 if section_type in expected else 0.3


def compute_cue_score(
    features: AnalysisFeatures | None,
    grid: BeatGrid,
    profile: GenreProfile | None,
    section_type,
    time: float,
    weights: dict[str, float] | None = None,
) -> CueScoreResult:
    """Compute the weighted CueScore for a candidate cue time (Section 9).

    `features` may be None for grid-only scoring (tests / degenerate paths);
    feature-based sub-scores are then 0.
    """
    if weights is None:
        weights = (
            profile.cue_weights
            if profile is not None
            else {
                "phrase_boundary": 0.25,
                "energy_change": 0.20,
                "spectral_change": 0.15,
                "rhythm_change": 0.15,
                "vocal_onset": 0.10,
                "beat_alignment": 0.10,
                "genre_prior": 0.05,
            }
        )

    components = {
        "phrase_boundary": phrase_boundary_score(time, grid),
        "energy_change": energy_change_score(features, time) if features is not None else 0.0,
        "spectral_change": spectral_change_score(features, time) if features is not None else 0.0,
        "rhythm_change": rhythm_change_score(features, time) if features is not None else 0.0,
        "vocal_onset": vocal_onset_score(features, time) if features is not None else 0.0,
        "beat_alignment": beat_alignment_score(time, grid),
        "genre_prior": genre_prior_score(section_type, profile),
    }

    total = 0.0
    weight_sum = 0.0
    for key, comp_score in components.items():
        w = weights.get(key, 0.0)
        total += w * comp_score
        weight_sum += w

    if weight_sum <= 0.0:
        total = 0.0
    else:
        total /= weight_sum

    return CueScoreResult(total=float(np.clip(total, 0.0, 1.0)), components=components)
