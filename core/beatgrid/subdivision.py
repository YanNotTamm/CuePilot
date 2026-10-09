"""Subdivision detection for non-standard beat grids (Genre_pattern.md §707-721).

A generic 16th-note assumption misreads the beatgrid in several genres
(Genre_pattern.md §717):

- Baile Funk / tamborzão  : triplet-based kick subdivision
- Reggaeton / dembow      : tresillo (3-3-2) subdivision
- UK Garage 2-step        : broken kick pattern (not four-on-the-floor)
- Jersey Club             : five-kick syncopated pattern

This module analyzes the intra-beat onset pattern and classifies the dominant
subdivision, so the beatgrid/downbeat logic can adapt instead of assuming a
straight 16th grid.
"""

from __future__ import annotations

import numpy as np

from ..analyzer import AnalysisFeatures

# Subdivision labels (canonical, lowercase).
STRAIGHT = "straight"
TRIPLET = "triplet"
TRESILLO = "tresillo"
TWO_STEP = "two_step"
FIVE_KICK = "five_kick"
UNKNOWN = "unknown"

SUBDIVISIONS = (STRAIGHT, TRIPLET, TRESILLO, TWO_STEP, FIVE_KICK, UNKNOWN)

# Sub-beat probe offsets (fraction of the beat period).
_TRIPLET_OFFSETS = (0.0, 1.0 / 3.0, 2.0 / 3.0)
_STRAIGHT_OFFSETS = (0.0, 0.5)
_TRESILLO_OFFSETS = (0.0, 3.0 / 8.0, 6.0 / 8.0)


def _probe_onsets(
    features: AnalysisFeatures,
    beat_times: np.ndarray,
    offsets: tuple[float, ...],
) -> np.ndarray:
    """Mean onset strength at `offsets` after each beat (per offset)."""
    onset = np.asarray(features.onset_env, dtype=float)
    times = np.asarray(features.frame_times, dtype=float)
    if onset.size == 0 or times.size == 0 or beat_times.size < 2:
        return np.zeros(len(offsets), dtype=float)

    periods = np.diff(beat_times)
    period = float(np.median(periods)) if periods.size else 0.0
    if period <= 0:
        return np.zeros(len(offsets), dtype=float)

    out: list[float] = []
    for off in offsets:
        probe = beat_times[:-1] + off * period
        idx = np.searchsorted(times, probe)
        idx = np.clip(idx, 0, onset.size - 1)
        out.append(float(np.mean(onset[idx])))
    return np.asarray(out, dtype=float)


def _four_on_floor_strength(features: AnalysisFeatures, beat_times: np.ndarray) -> float:
    """0..1 measure of how strongly kicks land on every beat.

    Four-on-the-floor genres have a strong onset on every beat; 2-step and
    Jersey Club break that up. Computed as the mean onset of even vs odd beats.
    """
    onset = np.asarray(features.onset_env, dtype=float)
    times = np.asarray(features.frame_times, dtype=float)
    if onset.size == 0 or times.size == 0 or beat_times.size < 8:
        return 1.0
    idx = np.searchsorted(times, beat_times)
    idx = np.clip(idx, 0, onset.size - 1)
    vals = onset[idx]
    if vals.size < 8:
        return 1.0
    even = vals[::2]
    odd = vals[1::2]
    m = float(np.max(vals))
    if m <= 1e-9:
        return 1.0
    # For four-on-the-floor, even/odd beats have similar strength (diff ~0).
    return float(np.clip(1.0 - abs(np.mean(even) - np.mean(odd)) / m, 0.0, 1.0))


def _five_kick_strength(features: AnalysisFeatures, beat_times: np.ndarray) -> float:
    """0..1 evidence of the Jersey Club 5-kick gallop.

    Five-kick syncopation puts ghost kicks on the "and"/"e" of beats 2 and 3,
    so sub-beat positions ~0.25/0.75 of the bar are unusually strong while the
    even-beat regularity of four-on-the-floor is broken.
    """
    bar_period = 4.0
    if beat_times.size < 8:
        return 0.0
    ghost_onsets = _probe_onsets(features, beat_times, (0.5 / 4.0, 3.0 / 4.0))
    m = float(np.max(ghost_onsets)) if ghost_onsets.size else 0.0
    if m <= 1e-9:
        return 0.0
    return float(np.clip(m * 2.0, 0.0, 1.0))


def _tresillo_strength(features: AnalysisFeatures, beat_times: np.ndarray) -> float:
    """0..1 evidence of the 3-3-2 (tresillo) dembow accent."""
    if beat_times.size < 8:
        return 0.0
    probes = _probe_onsets(features, beat_times, _TRESILLO_OFFSETS)
    straight = _probe_onsets(features, beat_times, _STRAIGHT_OFFSETS)
    if probes.size != 3 or straight.size != 2:
        return 0.0
    m = float(np.max(np.concatenate([probes, straight])))
    if m <= 1e-9:
        return 0.0
    # Tresillo places energy at 0, 3/8, 6/8 of the bar; strong at the 3/8 and
    # 6/8 accents relative to the simple half-beat probe is the signature.
    return float(np.clip((probes[1] + probes[2] - straight[1]) / m, 0.0, 1.0))


def _triplet_strength(features: AnalysisFeatures, beat_times: np.ndarray) -> float:
    """0..1 evidence of triplet subdivision (Baile Funk tamborzão)."""
    if beat_times.size < 8:
        return 0.0
    trip = _probe_onsets(features, beat_times, _TRIPLET_OFFSETS)
    straight = _probe_onsets(features, beat_times, _STRAIGHT_OFFSETS)
    if trip.size != 3 or straight.size != 2:
        return 0.0
    m = float(np.max(np.concatenate([trip, straight])))
    if m <= 1e-9:
        return 0.0
    # Triplet puts energy at 1/3 and 2/3 of the beat; the half-beat position
    # is weak. Compare the triplet mid (1/3, 2/3) vs the half-beat probe.
    return float(np.clip((trip[1] + trip[2]) / m, 0.0, 1.0))


def classify_subdivision(
    features: AnalysisFeatures,
    beat_times: np.ndarray | list[float],
    hint: str | None = None,
) -> tuple[str, float]:
    """Classify the dominant subdivision from the intra-beat onset pattern.

    Returns ``(subdivision, confidence)``. A genre `hint` (e.g. "baile",
    "reggaeton", "uk_garage", "jersey_club") is used as a soft tie-breaker and
    to bias borderline cases toward the genre's documented grid.
    """
    beat_times = np.asarray(beat_times, dtype=float)
    if beat_times.size < 8:
        return UNKNOWN, 0.0

    four = _four_on_floor_strength(features, beat_times)
    triplet = _triplet_strength(features, beat_times)
    tresillo = _tresillo_strength(features, beat_times)
    five = _five_kick_strength(features, beat_times)

    scores = {
        STRAIGHT: four,
        TRIPLET: triplet,
        TRESILLO: tresillo,
        FIVE_KICK: five,
    }

    # Genre hint is a tie-breaker and overrides a four-on-the-floor reading
    # when the genre is documented with a non-straight grid (Genre_pattern.md
    # §717): a Baile track still plays tamborzão triplets even when the onset
    # envelope reads as strong on every beat.
    hint_key = (hint or "").strip().lower().replace("-", "_").replace(" ", "_")
    hint_map = {
        "baile": TRIPLET,
        "reggaeton": TRESILLO,
        "dancehall": TRESILLO,
        "uk_garage": TWO_STEP,
        "speed_garage": TWO_STEP,
        "jersey_club": FIVE_KICK,
        "jersey": FIVE_KICK,
    }
    hinted = hint_map.get(hint_key)

    if hinted in (TRIPLET, TRESILLO, FIVE_KICK):
        # Syncopated grids: the hint decides unless the evidence strongly
        # disagrees (confidence of the hinted grid essentially zero).
        base = scores[hinted]
        return hinted, round(max(base, 0.35), 3)

    if hinted == TWO_STEP:
        # 2-step = broken kick: return it when four-on-the-floor is weak.
        if four < 0.55:
            return TWO_STEP, round(four, 3)
        return STRAIGHT, round(min(0.99, four), 3)

    # No hint or no documented non-straight grid: straight if strong.
    if four >= 0.55 and triplet < 0.5 and tresillo < 0.5 and five < 0.5:
        return STRAIGHT, round(min(0.99, four), 3)

    # Determine the dominant syncopated pattern.
    best = max(scores, key=lambda k: scores[k])
    best_score = scores[best]
    if best_score >= 0.5:
        return best, round(min(0.99, best_score), 3)
    return UNKNOWN, round(best_score, 3)