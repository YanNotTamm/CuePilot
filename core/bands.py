"""Band-limited audio features (logic_new.md §6.2, §6.5, Genre_pattern.md §42).

Adds the sub-bass (20-120 Hz) band-limited energy/novelty used for drop
detection in energy-based genres (Dubstep/Trap) and the reverse-bass detector
for Hardstyle (Genre_pattern.md §42).

The analyzer already computes `subbass_energy` (20-120 Hz per-frame RMS).
These helpers derive per-beat novelty and the reverse-bass signature from it.
"""

from __future__ import annotations

import numpy as np

from .analyzer import AnalysisFeatures
from .beatgrid import BeatGrid


def subbass_per_beat(features: AnalysisFeatures, grid: BeatGrid) -> np.ndarray:
    """Sub-bass energy sampled at each beat of the grid (aligned to beats)."""
    energy = np.asarray(features.subbass_energy, dtype=float)
    times = np.asarray(features.frame_times, dtype=float)
    beats = np.asarray(grid.beat_times, dtype=float)
    if energy.size == 0 or times.size == 0 or beats.size == 0:
        return np.zeros(beats.size, dtype=float)
    idx = np.searchsorted(times, beats)
    idx = np.clip(idx, 0, energy.size - 1)
    return np.asarray(energy[idx], dtype=float)


def subbass_novelty(features: AnalysisFeatures, grid: BeatGrid) -> np.ndarray:
    """Per-beat band-limited novelty of the sub-bass band (20-120 Hz).

    Drops are signalled by a sudden *jump* in sub-bass energy (logic_new.md
    §6.5). Returns the absolute per-beat difference, normalized to 0..1.
    """
    per_beat = subbass_per_beat(features, grid)
    if per_beat.size < 2:
        return np.zeros_like(per_beat)
    diff = np.abs(np.diff(per_beat))
    diff = np.concatenate([[0.0], diff])
    m = float(np.max(diff))
    if m <= 1e-9:
        return np.zeros_like(diff)
    return diff / m


def reverse_bass_score(features: AnalysisFeatures, grid: BeatGrid) -> float:
    """0..1 confidence that the track uses a Hardstyle reverse-bass pattern.

    Reverse bass (Genre_pattern.md §42) is an offbeat bass whose energy rises
    through the beat (a reversed kick sweep). It shows up as a strong, stable
    sub-bass pulse on the offbeats. We measure the ratio of offbeat to
    onbeat sub-bass energy across the track; values near 1 (offbeat ≈ onbeat)
    indicate reverse bass, values near 0 indicate a standard onbeat kick.
    """
    per_beat = subbass_per_beat(features, grid)
    if per_beat.size < 8:
        return 0.0
    # Beats are indexed 0..; within a 4/4 bar the onbeat kick is at phase 0.
    onbeat = per_beat[::4] if per_beat.size >= 4 else per_beat
    offbeat_mask = np.arange(per_beat.size) % 4 != 0
    offbeat = per_beat[offbeat_mask]
    on_mean = float(np.mean(onbeat)) if onbeat.size else 0.0
    off_mean = float(np.mean(offbeat)) if offbeat.size else 0.0
    denom = on_mean + off_mean
    if denom <= 1e-9:
        return 0.0
    return float(np.clip(off_mean / denom, 0.0, 1.0))