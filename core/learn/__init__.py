"""Learning engine for user cue personalization and correction tracking.

The DJ's manual hot-cue corrections are stored in a local SQLite store
(`core.learn.store.LearningStore`). `apply_learned_patterns` nudges freshly
generated cue positions toward the user's learned per-genre prior so future
analyses get closer to what the DJ actually wants.
"""

from __future__ import annotations

from typing import Any, Iterable

from .store import MIN_SAMPLES, LearningStore, get_store
from .preferences import (
    ADAPT_MAX_DELTA,
    PreferenceProfile,
    PreferenceStore,
    adaptive_weights,
    apply_preference,
    get_preference_store,
    slot_strategy_for,
)

__all__ = [
    "MIN_SAMPLES",
    "LearningStore",
    "get_store",
    "apply_learned_patterns",
    "snap_to_nearest_grid",
    "PreferenceProfile",
    "PreferenceStore",
    "get_preference_store",
    "adaptive_weights",
    "apply_preference",
    "slot_strategy_for",
    "ADAPT_MAX_DELTA",
    "MAX_LEARNED_SHIFT_SECONDS",
]

MAX_LEARNED_SHIFT_SECONDS = 4.0
"""Max seconds a learned prior may shift a cue in one pass.

Rationale: the old cap was 15% of track length — 45 s on a 5-minute
track, far beyond any plausible DJ correction. 4 s is ~1 bar at slow
tempos (60 BPM) and ~2 bars at 120 BPM: large enough to absorb a real
systematic placement bias, small enough that a learned prior can never
drag a cue into a different section. The shift is additionally
snapped back to the beatgrid (see :func:`snap_to_nearest_grid`).
"""


def snap_to_nearest_grid(
    time: float,
    beat_times: list[float] | None = None,
    downbeat_times: list[float] | None = None,
) -> float:
    """Snap a cue time to the nearest beat (or downbeat) from given lists.

    This module has no BeatGrid object of its own, so it takes plain
    time lists — the pipeline must pass ``grid.beat_times`` /
    ``grid.downbeat_times`` when calling :func:`apply_learned_patterns`
    (see coordinator notes). Policy: snap to the nearest BEAT when
    beats are given (beats include downbeats, and beat-level snapping
    moves a cue by at most half a beat interval); when only downbeats
    are given, snap to the nearest downbeat; with neither list, return
    the time unchanged. Times outside the grid clamp to its ends.
    """
    beats = [float(t) for t in (beat_times or [])]
    downbeats = [float(t) for t in (downbeat_times or [])]
    grid = beats if beats else downbeats
    if not grid:
        return float(time)
    grid = sorted(grid)
    if time <= grid[0]:
        return grid[0]
    if time >= grid[-1]:
        return grid[-1]
    import bisect

    idx = bisect.bisect_left(grid, time)
    before, after = grid[idx - 1], grid[idx]
    return before if (time - before) <= (after - time) else after


def apply_learned_patterns(
    cues: list[Any],
    track_duration: float,
    genre: str,
    store: LearningStore | None = None,
    beat_times: list[float] | None = None,
    downbeat_times: list[float] | None = None,
) -> list[Any]:
    """Adjust auto-generated cue times toward the user's learned patterns.

    Only trusted patterns (>= MIN_SAMPLES distinct tracks for that genre+slot)
    are applied, and each shift is clamped to at most
    `MAX_LEARNED_SHIFT_SECONDS` (4 s — see above). After shifting, the cue
    is re-snapped to the nearest beat/downbeat when ``beat_times`` /
    ``downbeat_times`` are supplied, and finally clamped to
    [0, track_duration]. Cues that the user locked or added manually are
    never moved.

    Args:
        cues: list of core.models.Cue (in place adjusted; new list returned).
        track_duration: track length in seconds.
        genre: genre profile name (e.g. "baile").
        store: learning store; defaults to the module singleton.
        beat_times: beatgrid beat times to snap to (pipeline: grid.beat_times).
        downbeat_times: downbeat times, used only when no beats given.

    Returns:
        The same list of cues (mutated in place).
    """
    if not cues or track_duration <= 0:
        return cues

    store = store or get_store()
    patterns = {p["slot"]: p for p in store.patterns_for(genre) if p.get("trusted")}
    if not patterns:
        return cues

    for cue in cues:
        if cue.locked or getattr(cue, "source", "dsp") == "user":
            continue
        pat = patterns.get(cue.slot)
        if not pat:
            continue

        learned_time = float(pat["avgPosition"]) * track_duration
        shift = learned_time - cue.time
        if abs(shift) > MAX_LEARNED_SHIFT_SECONDS:
            shift = MAX_LEARNED_SHIFT_SECONDS * (1.0 if shift > 0 else -1.0)
        new_time = cue.time + shift
        if beat_times or downbeat_times:
            new_time = snap_to_nearest_grid(new_time, beat_times, downbeat_times)
        cue.time = min(max(0.0, new_time), track_duration)
        reason = f"learned {pat['samples']}x"
        if reason not in cue.reason:
            cue.reason.append(reason)

    return cues


def record_feedback_rows(cues: list[Any], track_path: str, genre: str, track_duration: float, store: LearningStore | None = None) -> int:
    """Record the DJ-approved cue layout as learning feedback.

    Args:
        cues: final (user-approved) cue list.
        track_path: audio file path (identifier for the upsert key).
        genre: genre profile name.
        track_duration: track length in seconds.
        store: learning store; defaults to the module singleton.

    Returns:
        Number of feedback rows written.
    """
    store = store or get_store()
    rows = []
    for cue in cues:
        if getattr(cue, "deleted", False):
            continue
        d = cue.to_dict() if hasattr(cue, "to_dict") else cue
        rows.append(
            {
                "trackPath": track_path,
                "genre": genre,
                "slot": int(d.get("slot", 0)),
                "time": float(d.get("time", 0.0)),
                "trackDuration": float(track_duration),
                "sectionType": str(d.get("type", "")),
                "cueRole": str(d.get("label", "")),
            }
        )
    return store.record_batch(rows) if rows else 0
