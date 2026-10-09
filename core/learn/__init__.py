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
    "PreferenceProfile",
    "PreferenceStore",
    "get_preference_store",
    "adaptive_weights",
    "apply_preference",
    "slot_strategy_for",
    "ADAPT_MAX_DELTA",
]

MAX_SHIFT_RATIO = 0.15
"""Max fraction of track length a learned prior may shift a cue in one pass."""


def apply_learned_patterns(
    cues: list[Any],
    track_duration: float,
    genre: str,
    store: LearningStore | None = None,
) -> list[Any]:
    """Adjust auto-generated cue times toward the user's learned patterns.

    Only trusted patterns (>= MIN_SAMPLES distinct tracks for that genre+slot)
    are applied, and each shift is clamped so a prior can never drag a cue
    more than `MAX_SHIFT_RATIO` of the track length. Cues that the user locked
    or added manually are never moved.

    Args:
        cues: list of core.models.Cue (in place adjusted; new list returned).
        track_duration: track length in seconds.
        genre: genre profile name (e.g. "baile").
        store: learning store; defaults to the module singleton.

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
        max_shift = track_duration * MAX_SHIFT_RATIO
        if abs(shift) > max_shift:
            shift = max_shift * (1.0 if shift > 0 else -1.0)
        cue.time = max(0.0, cue.time + shift)
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
