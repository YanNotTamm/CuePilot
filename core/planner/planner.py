"""Intelligent Set Planner for automated setlist drafting.

Input (from the user):
    target_duration (min), mood_arc (e.g. warm-up → peak → cool-down),
    genre_pool, must_include track ids.

Output: a *draft* ordered sequence (never auto-play) with recommended cue
transitions between each pair (reusing the Harmonic Mixing Advisor, 45.1)
and an energy-arc chart across the whole set. The DJ always finalizes —
consistent with "system recommends, user approves".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..compatibility.advisor import score_pair


@dataclass
class SetTrack:
    """One track in the drafted set."""

    track_id: str
    position: int
    transition: dict[str, str] | None = None
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "trackId": self.track_id,
            "position": self.position,
            "transition": self.transition,
            "score": round(self.score, 3),
        }


@dataclass
class SetDraft:
    """A draft setlist."""

    tracks: list[SetTrack]
    total_duration: float
    energy_arc: list[dict[str, float]] = field(default_factory=list)
    mood_arc: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tracks": [t.to_dict() for t in self.tracks],
            "totalDuration": round(self.total_duration, 1),
            "energyArc": self.energy_arc,
            "moodArc": self.mood_arc,
        }


def _track_duration(track: dict[str, Any]) -> float:
    return max(0.0, float(track.get("duration", 240.0) or 240.0))


def _track_energy(track: dict[str, Any]) -> float:
    energy = track.get("energy")
    if energy is not None:
        return float(energy)
    return 0.5


def _target_energy(position_ratio: float, mood_arc: list[str]) -> float:
    """Map a 0..1 position into a target energy from the mood arc."""
    if not mood_arc:
        return 0.5 + 0.5 * position_ratio
    step = 1.0 / max(1, len(mood_arc) - 1)
    idx = min(len(mood_arc) - 1, int(round(position_ratio / max(step, 1e-9))))
    label = mood_arc[idx].lower()
    if "peak" in label:
        return 0.85
    if "warm" in label or "cool" in label or "down" in label:
        return 0.35
    return 0.6


def _pick_next(
    current: dict[str, Any],
    remaining: list[dict[str, Any]],
    target_energy: float,
) -> dict[str, Any] | None:
    """Greedily pick the best next track: high harmonic score, energy close
    to the arc target, preference for must-include first."""
    best: dict[str, Any] | None = None
    best_key: tuple[float, float] = (-1.0, 1.0)
    for cand in remaining:
        r = score_pair(current, cand)
        energy_match = 1.0 - abs(_track_energy(cand) - target_energy)
        must_bonus = 0.2 if cand.get("_must_include") else 0.0
        key = (r.score + must_bonus, energy_match)
        if best is None or key[0] > best_key[0] or (key[0] == best_key[0] and key[1] > best_key[1]):
            best = cand
            best_key = key
    return best


def _transition_between(current: dict[str, Any], nxt: dict[str, Any]) -> dict[str, str] | None:
    r = score_pair(current, nxt)
    return r.suggested_transition


def plan_set(
    library: list[dict[str, Any]],
    target_minutes: float = 60.0,
    mood_arc: list[str] | None = None,
    must_include: list[str] | None = None,
    genre_pool: list[str] | None = None,
) -> SetDraft:
    """Draft an ordered setlist from the library.

    Args:
        library: per-track records with `id`/`trackId`, `duration`, `bpm`,
            `key`, `energy`, optionally `genre`.
        target_minutes: target set duration (default 60 min).
        mood_arc: energy arc, e.g. ["warm-up", "build", "peak", "cool-down"].
        must_include: track ids that must appear in the draft.
        genre_pool: restrict the pool to these genres when present.
    """
    mood_arc = mood_arc or ["warm-up", "build", "peak", "cool-down"]
    must_include = must_include or []
    genre_pool = genre_pool or []

    pool = [t for t in library if not genre_pool or (t.get("genre", "") in genre_pool)]
    # Ensure must-include tracks are in the pool even if genre filtering dropped them.
    must_ids = set(must_include)
    pool = [t for t in pool] + [
        t for t in library if (t.get("id") or t.get("trackId")) in must_ids and t not in pool
    ]
    if not pool:
        return SetDraft(tracks=[], total_duration=0.0, energy_arc=[], mood_arc=list(mood_arc))

    target_seconds = max(0.0, float(target_minutes) * 60.0)
    if target_seconds <= 0:
        target_seconds = sum(_track_duration(t) for t in pool)

    # Mark must-include for priority ordering.
    for t in pool:
        t["_must_include"] = (t.get("id") or t.get("trackId")) in must_ids

    # Start the arc warm (low energy opener) if any.
    remaining = list(pool)
    opener = min(remaining, key=lambda t: _track_energy(t))
    opener["_must_include"] = opener.get("_must_include") or bool(must_ids)
    remaining = [t for t in remaining if t is not opener]

    ordered: list[dict[str, Any]] = [opener]
    used_seconds = _track_duration(opener)

    while remaining and used_seconds < target_seconds:
        ratio = used_seconds / max(target_seconds, 1.0)
        target_e = _target_energy(ratio, mood_arc)
        nxt = _pick_next(ordered[-1], remaining, target_e)
        if nxt is None:
            break
        ordered.append(nxt)
        remaining.remove(nxt)
        used_seconds += _track_duration(nxt)

    # If must-include tracks were never placed, append them (honor the request).
    placed_ids = {(t.get("id") or t.get("trackId")) for t in ordered}
    for t in pool:
        tid = t.get("id") or t.get("trackId")
        if tid in must_ids and tid not in placed_ids:
            ordered.append(t)
            used_seconds += _track_duration(t)

    tracks: list[SetTrack] = []
    for i, t in enumerate(ordered):
        transition = None
        if i > 0:
            transition = _transition_between(ordered[i - 1], t)
        tracks.append(
            SetTrack(
                track_id=str(t.get("id") or t.get("trackId", "")),
                position=i + 1,
                transition=transition,
                score=_track_energy(t),
            )
        )

    # Energy arc chart (per-position target + actual).
    energy_arc = [
        {
            "position": i + 1,
            "target": round(_target_energy((i + 1) / max(len(tracks), 1), mood_arc), 3),
            "actual": round(t.score, 3),
        }
        for i, t in enumerate(tracks)
    ]

    return SetDraft(
        tracks=tracks,
        total_duration=round(used_seconds, 1),
        energy_arc=energy_arc,
        mood_arc=list(mood_arc),
    )