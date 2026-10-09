"""Harmonic Mixing & Transition Advisor.

Computes a compatibility score for every track pair in a library from:

- key compatibility (Camelot Wheel: same key, adjacent, relative major/minor,
  energy boost +1)
- BPM compatibility (exact, half/double-time, ±6% pitch range)
- energy-curve compatibility (outro energy of A vs intro energy of B)

Output: per track, a ranked list of "Best Next Tracks" plus the recommended
cue pairing (outro cue of A ↔ intro cue of B).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Camelot Wheel (harmonic mixing standard), 24 positions 1A..12B.
# Position number is 1..24; relative major/minor share the same root number
# (e.g. 8A = A minor and 8B = C major), adjacent = ±1 position in the circle.
_CAMELOT_POSITIONS: dict[str, int] = {
    "1A": 1, "1B": 2,
    "2A": 3, "2B": 4,
    "3A": 5, "3B": 6,
    "4A": 7, "4B": 8,
    "5A": 9, "5B": 10,
    "6A": 11, "6B": 12,
    "7A": 13, "7B": 14,
    "8A": 15, "8B": 16,
    "9A": 17, "9B": 18,
    "10A": 19, "10B": 20,
    "11A": 21, "11B": 22,
    "12A": 23, "12B": 24,
}

# Root → (minor position, major position) on the wheel.
_CAMELOT_MINOR = {"G#": 1, "D#": 3, "Bb": 5, "F": 7, "C": 9, "G": 11, "D": 13, "A": 15, "E": 17, "B": 19, "F#": 21, "C#": 23}
_CAMELOT_MAJOR = {"B": 2, "F#": 4, "Db": 6, "Ab": 8, "Eb": 10, "Bb": 12, "F": 14, "C": 16, "G": 18, "D": 20, "A": 22, "E": 24}

_CAMELOT_LABELS = [
    "1A", "1B", "2A", "2B", "3A", "3B", "4A", "4B",
    "5A", "5B", "6A", "6B", "7A", "7B", "8A", "8B",
    "9A", "9B", "10A", "10B", "11A", "11B", "12A", "12B",
]
_NUM_TO_LABEL = {i + 1: label for i, label in enumerate(_CAMELOT_LABELS)}


def key_to_camelot(key: str) -> int | None:
    """Convert a musical key string (e.g. 'F#m', 'C# minor', 'Bb') to Camelot.

    Returns the Camelot wheel position (1..24) or None when unparsable.
    """
    if not key:
        return None
    k = key.strip().replace("\u266f", "#").replace("\u266d", "b")
    lower = k.lower()
    if lower.endswith("minor") or lower.endswith("m") and not lower.endswith("maj"):
        root = k.replace("minor", "").replace("Minor", "").replace("m", "").strip()
        return _CAMELOT_MINOR.get(root)
    root = k.replace("major", "").replace("Major", "").replace("maj", "").strip()
    return _CAMELOT_MAJOR.get(root)


def camelot_to_key(num: int) -> str:
    return _NUM_TO_LABEL.get(num, f"{num}")


def key_compatibility(key_a: str | None, key_b: str | None) -> tuple[str, float]:
    """Camelot compatibility of two keys → (label, score 0..1).

    Scores: same key 1.0, relative major/minor (+1) 0.95, adjacent 0.85,
    energy boost (two steps, same family) 0.8, else 0.
    """
    if not key_a or not key_b:
        return "unknown", 0.5
    a = key_to_camelot(key_a)
    b = key_to_camelot(key_b)
    if a is None or b is None:
        return "unknown", 0.5
    if a == b:
        return "same", 1.0
    diff = abs(a - b)
    if diff == 1:
        return "relative", 0.95
    if diff == 2:
        return "adjacent", 0.85
    if diff == 3:
        return "energy_boost", 0.8
    return "unrelated", 0.0


def bpm_compatibility(bpm_a: float, bpm_b: float) -> tuple[str, float, float]:
    """BPM compatibility → (label, score 0..1, delta).

    Exact (±0.5) 1.0; half/double-time 0.9; within ±6% pitch range 0.7;
    else score decays. Returns adjusted pitch delta for the transition.
    """
    if bpm_a <= 0 or bpm_b <= 0:
        return "unknown", 0.5, 0.0
    a, b = bpm_a, bpm_b
    # Normalize so the slower is the base (mix both directions).
    if b >= a * 1.6 and b <= a * 2.1:
        return "double_time", 0.9, abs((b / 2.0 - a) / a)
    if a >= b * 1.6 and a <= b * 2.1:
        return "half_time", 0.9, abs((a / 2.0 - b) / b)
    delta = abs(a - b) / min(a, b)
    if delta <= 0.005:
        return "exact", 1.0, 0.0
    if delta <= 0.06:
        return "pitch", 0.7, delta
    return "incompatible", max(0.0, 0.7 - (delta - 0.06) * 2.0), delta


def _avg_energy(curve: list[dict[str, float]], start: float, end: float) -> float:
    vals = [p["energy"] for p in curve if start <= p["time"] <= end]
    return float(sum(vals) / len(vals)) if vals else 0.5


def energy_compatibility(
    energy_a: list[dict[str, float]],
    energy_b: list[dict[str, float]],
    duration_a: float,
    duration_b: float,
) -> float:
    """Outro energy of A vs intro energy of B (0..1 closer = better)."""
    if not energy_a or not energy_b:
        return 0.5
    outro_a = _avg_energy(energy_a, max(0.0, duration_a * 0.85), duration_a)
    intro_b = _avg_energy(energy_b, 0.0, duration_b * 0.15)
    diff = abs(outro_a - intro_b)
    return float(max(0.0, 1.0 - diff * 1.5))


@dataclass
class CompatibilityResult:
    """A ranked compatible track pair."""

    track_id: str
    key_compatibility: str
    bpm_compatibility: str
    bpm_delta: float
    score: float
    suggested_transition: dict[str, str] | None = None
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trackId": self.track_id,
            "keyCompatibility": self.key_compatibility,
            "bpmCompatibility": self.bpm_compatibility,
            "bpmDelta": round(self.bpm_delta, 3),
            "score": round(self.score, 3),
            "suggestedTransition": self.suggested_transition,
            "reasons": self.reasons,
        }


WEIGHTS = {"key": 0.45, "bpm": 0.35, "energy": 0.20}


def score_pair(
    track_a: dict[str, Any],
    track_b: dict[str, Any],
) -> CompatibilityResult:
    """Score one pair (A is the outgoing / current track)."""
    key_label, key_score = key_compatibility(
        str(track_a.get("key", "")), str(track_b.get("key", ""))
    )
    bpm_label, bpm_score, bpm_delta = bpm_compatibility(
        float(track_a.get("bpm", 0.0)), float(track_b.get("bpm", 0.0))
    )
    energy_score = energy_compatibility(
        track_a.get("energyCurve", []),
        track_b.get("energyCurve", []),
        float(track_a.get("duration", 0.0)),
        float(track_b.get("duration", 0.0)),
    )

    total = (
        WEIGHTS["key"] * key_score
        + WEIGHTS["bpm"] * bpm_score
        + WEIGHTS["energy"] * energy_score
    )

    reasons = []
    if key_label == "same":
        reasons.append("same key")
    elif key_label == "relative":
        reasons.append("relative major/minor")
    elif key_label == "adjacent":
        reasons.append("adjacent key")
    elif key_label == "energy_boost":
        reasons.append("energy boost +1")
    if bpm_label != "incompatible":
        reasons.append(f"bpm {bpm_label}")

    transition = None
    if bpm_label != "incompatible" and total >= 0.6:
        # Recommended pairing: outro cue of A ↔ intro cue of B.
        cues_a = track_a.get("cues", [])
        cues_b = track_b.get("cues", [])
        outro_a = next((c for c in cues_a if str(c.get("type", "")).upper() == "OUTRO"), None)
        intro_b = next((c for c in cues_b if str(c.get("type", "")).upper() == "INTRO"), None)
        if outro_a is not None and intro_b is not None:
            transition = {"fromCue": "OUTRO", "toCue": "INTRO"}
        else:
            transition = {"fromCue": "OUTRO", "toCue": "INTRO"}

    return CompatibilityResult(
        track_id=str(track_b.get("id") or track_b.get("trackId", "")),
        key_compatibility=key_label,
        bpm_compatibility=bpm_label,
        bpm_delta=float(bpm_delta),
        score=float(total),
        suggested_transition=transition,
        reasons=reasons,
    )


def best_next_tracks(
    track: dict[str, Any],
    library: list[dict[str, Any]],
    top_n: int = 5,
    min_score: float = 0.5,
) -> list[CompatibilityResult]:
    """Rank compatible tracks for one track within a library."""
    results: list[CompatibilityResult] = []
    track_id = track.get("id") or track.get("trackId")
    for other in library:
        other_id = other.get("id") or other.get("trackId")
        if track_id is not None and other_id == track_id:
            continue
        r = score_pair(track, other)
        if r.score >= min_score:
            results.append(r)
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_n]