"""Mashup / Acapella Compatibility Finder.

Combines the Harmonic Mixing Advisor (45.1) with stem information (45.2):
suggests track pairs where the *acapella* of track A (vocal stem) is likely
to work over the *instrumental* of track B (everything but vocals) in key +
BPM — surfacing on-the-fly mashup candidates.

Note: this never renders new audio; it only
suggests *timing & pairings*. Execution stays manual in the DJ's own
software.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .advisor import (
    WEIGHTS,
    CompatibilityResult,
    bpm_compatibility,
    key_compatibility,
)

# Weight of the harmonic/BPM components vs stem-availability bonus.
STEM_AVAIL_BONUS = 0.08


@dataclass
class MashupCandidate:
    """A suggested acapella-over-instrumental pairing."""

    acapella_track_id: str
    instrumental_track_id: str
    score: float
    key_compatibility: str
    bpm_compatibility: str
    bpm_delta: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "acapellaTrackId": self.acapella_track_id,
            "instrumentalTrackId": self.instrumental_track_id,
            "score": round(self.score, 3),
            "keyCompatibility": self.key_compatibility,
            "bpmCompatibility": self.bpm_compatibility,
            "bpmDelta": round(self.bpm_delta, 3),
            "reasons": self.reasons,
        }


def _has_vocals(track: dict[str, Any]) -> bool:
    return bool(track.get("hasVocals", True)) and not bool(track.get("instrumentalOnly", False))


def _has_instrumental(track: dict[str, Any]) -> bool:
    return not bool(track.get("acapellaOnly", False)) and not bool(track.get("instrumentalOnly", False))


def _stem_ratio(track: dict[str, Any]) -> float:
    """0..1 confidence that this track exposes usable acapella/instrumental."""
    stems = track.get("stems")
    if not stems:
        return 0.3
    ratio = stems.get("vocalRatio")
    if ratio is not None:
        return float(ratio)
    return 0.5


def find_mashups(
    library: list[dict[str, Any]],
    top_n: int = 5,
    min_score: float = 0.55,
) -> list[MashupCandidate]:
    """Find acapella-over-instrumental candidates across the whole library.

    Args:
        library: per-track records including `key`, `bpm`, and optionally
            `hasVocals` / `instrumentalOnly` / `acapellaOnly` / `stems`.
        top_n: max candidates returned.
        min_score: minimum combined score to keep a candidate.
    """
    candidates: list[MashupCandidate] = []
    for a in library:
        if not _has_vocals(a):
            continue
        for b in library:
            a_id = a.get("id") or a.get("trackId")
            b_id = b.get("id") or b.get("trackId")
            if a_id == b_id or a_id is None or b_id is None:
                continue
            if not _has_instrumental(b):
                continue
            if b.get("acapellaOnly", False):
                continue

            key_label, key_score = key_compatibility(str(a.get("key", "")), str(b.get("key", "")))
            bpm_label, bpm_score, bpm_delta = bpm_compatibility(
                float(a.get("bpm", 0.0)), float(b.get("bpm", 0.0))
            )

            harmonic = WEIGHTS["key"] * key_score + WEIGHTS["bpm"] * bpm_score
            # Vocal-density of A + instrumental availability of B as a bonus.
            stems_bonus = STEM_AVAIL_BONUS * ((_stem_ratio(a) + _stem_ratio(b)) / 2.0)
            total = harmonic + stems_bonus

            if total < min_score:
                continue

            reasons = []
            if key_label == "same":
                reasons.append("same key")
            elif key_label == "relative":
                reasons.append("relative key")
            if bpm_label in ("exact", "pitch", "half_time", "double_time"):
                reasons.append(f"bpm {bpm_label}")
            if bpm_delta <= 0.06:
                reasons.append("pitch-friendly")
            reasons.append("acapella over instrumental")

            candidates.append(
                MashupCandidate(
                    acapella_track_id=str(a_id),
                    instrumental_track_id=str(b_id),
                    score=float(total),
                    key_compatibility=key_label,
                    bpm_compatibility=bpm_label,
                    bpm_delta=float(bpm_delta),
                    reasons=reasons,
                )
            )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:top_n]