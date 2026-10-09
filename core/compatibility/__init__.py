"""Harmonic mixing and track compatibility layer.

- `core.compatibility.advisor` — Harmonic Mixing & Transition Advisor (45.1).
- `core.compatibility.mashup` — Mashup / Acapella Compatibility Finder (45.4).
"""

from __future__ import annotations

from .advisor import (
    WEIGHTS,
    CompatibilityResult,
    best_next_tracks,
    bpm_compatibility,
    key_compatibility,
    key_to_camelot,
    score_pair,
)
from .mashup import MashupCandidate, find_mashups

__all__ = [
    "WEIGHTS",
    "CompatibilityResult",
    "best_next_tracks",
    "bpm_compatibility",
    "key_compatibility",
    "key_to_camelot",
    "score_pair",
    "MashupCandidate",
    "find_mashups",
]