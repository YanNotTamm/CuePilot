"""Beatgrid detection package."""

from .detector import (
    BeatGrid,
    detect_beatgrid,
    snap_to_beat,
    snap_to_downbeat,
    verify_kick_snare_tempo,
)
from .subdivision import (
    FIVE_KICK,
    STRAIGHT,
    TRESILLO,
    TRIPLET,
    TWO_STEP,
    UNKNOWN,
    classify_subdivision,
)

__all__ = [
    "BeatGrid",
    "detect_beatgrid",
    "snap_to_beat",
    "snap_to_downbeat",
    "verify_kick_snare_tempo",
    "classify_subdivision",
    "STRAIGHT",
    "TRIPLET",
    "TRESILLO",
    "TWO_STEP",
    "FIVE_KICK",
    "UNKNOWN",
]
