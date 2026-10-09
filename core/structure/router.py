"""Genre routing layer (logic_new.md §6.3).

Instead of one generic detection strategy for every genre, CuePilot routes a
track to one of four detection strategies based on its genre profile:

- Grid-based     : House, Tech House, Future House, Bass House, Electro House,
                   Progressive House, Big Room, Techno, Trance, EDM,
                   Indobounce (§19)
- Energy-based   : Breakbeat, Drum & Bass, Jungle, Dubstep, Trap, Hardstyle,
                   UK Garage, Speed Garage, Baile Funk (§23), Jersey Club (§24)
- Vocal-based    : Hip-Hop, R&B, Reggaeton, Dancehall, Afrobeats, Amapiano,
                   Open Format, Pop
- Lokal-Custom   : Bassline Bounce (§20), Hipdut (§25) - start from the Energy
                   baseline but carry lower confidence until enough
                   manual-feedback data is collected (logic_new.md §6.7).

The router is cheap: it keys off the genre profile name / energy_profile hint
(no expensive per-track classification needed for the MVP). Structure detection
then adjusts novelty weighting, boundary snapping and confidence caps per
strategy (logic_new.md §6.4-6.7).
"""

from __future__ import annotations

from enum import Enum

from ..models import GenreProfile

STRATEGY_GRID = "grid"
STRATEGY_ENERGY = "energy"
STRATEGY_VOCAL = "vocal"
STRATEGY_LOCAL = "local_custom"
STRATEGIES = (STRATEGY_GRID, STRATEGY_ENERGY, STRATEGY_VOCAL, STRATEGY_LOCAL)


class DetectionStrategy(Enum):
    GRID = STRATEGY_GRID
    ENERGY = STRATEGY_ENERGY
    VOCAL = STRATEGY_VOCAL
    LOCAL = STRATEGY_LOCAL


# Genre -> strategy baseline (logic_new.md §6.3). Keys are normalized profile
# names (lowercase, "_").
_GRID_GENRES = {
    "house",
    "tech_house",
    "future_house",
    "bass_house",
    "electro_house",
    "progressive_house",
    "big_room",
    "techno",
    "trance",
    "edm",
    "indobounce",
}
_ENERGY_GENRES = {
    "breakbeat",
    "drum_and_bass",
    "jungle",
    "dubstep",
    "trap",
    "hardstyle",
    "uk_garage",
    "speed_garage",
    "baile",
    "jersey_club",
}
_VOCAL_GENRES = {
    "hip_hop",
    "rnb",
    "reggaeton",
    "dancehall",
    "afrobeats",
    "amapiano",
    "open_format",
    "pop",
}
# Lokal-Custom: structurally closest to bass/EDM local music; confidence is
# capped until feedback data accumulates (logic_new.md §6.7).
_LOCAL_GENRES = {
    "bassline_bounce",
    "hipdut",
}


def strategy_for_genre(profile_name: str) -> str:
    """Map a genre profile name to a detection strategy (logic_new.md §6.3)."""
    key = profile_name.strip().lower().replace("-", "_").replace(" ", "_")
    if key in _GRID_GENRES:
        return STRATEGY_GRID
    if key in _ENERGY_GENRES:
        return STRATEGY_ENERGY
    if key in _VOCAL_GENRES:
        return STRATEGY_VOCAL
    if key in _LOCAL_GENRES:
        return STRATEGY_LOCAL
    # Fallback: unknown genre -> flexible strategy.
    return STRATEGY_VOCAL if "vocal" in key or "bounce" in key else STRATEGY_GRID


def strategy_for_profile(profile: GenreProfile | None) -> str:
    """Route a genre profile to a detection strategy.

    Uses the profile's `energy_profile` hint when it names a strategy
    (e.g. ``vocal_based`` -> vocal), otherwise the genre-name table.
    """
    if profile is None:
        return STRATEGY_GRID
    name_key = profile.name.strip().lower().replace("-", "_").replace(" ", "_")
    if name_key in _LOCAL_GENRES:
        return STRATEGY_LOCAL
    hint = (profile.energy_profile or "").lower()
    if "vocal" in hint:
        return STRATEGY_VOCAL
    if "breakbeat" in hint or "2step" in hint or "808" in hint:
        return STRATEGY_ENERGY
    if "bass_bounce" in hint or "climax" in hint or "energy_peak" in hint:
        return STRATEGY_ENERGY
    if hint == "flexible":
        return STRATEGY_VOCAL
    return strategy_for_genre(profile.name)


def confidence_cap(strategy: str) -> float:
    """Maximum cue/section confidence for a strategy (logic_new.md §6.7).

    Lokal-Custom is capped at 0.7 until manual-feedback data raises it;
    every other strategy allows up to 0.99.
    """
    if strategy == STRATEGY_LOCAL:
        return 0.7
    return 0.99