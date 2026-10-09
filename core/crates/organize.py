"""Smart Crate Auto-Organization.

Suggests Serato Smart Crates from the Gig Readiness Score, genre profile,
energy level and key of every analyzed track. Crates are *proposals* the DJ
approves; applying them goes through the LibraryAdapter lifecycle
(backup/verify/rollback).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Crate:
    """A proposed smart crate definition."""

    name: str
    description: str = ""
    tracks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "tracks": list(self.tracks)}


@dataclass
class CrateOrganization:
    """Full auto-organization result (all proposed crates)."""

    crates: list[Crate] = field(default_factory=list)
    assigned: int = 0
    unassigned: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "crates": [c.to_dict() for c in self.crates],
            "assigned": self.assigned,
            "unassigned": self.unassigned,
        }


def _energy_level(energy: float) -> str:
    if energy >= 0.75:
        return "high"
    if energy >= 0.45:
        return "mid"
    return "low"


def _key_family(key: str) -> str:
    from core.compatibility import key_to_camelot

    camelot = key_to_camelot(key)
    if camelot is None:
        return "any"
    return f"camelot-{camelot}"


def _default_crate_rules() -> list[dict[str, Any]]:
    """Built-in crate rule set."""
    return [
        {
            "name": "Peak Time — 124-132 BPM — High Energy — Ready",
            "description": "Peak-hour weapons: ready-to-play, fast, energetic",
            "bpm_min": 124, "bpm_max": 132,
            "energy_min": 0.75,
            "readiness": "ready",
        },
        {
            "name": "Peak Time — High Energy — Ready",
            "description": "High-energy tracks cleared for gig use",
            "energy_min": 0.75,
            "readiness": "ready",
        },
        {
            "name": "Warm Up — Low Energy",
            "description": "Low-energy openers for early sets",
            "energy_max": 0.45,
        },
        {
            "name": "Needs Review",
            "description": "Tracks flagged for manual checks before a gig",
            "readiness": "needs_review",
        },
        {
            "name": "Not Analyzed",
            "description": "Tracks without a completed analysis yet",
            "readiness": "not_analyzed",
        },
    ]


def _track_readiness(track: dict[str, Any]) -> str:
    readiness = track.get("readiness") or (track.get("gigReadiness") or {}).get("bucket", "")
    return str(readiness)


def _matches_rule(track: dict[str, Any], rule: dict[str, Any]) -> bool:
    bpm = float(track.get("bpm", 0.0) or 0.0)
    energy = float(track.get("energy", 0.5))
    readiness = _track_readiness(track)

    if "bpm_min" in rule and bpm < float(rule["bpm_min"]):
        return False
    if "bpm_max" in rule and bpm > float(rule["bpm_max"]):
        return False
    if "energy_min" in rule and energy < float(rule["energy_min"]):
        return False
    if "energy_max" in rule and energy > float(rule["energy_max"]):
        return False
    if "readiness" in rule and readiness != rule["readiness"]:
        return False
    if rule.get("not_readiness") and readiness == rule["not_readiness"]:
        return False
    return True


def organize_crates(
    tracks: list[dict[str, Any]],
    rules: list[dict[str, Any]] | None = None,
) -> CrateOrganization:
    """Assign every track into proposed smart crates.

    A track may match several crates; each crate lists its matching tracks.
    Tracks that match no rule are reported as unassigned.
    """
    rules = rules if rules is not None else _default_crate_rules()
    crates = [Crate(name=r.get("name", "Crate"), description=r.get("description", "")) for r in rules]
    if not tracks:
        return CrateOrganization(crates=[], assigned=0, unassigned=[])

    assigned: set[str] = set()
    for i, rule in enumerate(rules):
        for track in tracks:
            track_id = str(track.get("id") or track.get("trackId", ""))
            if not track_id:
                continue
            if _matches_rule(track, rule):
                crates[i].tracks.append(track_id)
                assigned.add(track_id)

    unassigned = [
        str(t.get("id") or t.get("trackId", ""))
        for t in tracks
        if (t.get("id") or t.get("trackId")) and str(t.get("id") or t.get("trackId")) not in assigned
    ]

    return CrateOrganization(
        crates=crates,
        assigned=len(assigned),
        unassigned=unassigned,
    )