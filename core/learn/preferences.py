"""DJ preference profiles and adaptive cue scoring.

Extends the correction-tracking store (`core.learn.store`) with two more
personalization pieces:

1. **Preference profiles** — a named, persisted recipe of per-genre cue weights
   and slot strategy that a DJ can save and switch between. Stored as JSON in
   the CuePilot data dir (`preferences.json`).
2. **Adaptive scoring** — derives per-genre cue weights from the DJ's
   accumulated corrections. When a slot is consistently used for a role (e.g.
   slot 3 is always a VOCAL), the weights bias scoring toward that role, so
   future auto-cues match the DJ's instincts instead of the generic default.

Both are layered on top of the learned `position_ratio` priors already applied
by `apply_learned_patterns` (correction tracking).
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from .store import MIN_SAMPLES, LearningStore, default_data_dir

# Which sub-scores the adaptive layer is allowed to nudge. Role-slot pressure is
# applied to the most semantically-relevant sub-score (Section 9 formula).
_ROLE_TO_SUBSCORE: dict[str, str] = {
    "VOCAL": "vocal_onset",
    "HOOK": "vocal_onset",
    "MELODY": "vocal_onset",
    "CHORUS": "vocal_onset",
    "DROP": "energy_change",
    "CLIMAX": "energy_change",
    "GROOVE": "rhythm_change",
    "PRE_GROOVE": "rhythm_change",
    "BUILD": "spectral_change",
    "TRANSITION": "spectral_change",
    "BREAKDOWN": "energy_change",
}

ADAPT_MAX_DELTA = 0.12
"""Max absolute weight adjustment per slot from adaptive scoring (keeps the
default weights dominant so learned quirks never overwhelm DSP evidence)."""


def preferences_path() -> Path:
    return default_data_dir() / "preferences.json"


class PreferenceProfile:
    """A named set of per-genre cue weights + slot strategy overrides."""

    def __init__(
        self,
        name: str,
        weights: dict[str, dict[str, float]] | None = None,
        slot_strategy: dict[str, dict[int, str]] | None = None,
    ) -> None:
        self.name = name
        self.weights = weights or {}
        self.slot_strategy = slot_strategy or {}

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "weights": self.weights, "slotStrategy": self.slot_strategy}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreferenceProfile":
        return cls(
            name=str(data.get("name", "default")),
            weights=data.get("weights", {}) or {},
            slot_strategy={
                str(g): {int(k): v for k, v in (m or {}).items()}
                for g, m in (data.get("slotStrategy", {}) or {}).items()
            },
        )


class PreferenceStore:
    """Thread-safe JSON store for DJ preference profiles."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or preferences_path())
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"profiles": {}}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {"profiles": {}}

    def _save(self, data: dict[str, Any]) -> None:
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def list_profiles(self) -> list[PreferenceProfile]:
        with self._lock:
            raw = self._load().get("profiles", {})
        return [PreferenceProfile.from_dict(p) for p in raw.values()]

    def get(self, name: str) -> PreferenceProfile | None:
        with self._lock:
            raw = self._load().get("profiles", {}).get(name)
        return PreferenceProfile.from_dict(raw) if raw else None

    def save(self, profile: PreferenceProfile) -> PreferenceProfile:
        with self._lock:
            data = self._load()
            data.setdefault("profiles", {})[profile.name] = profile.to_dict()
            self._save(data)
        return profile

    def delete(self, name: str) -> bool:
        with self._lock:
            data = self._load()
            removed = data.get("profiles", {}).pop(name, None) is not None
            if removed:
                self._save(data)
        return removed


_default_store: PreferenceStore | None = None
_store_lock = threading.Lock()


def get_preference_store(path: str | Path | None = None) -> PreferenceStore:
    """Module-level singleton preference store."""
    global _default_store
    if path is not None:
        return PreferenceStore(path)
    with _store_lock:
        if _default_store is None:
            _default_store = PreferenceStore()
        return _default_store


def _role_pressure(store: LearningStore, genre: str) -> dict[str, float]:
    """Per-sub-score pressure derived from the DJ's correction patterns.

    For each slot with a trusted pattern, add pressure toward the sub-score
    matching the role the DJ consistently assigns to that slot. Repeatedly
    re-rolling a slot as VOCAL raises `vocal_onset` weight for that genre.
    """
    pressure: dict[str, float] = {}
    patterns = store.patterns_for(genre)
    for pat in patterns:
        if not pat.get("trusted"):
            continue
        roles = pat.get("roles", [])
        if not roles:
            continue
        role = max(roles, key=roles.count)
        sub = _ROLE_TO_SUBSCORE.get(role)
        if sub:
            pressure[sub] = pressure.get(sub, 0.0) + 1.0
    total = sum(pressure.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in pressure.items()}


def adaptive_weights(
    genre: str,
    base_weights: dict[str, float] | None = None,
    store: LearningStore | None = None,
    min_samples: int = MIN_SAMPLES,
) -> dict[str, float]:
    """Compute genre cue weights blended from base + learned pressure.

    The learned pressure is redistributed within `ADAPT_MAX_DELTA` so the base
    formula stays authoritative (weights configurable, tunable
    per dataset). Slots with fewer than `min_samples` corrections contribute
    nothing.
    """
    weights = dict(base_weights or {})
    if not weights:
        return weights
    store = store or LearningStore()
    pressure = _role_pressure(store, genre)
    if not pressure:
        return weights

    # Convert pressure (sums to 1.0) into a bounded delta proportional to the
    # number of sub-scores it spans, capped by ADAPT_MAX_DELTA.
    span = max(1, len(pressure))
    delta_per = ADAPT_MAX_DELTA / span
    for sub, p in pressure.items():
        if sub not in weights:
            continue
        weights[sub] = min(1.0, weights.get(sub, 0.0) + delta_per * p)

    # Renormalize so weights still sum to ~1.0 (keeps total score comparable).
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    return weights


def apply_preference(
    profile: PreferenceProfile,
    genre: str,
    base_weights: dict[str, float],
) -> dict[str, float]:
    """Apply a named preference profile's per-genre weight overrides."""
    weights = dict(base_weights)
    overrides = profile.weights.get(genre, {})
    for k, v in overrides.items():
        if k in weights:
            weights[k] = float(v)
    return weights


def slot_strategy_for(
    profile: PreferenceProfile | None,
    genre: str,
    base: dict[int, str],
) -> dict[int, str]:
    """Return the slot strategy, preferring the preference profile override."""
    if profile is None:
        return dict(base)
    overrides = profile.slot_strategy.get(genre)
    if not overrides:
        return dict(base)
    merged = dict(base)
    for slot, role in overrides.items():
        merged[int(slot)] = role
    return merged