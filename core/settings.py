"""User settings persistence (engine mode, future prefs).

Stored as JSON in the same data dir as the learning DB (`~/.cuepilot/`).
Engine mode selects the analysis engine: "basic" (classic DSP pipeline) or
"intellistem" (Stem Intelligence Engine, SIE.md).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ENGINE_MODES = ("basic", "intellistem", "onnx")
DEFAULT_ENGINE = "basic"

STEM_NAMES = ("vocal", "drums", "bass", "other")
DEFAULT_STEMS = list(STEM_NAMES)


def _data_dir() -> Path:
    override = os.environ.get("CUEPILOT_DATA_DIR", "")
    if override:
        p = Path(override)
    else:
        p = Path.home() / ".cuepilot"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _settings_path() -> Path:
    return _data_dir() / "settings.json"


def load_settings() -> dict:
    try:
        if _settings_path().exists():
            data = json.loads(_settings_path().read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def save_settings(settings: dict) -> dict:
    data = load_settings()
    data.update(settings)
    _settings_path().write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def get_engine_mode() -> str:
    mode = load_settings().get("engine", DEFAULT_ENGINE)
    return mode if mode in ENGINE_MODES else DEFAULT_ENGINE


def set_engine_mode(mode: str) -> str:
    if mode not in ENGINE_MODES:
        mode = DEFAULT_ENGINE
    save_settings({"engine": mode})
    return mode


def get_enabled_stems() -> list[str]:
    """Stems whose sensors participate in SIE analysis (default: all)."""
    stems = load_settings().get("enabled_stems")
    if not isinstance(stems, list) or not stems:
        return list(DEFAULT_STEMS)
    return [s for s in stems if s in STEM_NAMES] or list(DEFAULT_STEMS)


def set_enabled_stems(stems: list[str] | None) -> list[str]:
    """Persist the set of stems used for analysis. Invalid names are dropped."""
    if not stems:
        stems = list(DEFAULT_STEMS)
    valid = [s for s in stems if s in STEM_NAMES]
    if not valid:
        valid = list(DEFAULT_STEMS)
    save_settings({"enabled_stems": valid})
    return valid
