"""Canonical CuePlan export.

Exports a CuePlan to the versioned canonical JSON format consumed by
integrations (Serato and Rekordbox adapters) and validated against the JSON Schema in
`schemas/cueplan.schema.json`.
"""

from __future__ import annotations

import json
import os
from typing import Any

from .models import CuePlan

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "..", "schemas", "cueplan.schema.json")


def cue_plan_to_dict(plan: CuePlan, track: dict[str, Any] | None = None) -> dict[str, Any]:
    """Serialize a CuePlan to the canonical dict (Section 22 format).

    Includes the strategy/genre provenance block (logic_new.md §7):
    ``strategyUsed`` / ``genreDetected`` / ``genreConfidence``. The `track`
    block (if given) is merged in; `plan.track` takes precedence when set.
    """
    payload = plan.to_dict()
    if track:
        merged = dict(track)
        merged.update(payload.get("track") or {})
        payload["track"] = merged
    return payload


def export_cue_plan(
    plan: CuePlan,
    output_path: str,
    track: dict[str, Any] | None = None,
) -> str:
    """Write the canonical JSON to `output_path`. Returns the path."""
    payload = cue_plan_to_dict(plan, track=track)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return output_path


def cue_plan_to_json(plan: CuePlan, track: dict[str, Any] | None = None) -> str:
    """Serialize a CuePlan to a canonical JSON string."""
    return json.dumps(cue_plan_to_dict(plan, track=track), indent=2, ensure_ascii=False)


def build_track_block(track) -> dict[str, Any]:
    """Build the `track` block for export from a Track model (Section 22)."""
    return {
        "path": track.path.replace("\\", "/"),
        "artist": track.artist,
        "title": track.title,
        "duration": round(track.duration, 2),
        "bpm": 0.0,
        "key": "",
    }


def validate_cue_plan_dict(payload: dict[str, Any]) -> list[str]:
    """Validate a canonical payload against the JSON Schema.

    Returns a list of human-readable errors; empty list means valid.
    """
    import jsonschema

    errors: list[str] = []
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)
    validator = jsonschema.Draft7Validator(schema)
    for err in sorted(validator.iter_errors(payload), key=lambda e: list(e.path)):
        errors.append(f"{'/'.join(str(p) for p in err.path)}: {err.message}")
    return errors
