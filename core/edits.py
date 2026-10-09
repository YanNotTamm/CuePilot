"""Human review edits for a track.

DJ corrections (move / delete / lock a hot cue) are persisted as a sidecar
JSON file next to the audio file (`<track>.cuepilot.json`). Edits survive
re-analysis: locked cues keep their edited position, deleted cues stay gone.

Before any write, the previous sidecar (and original tag snapshot) is
copied to the per-track backup folder.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any

EDITS_EXT = ".cuepilot.json"
BACKUP_DIR_NAME = ".cuepilot-backup"

_CURRENT_VERSION = "1.0"


@dataclass
class CueEdit:
    """One edited cue persisted by the user."""

    slot: int
    time: float
    label: str = ""
    color: str = ""
    locked: bool = False
    deleted: bool = False
    type: str = ""
    source: str = "user"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EditsFile:
    """Contents of a sidecar edits file."""

    schema_version: str = _CURRENT_VERSION
    track_path: str = ""
    profile: str = ""
    bpm: float = 0.0
    key: str = ""
    cues: list[CueEdit] = field(default_factory=list)
    updated_at: str = ""
    app_version: str = "0.1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "trackPath": self.track_path,
            "profile": self.profile,
            "bpm": self.bpm,
            "key": self.key,
            "cues": [c.to_dict() for c in self.cues],
            "updatedAt": self.updated_at,
            "appVersion": self.app_version,
        }


def sidecar_path(track_path: str) -> str:
    """Path of the sidecar edits file for `track_path`."""
    return f"{track_path}{EDITS_EXT}"


def backup_dir(track_path: str) -> str:
    """Backup folder next to the audio file."""
    return os.path.join(os.path.dirname(track_path), BACKUP_DIR_NAME)


def _backup_existing(track_path: str) -> str | None:
    """Copy the current sidecar to the backup folder. Returns backup path or None."""
    src = sidecar_path(track_path)
    if not os.path.isfile(src):
        return None
    os.makedirs(backup_dir(track_path), exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dst = os.path.join(backup_dir(track_path), f"{os.path.basename(src)}.{stamp}.{uuid.uuid4().hex[:6]}")
    try:
        with open(src, "rb") as f_in, open(dst, "wb") as f_out:
            f_out.write(f_in.read())
        return dst
    except OSError:
        return None


def load_edits(track_path: str) -> EditsFile | None:
    """Load persisted edits for `track_path`, or None if none exist."""
    path = sidecar_path(track_path)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        edits = EditsFile(
            schema_version=str(data.get("schemaVersion", _CURRENT_VERSION)),
            track_path=str(data.get("trackPath", track_path)),
            profile=str(data.get("profile", "")),
            bpm=float(data.get("bpm", 0.0) or 0.0),
            key=str(data.get("key", "")),
            cues=[
                CueEdit(
                    slot=int(c.get("slot", 0)),
                    time=float(c.get("time", 0.0)),
                    label=str(c.get("label", "")),
                    color=str(c.get("color", "")),
                    locked=bool(c.get("locked", False)),
                    deleted=bool(c.get("deleted", False)),
                    type=str(c.get("type", "")),
                    source=str(c.get("source", "user")),
                )
                for c in data.get("cues", [])
            ],
            updated_at=str(data.get("updatedAt", "")),
            app_version=str(data.get("appVersion", "0.1.0")),
        )
        return edits
    except Exception:
        return None


def _snapshot_analysis(track_path: str, profile: str, bpm: float, key: str) -> EditsFile:
    """Build an EditsFile capturing the current auto-generated cue plan.

    Used so that a fresh analysis can be layered onto previous user edits.
    `track_path`/`profile`/`bpm`/`key` come from the caller; the cue list is
    reconstructed from the persisted edits if available, otherwise empty.
    """
    prev = load_edits(track_path)
    edits = EditsFile(
        track_path=track_path,
        profile=profile,
        bpm=bpm,
        key=key,
        updated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    if prev:
        edits.cues = prev.cues
    return edits


def apply_edits_to_cues(cues: list[Any], edits: EditsFile | None) -> list[Any]:
    """Lay persisted user edits onto a freshly generated cue list.

    Rules (Genre_pattern.md 68):
      - deleted cues are removed entirely;
      - locked cues keep the edited time/label/color instead of the auto one;
      - unlocked cues keep the auto-generated position but inherit any label/color
        the user typed for that slot;
      - any extra user-added cues are appended.
    `cues` items must expose to_dict()/attributes like core.models.Cue. Returned
    items keep the caller's type (extra cues are created via `_cue_cls`).
    """
    if not edits or not edits.cues:
        return list(cues)

    edited_by_slot: dict[int, CueEdit] = {e.slot: e for e in edits.cues}
    result: list[Any] = []
    seen_slots: set[int] = set()

    for c in cues:
        slot = c.slot
        seen_slots.add(slot)
        edit = edited_by_slot.get(slot)
        if edit and edit.deleted:
            continue
        if edit and edit.locked:
            c.time = edit.time
            if edit.label:
                c.label = edit.label
            if edit.color:
                c.color = edit.color
            c.locked = True
        elif edit:
            if edit.label:
                c.label = edit.label
            if edit.color:
                c.color = edit.color
        result.append(c)

    # user-added cues not present in auto output
    for edit in sorted(edits.cues, key=lambda e: e.slot):
        if edit.deleted or edit.slot in seen_slots:
            continue
        result.append(
            _make_extra_cue(edit)
        )
    result.sort(key=lambda c: c.slot)
    return result


def _make_extra_cue(edit: CueEdit) -> Any:
    """Create a core.models.Cue (or dict fallback) for a user-added cue."""
    try:
        from .models import Cue, CueType

        try:
            ctype = CueType(edit.type) if edit.type else CueType.CUSTOM
        except ValueError:
            ctype = CueType.CUSTOM
        return Cue(
            slot=edit.slot,
            type=ctype,
            time=edit.time,
            label=edit.label or f"Cue {edit.slot}",
            confidence=1.0,
            color=edit.color or "#22c55e",
            locked=edit.locked,
            source="user",
            reason=["user edit"],
        )
    except Exception:
        return {
            "slot": edit.slot,
            "type": edit.type or "CUSTOM",
            "time": edit.time,
            "label": edit.label or f"Cue {edit.slot}",
            "confidence": 1.0,
            "color": edit.color or "#22c55e",
            "locked": edit.locked,
            "source": "user",
            "reason": ["user edit"],
        }


def edits_by_slot_auto_appended(edits: EditsFile, seen_slots: set[int]) -> list[CueEdit]:
    return [e for e in edits.cues if e.slot not in seen_slots]


def persist_edits(
    track_path: str,
    profile: str,
    bpm: float,
    key: str,
    cues: list[Any],
    dry_run: bool = False,
) -> dict[str, Any]:
    """Persist the (possibly edited) cue list as a sidecar file.

    Returns a result dict with the sidecar path, whether a backup was made,
    and a preview of what would change when `dry_run` is True (no write).
    """
    edits = EditsFile(
        schema_version=_CURRENT_VERSION,
        track_path=os.path.abspath(track_path),
        profile=profile,
        bpm=bpm,
        key=key,
        updated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    for c in cues:
        d = c.to_dict() if hasattr(c, "to_dict") else c
        edits.cues.append(
            CueEdit(
                slot=int(d.get("slot", 0)),
                time=float(d.get("time", 0.0)),
                label=str(d.get("label", "")),
                color=str(d.get("color", "")),
                locked=bool(d.get("locked", False)),
                deleted=bool(d.get("deleted", False)),
                type=str(d.get("type", "")),
                source=str(d.get("source", "dsp")),
            )
        )

    out_path = sidecar_path(track_path)
    backup_path = _backup_existing(track_path) if not dry_run else None
    payload = edits.to_dict()

    if dry_run:
        return {
            "path": out_path,
            "backup": backup_path,
            "dryRun": True,
            "written": False,
            "cueCount": len(edits.cues),
            "preview": payload,
        }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return {
        "path": out_path,
        "backup": backup_path,
        "dryRun": False,
        "written": True,
        "cueCount": len(edits.cues),
        "updatedAt": edits.updated_at,
    }
