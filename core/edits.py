"""Human review edits for a track.

DJ corrections (move / delete / lock a hot cue) are persisted as a sidecar
JSON file next to the audio file (`<track>.cuepilot.json`). Edits survive
re-analysis: locked cues keep their edited position, deleted cues stay gone.

Before any write, the previous sidecar (and original tag snapshot) is
copied to the per-track backup folder.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any

logger = logging.getLogger(__name__)

# Valid hot-cue slot range per schemas/cueplan.schema.json (slot: 1..8).
MIN_SLOT = 1
MAX_SLOT = 8

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
    """Contents of a sidecar edits file.

    ``corrupt`` is True only when the sidecar file exists but could not
    be parsed/validated (see :func:`load_edits`): in that case ``cues``
    is empty, ``load_error`` explains why, and callers must NOT treat
    the result as "no edits" — the user's edits may be lost, and the
    UI should warn instead of silently re-analysing from scratch.
    """

    schema_version: str = _CURRENT_VERSION
    track_path: str = ""
    profile: str = ""
    bpm: float = 0.0
    key: str = ""
    cues: list[CueEdit] = field(default_factory=list)
    updated_at: str = ""
    app_version: str = "0.1.0"
    corrupt: bool = False
    load_error: str = ""

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
            "corrupt": self.corrupt,
            "loadError": self.load_error,
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
    """Load persisted edits for `track_path`, or None if none exist.

    A corrupt sidecar (unparsable JSON, wrong top-level type, or a cue
    entry that cannot be converted) is NO LONGER swallowed into None:
    it logs a warning and returns an ``EditsFile`` with ``corrupt=True``,
    empty cues and ``load_error`` set, so callers can warn the user
    instead of silently discarding their edits. ``None`` now strictly
    means "no sidecar file exists". See also :func:`load_edits_status`.
    """
    path = sidecar_path(track_path)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"sidecar top-level JSON is {type(data).__name__}, expected object")
        if not isinstance(data.get("cues", []), list):
            raise ValueError("sidecar 'cues' is not a list")
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
    except Exception as exc:  # noqa: BLE001 - any load failure = corrupt sidecar
        logger.warning("Corrupt edits sidecar %s: %s — edits NOT applied.", path, exc)
        return EditsFile(
            track_path=track_path,
            corrupt=True,
            load_error=f"{type(exc).__name__}: {exc}",
        )


def load_edits_status(track_path: str) -> dict[str, Any]:
    """Load edits plus an explicit status dict for UI reporting.

    Returns ``{"edits": EditsFile | None, "status": "ok" | "missing" |
    "corrupt", "error": str}``. This is the preferred entry point for
    callers that need to tell the user a sidecar is damaged.
    """
    edits = load_edits(track_path)
    if edits is None:
        return {"edits": None, "status": "missing", "error": ""}
    if edits.corrupt:
        return {"edits": edits, "status": "corrupt", "error": edits.load_error}
    return {"edits": edits, "status": "ok", "error": ""}


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


def validate_cue_edit(edit: CueEdit, duration: float | None = None) -> tuple[CueEdit | None, dict[str, Any] | None]:
    """Validate one edit against the cueplan schema; clamp its time.

    Policy (documented, notification-driven — never silent):
      - slot outside 1..8 (schemas/cueplan.schema.json) -> REJECTED
        (returned edit is None); a cue outside the schema cannot be
        exported, so it must not be applied at all.
      - time outside [0, duration] when ``duration`` is known -> CLAMPED
        to the nearest bound and the original time is reported.
      - time < 0 when ``duration`` is unknown -> CLAMPED to 0.
      - non-finite time -> REJECTED.
    Returns ``(edit_or_None, notification_or_None)`` where a notification
    is ``{"slot", "action": "rejected"|"clamped", "reason", ...}``.
    The input edit is not mutated; clamping returns a copy.
    """
    import math

    slot = edit.slot
    if not isinstance(slot, int) or isinstance(slot, bool) or not (MIN_SLOT <= slot <= MAX_SLOT):
        return None, {
            "slot": slot,
            "action": "rejected",
            "reason": f"slot {slot!r} outside schema range {MIN_SLOT}..{MAX_SLOT}; edit ignored",
        }
    try:
        t = float(edit.time)
    except (TypeError, ValueError):
        return None, {
            "slot": slot,
            "action": "rejected",
            "reason": f"non-numeric time {edit.time!r}; edit ignored",
        }
    if not math.isfinite(t):
        return None, {
            "slot": slot,
            "action": "rejected",
            "reason": f"non-finite time {edit.time!r}; edit ignored",
        }
    lo, hi = 0.0, duration
    clamped = t
    if clamped < lo:
        clamped = lo
    if hi is not None and clamped > hi:
        clamped = hi
    if clamped != t:
        out = CueEdit(
            slot=edit.slot, time=clamped, label=edit.label, color=edit.color,
            locked=edit.locked, deleted=edit.deleted, type=edit.type, source=edit.source,
        )
        return out, {
            "slot": slot,
            "action": "clamped",
            "reason": f"time {t} outside [0, {hi}]" if hi is not None else f"time {t} below 0",
            "originalTime": t,
            "clampedTime": clamped,
        }
    return edit, None


def apply_edits_to_cues(
    cues: list[Any],
    edits: EditsFile | None,
    duration: float | None = None,
    notifications: list[dict[str, Any]] | None = None,
) -> list[Any]:
    """Lay persisted user edits onto a freshly generated cue list.

    Rules (Genre_pattern.md 68):
      - deleted cues are removed entirely;
      - locked cues keep the edited time/label/color instead of the auto one;
      - unlocked cues keep the auto-generated position but inherit any label/color
        the user typed for that slot;
      - any extra user-added cues are appended.
    `cues` items must expose to_dict()/attributes like core.models.Cue. Returned
    items keep the caller's type (extra cues are created via `_cue_cls`).

    Validation: every edit passes through :func:`validate_cue_edit`
    (slot 1..8 per schemas/cueplan.schema.json; 0 <= time <= duration
    when ``duration`` is given). Invalid edits are rejected and
    out-of-range times are clamped — and EVERY rejection/clamp is
    appended to ``notifications`` (when supplied) so the result never
    silently drops or moves a cue. A corrupt sidecar (``edits.corrupt``)
    applies nothing and reports itself in ``notifications``. Callers
    should pass the track duration (pipeline: ``analysis.duration``)
    for the full time check; without it only slot/negative-time checks
    can run.
    """
    if not edits or not edits.cues:
        return list(cues)
    if getattr(edits, "corrupt", False):
        if notifications is not None:
            notifications.append({
                "slot": None,
                "action": "rejected",
                "reason": f"corrupt edits sidecar not applied: {edits.load_error}",
            })
        return list(cues)

    valid_edits: list[CueEdit] = []
    for edit in edits.cues:
        checked, note = validate_cue_edit(edit, duration=duration)
        if note is not None and notifications is not None:
            notifications.append(note)
        if checked is not None:
            valid_edits.append(checked)

    edited_by_slot: dict[int, CueEdit] = {e.slot: e for e in valid_edits}
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

    # user-added cues not present in auto output (validated edits only)
    for edit in sorted(valid_edits, key=lambda e: e.slot):
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
