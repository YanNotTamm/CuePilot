"""Audio metadata reader and writer (ID3 / Vorbis).

Writes BPM, musical key, and the generated hot-cue plan into the audio file's
native tag container using mutagen. Before any write the original file is
copied to the per-track backup folder.

Tag conventions:
  - MP3 / AIFF (ID3):  TBPM, TKEY, and a custom `CUEPILOT_CUES` TXXX frame
    holding the cue plan as JSON.
  - FLAC / OGG (Vorbis): `BPM`, `KEY` (or `INITIALKEY`), and `CUEPILOT_CUES`.
  - WAV: ID3 chunk when possible (mutagen WAVE supports ID3).

Serato DJ reads BPM and key from these standard tags on import, so the written
values are immediately usable. Serato hot cues themselves are NOT written by
this module (and are not supported by the Serato adapter yet); the cue-plan
tag keeps a portable copy only.

Metadata guard (T-6): BPM <= 0 (or unparseable) and an empty/blank key are
NEVER written. Skipping the tag write leaves any good existing tag untouched
instead of overwriting it with ``0.00`` or an empty string. Every skipped
field is reported in the result dict as ``skipped_fields`` (e.g.
``["bpm", "key"]``) so callers (e.g. the Serato adapter) can surface it.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mutagen.id3 import ID3, TKEY, TBPM, TXXX
from mutagen.flac import FLAC
from mutagen.oggvorbis import OggVorbis

CUE_TAG = "CUEPILOT_CUES"

_BACKUP_DIR = ".cuepilot-backup"


def _backup_file(path: str) -> str | None:
    """Copy the original audio file next to it in a backup folder."""
    src = Path(path)
    backup_root = src.parent / _BACKUP_DIR
    backup_root.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dst = backup_root / f"{src.stem}.{stamp}.{uuid.uuid4().hex[:6]}{src.suffix}"
    try:
        shutil.copy2(src, dst)
        return str(dst)
    except OSError:
        return None


def prune_backups(path: str, keep: int = 3, max_age_days: float | None = None) -> dict[str, Any]:
    """Prune per-track backups in ``.cuepilot-backup`` (R-6).

    Keeps the newest ``keep`` backups for this track and, when
    ``max_age_days`` is given, also deletes backups older than that age
    — but never the newest backup, so a failed latest write can always
    be rolled back. Only files inside the track's own ``.cuepilot-backup``
    folder whose name starts with the track stem are ever deleted; the
    audio file itself is never touched. Returns ``{"kept": [...],
    "deleted": [...]}`` with file names only.
    """
    src = Path(path)
    backup_root = src.parent / _BACKUP_DIR
    if not backup_root.is_dir():
        return {"kept": [], "deleted": []}
    candidates = sorted(
        (p for p in backup_root.iterdir() if p.is_file() and p.name.startswith(src.stem + ".")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return {"kept": [], "deleted": []}
    keep_n = max(1, int(keep))
    kept = candidates[:keep_n]
    to_delete = list(candidates[keep_n:])
    if max_age_days is not None:
        # Never delete the newest backup, even if it is old: `kept`
        # already contains it, and only older candidates are considered.
        cutoff = time.time() - float(max_age_days) * 86400.0
        to_delete = [p for p in to_delete if p.stat().st_mtime < cutoff]
    deleted: list[str] = []
    for p in to_delete:
        try:
            p.unlink()
            deleted.append(p.name)
        except OSError:
            continue
    return {"kept": [p.name for p in kept], "deleted": deleted}


def _cue_payload(cues: list[Any], analysis: dict[str, Any] | None = None) -> str:
    payload = {
        "generator": "cuepilot",
        "version": 1,
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "bpm": round(float((analysis or {}).get("bpm", 0.0)), 2),
        "key": (analysis or {}).get("key", ""),
        "cues": [
            {"slot": c.slot, "type": c.type, "time": round(float(c.time), 3), "label": c.label}
            if hasattr(c, "slot")
            else c
            for c in cues
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=1)


def _ext(lower_path: str) -> str:
    return lower_path.rsplit(".", 1)[-1] if "." in lower_path else ""


def _normalize_bpm_key(bpm: Any, key: Any) -> tuple[float | None, str | None, list[str]]:
    """Validate BPM/key before any tag write.

    Returns ``(bpm_or_None, key_or_None, skipped_fields)``. BPM must parse
    as a number and be > 0; key must be a non-blank string. Invalid values
    are reported as skipped, never written.
    """
    skipped: list[str] = []
    try:
        bpm_value = float(bpm) if bpm is not None else None
    except (TypeError, ValueError):
        bpm_value = None
    if bpm_value is None or bpm_value <= 0:
        skipped.append("bpm")
        bpm_value = None
    key_value = str(key).strip() if key is not None else ""
    if not key_value:
        skipped.append("key")
        key_value = None
    return bpm_value, key_value, skipped


def write_metadata(
    path: str,
    bpm: Any,
    key: Any,
    cues: list[Any],
    analysis: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Write BPM, key, and cue-plan tag into the audio file.

    Args:
        path: audio file path.
        bpm: detected BPM.
        key: detected musical key (e.g. "12A").
        cues: cue objects (with slot/type/time/label) or dicts.
        analysis: optional analysis dict (used to embed BPM/key in the tag).
        dry_run: when True, only report what would change without writing.

    Returns:
        Result dict with written/backup path, written tags, and
        ``skipped_fields`` listing any BPM/key field that was NOT written
        because its value was invalid (BPM <= 0, empty key). Skipped fields
        leave existing tags untouched.
    """
    if not os.path.isfile(path):
        return {"ok": False, "error": "file not found"}

    lower = path.lower()
    ext = _ext(lower)
    bpm_value, key_value, skipped_fields = _normalize_bpm_key(bpm, key)
    backup_path = None if dry_run else _backup_file(path)

    try:
        if ext in ("mp3", "aiff", "aif"):
            result = _write_id3(path, bpm_value, key_value, cues, analysis, dry_run, backup_path)
        elif ext == "flac":
            result = _write_vorbis(FLAC, path, bpm_value, key_value, cues, analysis, dry_run, backup_path)
        elif ext == "ogg":
            result = _write_vorbis(OggVorbis, path, bpm_value, key_value, cues, analysis, dry_run, backup_path)
        elif ext == "wav":
            result = _write_wav(path, bpm_value, key_value, cues, analysis, dry_run, backup_path)
        else:
            return {"ok": False, "error": f"unsupported format: {ext or 'none'}"}
    except Exception as exc:  # noqa: BLE001 - surface tag write errors to the UI
        return {"ok": False, "error": str(exc)}
    result["skipped_fields"] = skipped_fields
    return result


def _written_tags(written: dict[str, Any]) -> dict[str, Any]:
    out = {
        "ok": True,
        "written": True,
        "bpm": written.get("bpm"),
        "key": written.get("key"),
        "cueCount": written.get("cueCount", 0),
        "skipped_fields": written.get("skipped_fields", []),
    }
    if written.get("backup"):
        out["backup"] = written["backup"]
    return out


def _skipped_for(bpm: float | None, key: str | None) -> list[str]:
    skipped = []
    if bpm is None:
        skipped.append("bpm")
    if key is None:
        skipped.append("key")
    return skipped


def _write_id3(
    path: str,
    bpm: float,
    key: str,
    cues: list[Any],
    analysis: dict[str, Any] | None,
    dry_run: bool,
    backup_path: str | None,
) -> dict[str, Any]:
    from mutagen.id3 import ID3NoHeaderError

    cue_payload = _cue_payload(cues, analysis)
    skipped = _skipped_for(bpm, key)
    tags = ((["TBPM"] if bpm is not None else [])
            + (["TKEY"] if key is not None else [])
            + [f"TXXX:{CUE_TAG}"])

    if dry_run:
        return {
            "ok": True,
            "written": False,
            "dryRun": True,
            "bpm": bpm,
            "key": key,
            "cueCount": len(cues),
            "tags": tags,
            "skipped_fields": skipped,
        }

    try:
        audio = ID3(path)
    except ID3NoHeaderError:
        audio = ID3()
        audio.filename = path

    # Guard: only add a frame when its value is valid, so an existing good
    # tag is never overwritten with 0.00 / an empty string.
    if bpm is not None:
        audio.add(TBPM(encoding=3, text=[f"{bpm:.2f}"]))
    if key is not None:
        audio.add(TKEY(encoding=3, text=[key]))
    audio.add(TXXX(encoding=3, desc=CUE_TAG, text=[cue_payload]))
    audio.save()
    return _written_tags({"bpm": bpm, "key": key, "cueCount": len(cues), "backup": backup_path, "skipped_fields": skipped})


def _write_vorbis(
    cls,
    path: str,
    bpm: float,
    key: str,
    cues: list[Any],
    analysis: dict[str, Any] | None,
    dry_run: bool,
    backup_path: str | None,
) -> dict[str, Any]:
    audio = cls(path)
    cue_payload = _cue_payload(cues, analysis)
    skipped = _skipped_for(bpm, key)
    tags = ((["BPM"] if bpm is not None else [])
            + (["KEY"] if key is not None else [])
            + [CUE_TAG])

    if dry_run:
        return {
            "ok": True,
            "written": False,
            "dryRun": True,
            "bpm": bpm,
            "key": key,
            "cueCount": len(cues),
            "tags": tags,
            "skipped_fields": skipped,
        }

    if bpm is not None:
        audio["BPM"] = f"{bpm:.2f}"
    if key is not None:
        audio["KEY"] = key
    audio[CUE_TAG] = cue_payload
    audio.save()
    return _written_tags({"bpm": bpm, "key": key, "cueCount": len(cues), "backup": backup_path, "skipped_fields": skipped})


def _write_wav(
    path: str,
    bpm: float,
    key: str,
    cues: list[Any],
    analysis: dict[str, Any] | None,
    dry_run: bool,
    backup_path: str | None,
) -> dict[str, Any]:
    # WAVE stores metadata in an embedded ID3 chunk (mutagen.wave.WAVE).
    # Files freshly encoded may lack an ID3 chunk entirely, so create one.
    from mutagen.wave import WAVE

    audio = WAVE(path)
    cue_payload = _cue_payload(cues, analysis)
    skipped = _skipped_for(bpm, key)
    tags = ((["TBPM"] if bpm is not None else [])
            + (["TKEY"] if key is not None else [])
            + [f"TXXX:{CUE_TAG}"])

    if dry_run:
        return {
            "ok": True,
            "written": False,
            "dryRun": True,
            "bpm": bpm,
            "key": key,
            "cueCount": len(cues),
            "tags": tags,
            "skipped_fields": skipped,
        }

    if audio.tags is None:
        audio.add_tags()
    if bpm is not None:
        audio.tags.add(TBPM(encoding=3, text=[f"{bpm:.2f}"]))
    if key is not None:
        audio.tags.add(TKEY(encoding=3, text=[key]))
    audio.tags.add(TXXX(encoding=3, desc=CUE_TAG, text=[cue_payload]))
    audio.save()
    return _written_tags({"bpm": bpm, "key": key, "cueCount": len(cues), "backup": backup_path, "skipped_fields": skipped})
