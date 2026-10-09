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
values are immediately usable. Hot cues live in Serato's DatabaseV2 (handled by
the future Serato adapter); the cue-plan tag keeps a portable copy.
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


def write_metadata(
    path: str,
    bpm: float,
    key: str,
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
        Result dict with written/backup path and written tags.
    """
    if not os.path.isfile(path):
        return {"ok": False, "error": "file not found"}

    lower = path.lower()
    ext = _ext(lower)
    backup_path = None if dry_run else _backup_file(path)

    try:
        if ext in ("mp3", "aiff", "aif"):
            return _write_id3(path, bpm, key, cues, analysis, dry_run, backup_path)
        if ext == "flac":
            return _write_vorbis(FLAC, path, bpm, key, cues, analysis, dry_run, backup_path)
        if ext == "ogg":
            return _write_vorbis(OggVorbis, path, bpm, key, cues, analysis, dry_run, backup_path)
        if ext == "wav":
            return _write_wav(path, bpm, key, cues, analysis, dry_run, backup_path)
        return {"ok": False, "error": f"unsupported format: {ext or 'none'}"}
    except Exception as exc:  # noqa: BLE001 - surface tag write errors to the UI
        return {"ok": False, "error": str(exc)}


def _written_tags(written: dict[str, Any]) -> dict[str, Any]:
    out = {
        "ok": True,
        "written": True,
        "bpm": written.get("bpm", 0.0),
        "key": written.get("key", ""),
        "cueCount": written.get("cueCount", 0),
    }
    if written.get("backup"):
        out["backup"] = written["backup"]
    return out


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

    if dry_run:
        return {
            "ok": True,
            "written": False,
            "dryRun": True,
            "bpm": bpm,
            "key": key,
            "cueCount": len(cues),
            "tags": ["TBPM", "TKEY", f"TXXX:{CUE_TAG}"],
        }

    try:
        audio = ID3(path)
    except ID3NoHeaderError:
        audio = ID3()
        audio.filename = path

    audio.add(TBPM(encoding=3, text=[f"{bpm:.2f}"]))
    audio.add(TKEY(encoding=3, text=[key]))
    audio.add(TXXX(encoding=3, desc=CUE_TAG, text=[cue_payload]))
    audio.save()
    return _written_tags({"bpm": bpm, "key": key, "cueCount": len(cues), "backup": backup_path})


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

    if dry_run:
        return {
            "ok": True,
            "written": False,
            "dryRun": True,
            "bpm": bpm,
            "key": key,
            "cueCount": len(cues),
            "tags": ["BPM", "KEY", CUE_TAG],
        }

    audio["BPM"] = f"{bpm:.2f}"
    audio["KEY"] = key
    audio[CUE_TAG] = cue_payload
    audio.save()
    return _written_tags({"bpm": bpm, "key": key, "cueCount": len(cues), "backup": backup_path})


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

    if dry_run:
        return {
            "ok": True,
            "written": False,
            "dryRun": True,
            "bpm": bpm,
            "key": key,
            "cueCount": len(cues),
            "tags": ["TBPM", "TKEY", f"TXXX:{CUE_TAG}"],
        }

    if audio.tags is None:
        audio.add_tags()
    audio.tags.add(TBPM(encoding=3, text=[f"{bpm:.2f}"]))
    audio.tags.add(TKEY(encoding=3, text=[key]))
    audio.tags.add(TXXX(encoding=3, desc=CUE_TAG, text=[cue_payload]))
    audio.save()
    return _written_tags({"bpm": bpm, "key": key, "cueCount": len(cues), "backup": backup_path})
