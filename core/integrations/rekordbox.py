"""Rekordbox (Pioneer DJ / AlphaTheta) integration adapter.

Rekordbox stores hot cues in the SQLite `rekordboxAgentStorage` database:
- macOS: `~/Library/Application Support/Pioneer/rekordboxAgentStorage/`
- Windows: `%LOCALAPPDATA%/Pioneer/rekordboxAgentStorage/`
- `master.db` holds `DjmdSong` (tracks) and `DjmdHotCue` (hot cues).

Unlike Serato's encrypted `DatabaseV2`, the Rekordbox master database is
plaintext SQLite. We still gate every write behind the backup/verify/
rollback lifecycle: never mutate without a fresh backup, and roll back if
verification fails. A `dry_run` describes the SQL without executing it.

Safety notes:
- The agent (Rekordbox itself) must be closed while writing, or the DB may be
  locked / overwritten on exit. `apply(..., require_closed=True)` (default)
  refuses to write if a lock file or the process is detected.
- UUIDs (`DjmdHotCue.UUID`) use the `0000-0000-...` style Rekordbox accepts.
"""

from __future__ import annotations

import os
import re
import shutil
import sqlite3
import subprocess
import uuid
from pathlib import Path
from typing import Any

from core.integrations.base import BackupManifest, LibraryAdapter, BackupError

# Rekordbox stores cue timing in milliseconds.
_MS = 1000.0

# Hot cue kinds: 0 = hot cue, 1 = memory cue, 2 = hot loop... 0 covers the MVP.
CUE_KIND_HOT = 0

# BPM/beat-grid related columns we leave untouched (read-only).
_SONG_COLS = ("ContentID", "TrackTitle", "ArtistName", "PlayCount")


def default_rekordbox_storage() -> str | None:
    """Locate the rekordboxAgentStorage folder (or the legacy `rekordbox` dir)."""
    override = os.environ.get("CUEPILOT_REKORDBOX_DIR", "")
    if override:
        return override if os.path.isdir(override) else None

    env = os.environ
    candidates: list[Path] = []
    if os.name == "nt":
        local = env.get("LOCALAPPDATA", "")
        appdata = env.get("APPDATA", "")
        if local:
            candidates.append(Path(local) / "Pioneer" / "rekordboxAgentStorage")
        if appdata:
            candidates.append(Path(appdata) / "Pioneer" / "rekordboxAgentStorage")
            candidates.append(Path(appdata) / "Pioneer" / "rekordbox")
    else:
        home = Path.home()
        candidates.append(home / "Library" / "Application Support" / "Pioneer" / "rekordboxAgentStorage")
        candidates.append(home / "Library" / "Application Support" / "Pioneer" / "rekordbox")

    for c in candidates:
        if c.is_dir():
            return str(c)
    return None


def _uuid_for_rekordbox() -> str:
    """Return an all-uppercase dashed UUID Rekordbox accepts."""
    return str(uuid.uuid4()).upper()


class RekordboxAdapter(LibraryAdapter):
    """Applies CuePlans into a Rekordbox library via its SQLite database."""

    name = "rekordbox"
    version = "0.1.0"

    def __init__(
        self,
        storage_dir: str | None = None,
        backups_dir: str | None = None,
        master_db: str | None = None,
    ) -> None:
        super().__init__(backups_dir=backups_dir)
        self.storage_dir = storage_dir or default_rekordbox_storage()
        self._master_db = master_db or self._find_master_db(self.storage_dir)

    @staticmethod
    def _find_master_db(storage_dir: str | None) -> str | None:
        if not storage_dir or not os.path.isdir(storage_dir):
            return None
        candidate = os.path.join(storage_dir, "master.db")
        return candidate if os.path.isfile(candidate) else None

    # -- discovery -----------------------------------------------------------

    def discover(self) -> dict[str, Any]:
        db_exists = self._master_db is not None and os.path.isfile(self._master_db)
        return {
            "ok": db_exists,
            "adapter": self.name,
            "storageDir": self.storage_dir,
            "masterDb": self._master_db,
            "tables": self._tables() if db_exists else [],
            "note": (
                "Rekordbox master.db is plaintext SQLite. Writes require the agent to be "
                "closed and always run through backup/verify/rollback."
            ),
        }

    def _tables(self) -> list[str]:
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                ).fetchall()
            return [r[0] for r in rows]
        except Exception:
            return []

    def _connect(self) -> sqlite3.Connection:
        if not self._master_db or not os.path.isfile(self._master_db):
            raise BackupError(f"master.db not found: {self._master_db}")
        return sqlite3.connect(self._master_db)

    def _agent_running(self) -> bool:
        """Best-effort detection that Rekordbox is open (Windows + macOS)."""
        if os.name == "nt":
            try:
                out = subprocess.run(
                    ["tasklist", "/FI", "IMAGENAME eq rekordbox.exe"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                ).stdout
                return "rekordbox.exe" in out
            except Exception:
                return False
        try:
            out = subprocess.run(
                ["pgrep", "-x", "rekordbox"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout
            return out.strip() != ""
        except Exception:
            return False

    # -- hooks ---------------------------------------------------------------

    def _target_paths(self) -> list[str]:
        if self._master_db and os.path.isfile(self._master_db):
            return [self._master_db]
        return []

    def _verifies_library_changes(self) -> bool:
        # master.db is legitimately mutated by apply, so verification confirms
        # the written hot cues instead of byte-identity.
        return True

    def _verify_after_apply(self, backup_id: str, manifest: "BackupManifest") -> dict[str, Any]:
        plan = manifest.cue_plan
        track = plan.get("track", {})
        path = track.get("path", "")
        expected = len(plan.get("cues", []))
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM DjmdHotCue WHERE ContentID = "
                    "(SELECT ContentID FROM DjmdSong WHERE FolderPath || '\\' || FileName = ? OR FolderPath || '/' || FileName = ? LIMIT 1)",
                    (path, path),
                ).fetchone()
            found = int(row[0]) if row else 0
        except Exception:
            found = 0
        return {
            "ok": found == expected,
            "backupId": backup_id,
            "hotCueRows": found,
            "expected": expected,
        }

    def _plan_steps(self, cue_plan: dict[str, Any]) -> list[dict[str, Any]]:
        track = cue_plan.get("track", {})
        path = track.get("path", "")
        steps: list[dict[str, Any]] = []
        for cue in cue_plan.get("cues", []):
            steps.append(
                {
                    "action": "insert-hot-cue",
                    "file": path,
                    "slot": cue.get("slot"),
                    "type": cue.get("type"),
                    "timeSeconds": cue.get("time"),
                    "label": cue.get("label"),
                    "sql": self._insert_sql(cue),
                }
            )
        steps.append(
            {
                "action": "note",
                "detail": "Requires Rekordbox agent to be closed; no beat-grid / memory-point writes (out of MVP scope).",
            }
        )
        return steps

    def _insert_sql(self, cue: dict[str, Any]) -> str:
        return (
            "INSERT INTO DjmdHotCue (UUID, ContentID, CueNumber, CueTiming, CueKind, Comment) "
            f"VALUES ('{_uuid_for_rekordbox()}', :contentId, {int(cue.get('slot', 1))}, "
            f"{int(round(float(cue.get('time', 0.0)) * _MS))}, {CUE_KIND_HOT}, :comment);"
        )

    def _apply(self, cue_plan: dict[str, Any]) -> dict[str, Any]:
        if self._agent_running():
            raise BackupError("rekordbox agent is running; close it before applying hot cues")

        track = cue_plan.get("track", {})
        path = track.get("path", "")
        content_id = self._resolve_content_id(path)
        if content_id is None:
            raise BackupError(f"track not found in Rekordbox library: {path}")

        inserted = 0
        try:
            with self._connect() as conn:
                for cue in cue_plan.get("cues", []):
                    cue_type = cue.get("type")
                    label = cue.get("label", "")
                    comment = label if label == cue_type else " | ".join(
                        [str(p) for p in (cue_type, label) if p]
                    )
                    conn.execute(
                        self._insert_sql(cue),
                        {
                            "contentId": content_id,
                            "comment": comment,
                        },
                    )
                    inserted += 1
        except sqlite3.Error as exc:
            raise BackupError(f"rekordbox write failed: {exc}") from exc

        return {"steps": [{"action": "insert-hot-cue", "file": path, "inserted": inserted}]}

    def _resolve_content_id(self, path: str) -> str | None:
        """Match a track by its full path in `DjmdSong.FolderPath` + `FileName`."""
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT ContentID FROM DjmdSong "
                    "WHERE FolderPath || '\\' || FileName = ? OR FolderPath || '/' || FileName = ? "
                    "LIMIT 1",
                    (path, path),
                ).fetchone()
            return str(row[0]) if row else None
        except Exception:
            return None