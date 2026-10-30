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
  locked / overwritten on exit. `apply` refuses to write unless
  `check_rekordbox_closed()` passes: no `master.db-wal`/`-shm` files, no
  SQLite lock (proven by a `BEGIN IMMEDIATE` probe, not just a process
  name), and no Rekordbox process detected.
- Hot-cue writes are idempotent per slot: within ONE transaction, any
  existing `DjmdHotCue` row for the same `ContentID` + `CueNumber` is
  deleted before the new row is inserted, so applying twice (or over a
  manually created cue on the same slot) never duplicates rows.
- UUIDs (`DjmdHotCue.UUID`) are deterministic per track+slot
  (`uuid5` over `ContentID:CueNumber`), so re-applying a plan produces the
  same UUID instead of random churn.
- Verification is set-based, not count-based: for every planned cue the
  exact `(CueNumber, CueTiming)` pair must exist exactly once. Extra cues
  on unplanned slots are reported, not mistaken for the plan.
- Track matching normalizes paths (casefold, `\`/`/` unified) and matches
  on FileName + parent-directory suffix, so Windows/macOS/Linux path
  spellings of the same track resolve to the same `ContentID`.
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
    """Return an all-uppercase dashed UUID Rekordbox accepts (display-only)."""
    return str(uuid.uuid4()).upper()


def _uuid_for_cue(content_id: str, cue_number: int) -> str:
    """Deterministic UUID for one track+slot, so re-apply is stable."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"cuepilot:{content_id}:{int(cue_number)}")).upper()


def normalize_path(path: str) -> str:
    """Normalize a track path for comparison: unify separators, casefold."""
    return path.replace("\\", "/").strip().casefold().rstrip("/")


def _path_parts(path: str) -> list[str]:
    return [p for p in normalize_path(path).split("/") if p]


def paths_match(stored_folder: str, stored_file: str, candidate: str) -> bool:
    """True if a DjmdSong row is the same track as ``candidate``.

    Requires the file names to match (case-insensitive) AND either an
    exact normalized-path match or a suffix match on FileName + parent
    directory, so differing drive letters / roots / separator styles do
    not break matching while same-named files in different folders do not
    collide.
    """
    stored_full = normalize_path(f"{stored_folder}/{stored_file}")
    cand = normalize_path(candidate)
    if stored_full == cand:
        return True
    stored_parts, cand_parts = _path_parts(stored_full), _path_parts(cand)
    if not stored_parts or not cand_parts or stored_parts[-1] != cand_parts[-1]:
        return False
    if len(stored_parts) >= 2 and len(cand_parts) >= 2:
        return stored_parts[-2] == cand_parts[-2]
    return False


def is_database_locked(master_db: str) -> tuple[bool, str]:
    """Probe whether ``master_db`` is locked, without relying on process names.

    Returns ``(locked, reason)``. A present ``-wal``/``-shm`` sidecar means
    another connection may still hold the DB; a failed ``BEGIN IMMEDIATE``
    probe proves SQLite itself reports the DB busy/locked.
    """
    if not master_db or not os.path.isfile(master_db):
        return False, "master.db not found"
    for suffix in ("-wal", "-shm"):
        sidecar = master_db + suffix
        if os.path.exists(sidecar):
            return True, f"sidecar present ({os.path.basename(sidecar)}): Rekordbox may still be open"
    try:
        conn = sqlite3.connect(f"file:{master_db}?mode=rw", uri=True, timeout=1)
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("ROLLBACK")
        finally:
            conn.close()
    except sqlite3.OperationalError as exc:
        if "locked" in str(exc).lower() or "busy" in str(exc).lower():
            return True, f"SQLite reports database is locked: {exc}"
        return True, f"SQLite probe failed: {exc}"
    except sqlite3.Error as exc:
        return True, f"SQLite probe failed: {exc}"
    return False, ""


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

    def check_rekordbox_closed(self) -> dict[str, Any]:
        """Explicit, testable closed-condition check (K-4).

        Rekordbox is considered closed only when ALL hold: no ``-wal``/``-shm``
        sidecar, the SQLite lock probe passes, and no Rekordbox process is
        detected. Process-name detection alone is NOT sufficient (the
        process may be named differently / the check may be unavailable),
        so a failing lock probe refuses the write even if no process is seen.
        """
        reasons: list[str] = []
        if self._master_db:
            locked, reason = is_database_locked(self._master_db)
            if locked:
                reasons.append(reason)
        if self._agent_running():
            reasons.append("rekordbox agent is running; close it before applying hot cues")
        return {"closed": not reasons, "reasons": reasons}

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
        """Set-based verification (K-2): exact (CueNumber, CueTiming) pairs.

        For every planned cue, exactly one row with that slot AND timing
        must exist for the track. A mere total count can pass while the
        wrong cues are present (or while duplicates hide a missing cue),
        so counts alone are never accepted. Rows on unplanned slots
        (e.g. manual cues the plan did not touch) are reported as
        ``extraHotCueRows`` but do not fail verification, because apply
        upserts per planned slot only.
        """
        plan = manifest.cue_plan
        track = plan.get("track", {})
        path = track.get("path", "")
        planned = {
            (int(c.get("slot", 1)), int(round(float(c.get("time", 0.0)) * _MS)))
            for c in plan.get("cues", [])
        }
        try:
            content_id = self._resolve_content_id(path)
            if content_id is None:
                return {"ok": False, "backupId": backup_id, "error": f"track not found: {path}"}
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT CueNumber, CueTiming FROM DjmdHotCue WHERE ContentID = ?",
                    (content_id,),
                ).fetchall()
            found_pairs = [(int(r[0]), int(r[1])) for r in rows]
        except Exception as exc:
            return {"ok": False, "backupId": backup_id, "error": str(exc)}
        from collections import Counter

        counts = Counter(found_pairs)
        missing = sorted(planned - set(found_pairs))
        duplicated = sorted(p for p, n in counts.items() if p in planned and n != 1)
        extras = sorted(set(found_pairs) - planned)
        ok = not missing and not duplicated
        return {
            "ok": ok,
            "backupId": backup_id,
            "hotCueRows": len(found_pairs),
            "expected": sorted(planned),
            "found": sorted(set(found_pairs)),
            "missing": missing,
            "duplicated": duplicated,
            "extraHotCueRows": extras,
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
                "detail": (
                    "Requires Rekordbox agent to be closed (no -wal/-shm, no SQLite lock, "
                    "no process). Apply upserts per ContentID+CueNumber in one transaction; "
                    "no beat-grid / memory-point writes (out of MVP scope)."
                ),
            }
        )
        return steps

    def _insert_sql(self, cue: dict[str, Any], content_id: str | None = None) -> str:
        slot = int(cue.get("slot", 1))
        cue_uuid = _uuid_for_cue(content_id, slot) if content_id else _uuid_for_rekordbox()
        return (
            "INSERT INTO DjmdHotCue (UUID, ContentID, CueNumber, CueTiming, CueKind, Comment) "
            f"VALUES ('{cue_uuid}', :contentId, {slot}, "
            f"{int(round(float(cue.get('time', 0.0)) * _MS))}, {CUE_KIND_HOT}, :comment);"
        )

    def _apply(self, cue_plan: dict[str, Any]) -> dict[str, Any]:
        closed = self.check_rekordbox_closed()
        if not closed["closed"]:
            raise BackupError(
                "refusing to write: Rekordbox does not look closed ("
                + "; ".join(closed["reasons"])
                + "); close Rekordbox and try again"
            )

        track = cue_plan.get("track", {})
        path = track.get("path", "")
        content_id = self._resolve_content_id(path)
        if content_id is None:
            raise BackupError(f"track not found in Rekordbox library: {path}")

        inserted = 0
        replaced = 0
        try:
            # ONE transaction: sqlite3's context manager commits on success
            # and rolls back on any error, so a partial upsert cannot persist.
            with self._connect() as conn:
                for cue in cue_plan.get("cues", []):
                    slot = int(cue.get("slot", 1))
                    # Per-slot upsert (K-2): delete any pre-existing row for
                    # this ContentID+CueNumber first, so re-applying or
                    # overwriting a manual cue on the same slot duplicates
                    # nothing. Other slots are left untouched.
                    cur = conn.execute(
                        "DELETE FROM DjmdHotCue WHERE ContentID = ? AND CueNumber = ?",
                        (content_id, slot),
                    )
                    replaced += cur.rowcount or 0
                    cue_type = cue.get("type")
                    label = cue.get("label", "")
                    comment = label if label == cue_type else " | ".join(
                        [str(p) for p in (cue_type, label) if p]
                    )
                    conn.execute(
                        self._insert_sql(cue, content_id=content_id),
                        {
                            "contentId": content_id,
                            "comment": comment,
                        },
                    )
                    inserted += 1
        except sqlite3.Error as exc:
            raise BackupError(f"rekordbox write failed: {exc}") from exc

        return {"steps": [{"action": "upsert-hot-cue", "file": path, "inserted": inserted, "replaced": replaced}]}

    def _resolve_content_id(self, path: str) -> str | None:
        """Match a track by normalized path (FileName + parent suffix, K-4)."""
        if not path:
            return None
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT ContentID, FolderPath, FileName FROM DjmdSong"
                ).fetchall()
            for content_id, folder, filename in rows:
                if paths_match(folder or "", filename or "", path):
                    return str(content_id)
            return None
        except Exception:
            return None