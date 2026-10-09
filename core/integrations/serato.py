"""Serato DJ Pro integration adapter.

Integration architecture:
- Direct binary modification of encrypted `DatabaseV2` is avoided for safety.
- Operations are gated behind the backup/verify/rollback lifecycle.
- Safe tag and cueplan export workflow:
- Apply the generated CuePlan as portable metadata tags on the audio file
  (BPM/key/cue-plan tag via `core.metadata.write_metadata`), which Serato reads
  on import. The canonical `cueplan.json` is exported alongside as the hand-off
  artifact for the future DatabaseV2 adapter.
- The library (`_Serato_`) files are still backed up before apply so a future
  adapter can safely grow into them without changing the contract.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

from core.integrations.base import LibraryAdapter, BackupError
from core.metadata import write_metadata
from core.scanner import SUPPORTED_EXTENSIONS


def default_serato_root() -> str | None:
    """Best-effort location of the Serato `_Serato_` library folder.

    Windows: `%USERPROFILE%/Music/_Serato_`
    macOS:   `~/Music/_Serato_`
    Overridable via `CUEPILOT_SERATO_DIR`.
    """
    override = os.environ.get("CUEPILOT_SERATO_DIR", "")
    if override:
        return override if os.path.isdir(override) else None

    music = Path.home() / "Music" / "_Serato_"
    if music.is_dir():
        return str(music)
    return None


class SeratoAdapter(LibraryAdapter):
    """Applies CuePlans into a Serato DJ Pro library (MVP-safe path)."""

    name = "serato"
    version = "0.1.0"

    def __init__(self, library_dir: str | None = None, backups_dir: str | None = None) -> None:
        super().__init__(backups_dir=backups_dir)
        self.library_dir = library_dir or default_serato_root()

    # -- discovery -----------------------------------------------------------

    def discover(self) -> dict[str, Any]:
        found = self.library_dir is not None and os.path.isdir(self.library_dir)
        db_v2 = os.path.join(self.library_dir, "database V2") if found else None
        db_exists = db_v2 is not None and os.path.isfile(db_v2)
        return {
            "ok": found,
            "adapter": self.name,
            "libraryDir": self.library_dir,
            "databaseV2": db_v2 if db_exists else None,
            "databaseV2Readable": self._database_readable(db_v2) if db_exists else False,
            "note": (
                "DatabaseV2 is encrypted; MVP applies metadata tags + exports cueplan.json "
                "instead of editing the library database."
            ),
        }

    @staticmethod
    def _database_readable(db_path: str | None) -> bool:
        """True if DatabaseV2 opens as plaintext SQLite (older Serato builds)."""
        if not db_path:
            return False
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            try:
                conn.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone()
            finally:
                conn.close()
            return True
        except Exception:
            return False

    # -- hooks ---------------------------------------------------------------

    def _target_paths(self) -> list[str]:
        if not self.library_dir or not os.path.isdir(self.library_dir):
            return []
        db_v2 = os.path.join(self.library_dir, "database V2")
        candidates = [db_v2] if os.path.isfile(db_v2) else []
        subcrates = os.path.join(self.library_dir, "Subcrates")
        if os.path.isdir(subcrates):
            candidates.append(subcrates)
        return candidates

    def _plan_steps(self, cue_plan: dict[str, Any]) -> list[dict[str, Any]]:
        track = cue_plan.get("track", {})
        path = track.get("path", "")
        steps: list[dict[str, Any]] = []

        if path and os.path.isfile(path):
            steps.append(
                {
                    "action": "write-metadata-tags",
                    "file": path,
                    "bpm": track.get("bpm"),
                    "key": track.get("key"),
                    "cueCount": len(cue_plan.get("cues", [])),
                    "detail": "Writes BPM, key, and the CUEPILOT_CUES tag (Serato reads these on import).",
                }
            )
        else:
            steps.append(
                {
                    "action": "skip-metadata-tags",
                    "file": path or "<missing track.path>",
                    "detail": "Track file not found; metadata tags will not be written.",
                }
            )

        steps.append(
            {
                "action": "export-cueplan",
                "target": self._cueplan_target(cue_plan),
                "detail": "Writes the canonical cueplan.json as the hand-off artifact.",
            }
        )
        steps.append(
            {
                "action": "no-database-edit",
                "detail": "DatabaseV2 is encrypted; non-destructive tag and export workflow used.",
            }
        )
        return steps

    def _apply(self, cue_plan: dict[str, Any]) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        errors: list[str] = []

        track = cue_plan.get("track", {})
        path = track.get("path", "")
        if path and os.path.isfile(path):
            ext = Path(path).suffix.lower().lstrip(".")
            if ext not in {e.lstrip(".") for e in SUPPORTED_EXTENSIONS}:
                errors.append(f"unsupported audio format: {path}")
            else:
                write_result = write_metadata(
                    path,
                    bpm=float(track.get("bpm") or 0.0),
                    key=str(track.get("key") or ""),
                    cues=cue_plan.get("cues", []),
                    analysis={"bpm": track.get("bpm"), "key": track.get("key")},
                )
                if not write_result.get("ok"):
                    errors.append(f"metadata write failed: {write_result.get('error')}")
                results.append(
                    {"action": "write-metadata-tags", "file": path, "result": write_result}
                )

        target = self._cueplan_target(cue_plan)
        try:
            import json

            with open(target, "w", encoding="utf-8") as f:
                json.dump(cue_plan, f, indent=2, ensure_ascii=False)
            results.append({"action": "export-cueplan", "file": target, "result": {"ok": True}})
        except OSError as exc:
            errors.append(f"cueplan export failed: {exc}")

        if errors:
            raise BackupError("; ".join(errors))
        return {"steps": results}

    def _cueplan_target(self, cue_plan: dict[str, Any]) -> str:
        track = cue_plan.get("track", {})
        path = track.get("path", "")
        stem = Path(path).stem if path else "track"
        return str(Path(path).parent / f"{stem}.cueplan.json") if path else os.path.join(os.getcwd(), f"{stem}.cueplan.json")