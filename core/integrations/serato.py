"""Serato DJ Pro integration adapter.

HONEST SCOPE (K-1) -- read before promising anything to a user:
- Serato hot cues are NOT supported by this adapter. Serato's Markers /
  DatabaseV2 format is NOT written, NOT guessed, and NOT fabricated here.
- What apply() actually writes: BPM/key + the `CUEPILOT_CUES` custom tag on
  the audio file (via `core.metadata.write_metadata`, which Serato reads on
  import) plus the canonical `.cueplan.json` file next to the track.
- Serato hot cues are NOT yet supported; only BPM/key + .cueplan.json are
  written. Every apply result, plan step, and docstring states this plainly.
  (English: Serato hot cues are NOT yet supported; only BPM/key +
  .cueplan.json are written.)

Integration architecture:
- Direct binary modification of encrypted `DatabaseV2` is avoided for safety.
- Operations are gated behind the backup/verify/rollback lifecycle.
- The library (`_Serato_`) files AND the audio file whose tags are written
  are backed up before apply, so rollback can restore the audio tags.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

from core.integrations.base import LibraryAdapter, BackupError, BACKUP_MANIFEST, BackupManifest
from core.metadata import write_metadata
from core.scanner import SUPPORTED_EXTENSIONS

SERATO_HOT_CUE_NOTE = (
    "Serato hot cues are NOT yet supported; only BPM/key + .cueplan.json are written. "
    "Hot cue Serato BELUM didukung; yang ditulis hanya BPM/key + .cueplan.json."
)


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
    """Applies CuePlans into a Serato DJ Pro library (MVP-safe path).

    Honest scope: Serato hot cues are NOT yet supported; only BPM/key +
    .cueplan.json are written (see module docstring / SERATO_HOT_CUE_NOTE).
    """

    name = "serato"
    version = "0.1.0"

    def __init__(self, library_dir: str | None = None, backups_dir: str | None = None) -> None:
        super().__init__(backups_dir=backups_dir)
        self.library_dir = library_dir or default_serato_root()
        self._current_track_path: str | None = None

    # -- backup: capture the track so its tags can be backed up/rolled back --

    def backup(self, cue_plan: dict[str, Any], metadata_snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
        """Capture the plan's track path, then run the shared backup.

        ``_target_paths`` takes no plan argument, so the track path must be
        stashed here (and in ``_apply``) for the audio file to be included
        among the backed-up targets. Without this, tag writes had no backup
        and rollback could not restore them.
        """
        track = cue_plan.get("track", {}) if isinstance(cue_plan, dict) else {}
        path = track.get("path", "")
        self._current_track_path = path if path and os.path.isfile(path) else None
        return super().backup(cue_plan, metadata_snapshot=metadata_snapshot)

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
        candidates: list[str] = []
        # The audio file itself (K-3): its tags are mutated by _apply, so it
        # must be backed up or rollback cannot restore the original tags.
        if self._current_track_path and os.path.isfile(self._current_track_path):
            candidates.append(self._current_track_path)
        if not self.library_dir or not os.path.isdir(self.library_dir):
            return candidates
        db_v2 = os.path.join(self.library_dir, "database V2")
        if os.path.isfile(db_v2):
            candidates.append(db_v2)
        subcrates = os.path.join(self.library_dir, "Subcrates")
        if os.path.isdir(subcrates):
            candidates.append(subcrates)  # backed up recursively by base.backup
        return candidates

    def _verifies_library_changes(self) -> bool:
        # Tag writes legitimately mutate the audio file, so byte-identity
        # verification would always fail once the audio file is a target.
        # Use the custom check below only when a real audio file was backed
        # up; otherwise (e.g. lifecycle tests with no track file) keep the
        # shared byte-identity verification.
        return bool(
            self._current_track_path
            and os.path.isfile(self._current_track_path)
            and self._current_track_path in self._target_paths()
        )

    def _verify_after_apply(self, backup_id: str, manifest: "BackupManifest") -> dict[str, Any]:
        """Verify the Serato-safe artifacts, not DatabaseV2 bytes.

        Checks the exported .cueplan.json exists, parses, and carries the
        planned cues, and that the audio file still exists. It deliberately
        does NOT claim Serato hot cues were written.
        """
        plan = manifest.cue_plan
        target = self._cueplan_target(plan)
        ok = False
        detail = ""
        try:
            import json

            with open(target, "r", encoding="utf-8") as f:
                exported = json.load(f)
            planned = [(c.get("slot"), c.get("time")) for c in plan.get("cues", [])]
            got = [(c.get("slot"), c.get("time")) for c in exported.get("cues", [])]
            ok = planned == got
            detail = "cueplan cues match" if ok else f"cueplan mismatch: planned {planned}, got {got}"
        except OSError as exc:
            detail = f"cueplan not readable: {exc}"
        except ValueError as exc:
            detail = f"cueplan not valid JSON: {exc}"
        track_path = plan.get("track", {}).get("path", "")
        audio_exists = bool(track_path) and os.path.isfile(track_path)
        return {
            "ok": bool(ok and audio_exists),
            "backupId": backup_id,
            "cueplanFile": target,
            "cueplanOk": ok,
            "audioFileExists": audio_exists,
            "detail": detail,
            "seratoHotCuesSupported": False,
            "note": SERATO_HOT_CUE_NOTE,
        }

    def rollback(self, backup_id: str) -> dict[str, Any]:
        """Restore backed-up targets, plus the audio tags as a fallback.

        Primary path: the shared rollback restores every target, including
        the audio file (now in ``_target_paths``). Fallback: if the track
        audio file was NOT restored that way (e.g. an older backup made
        before audio files were targeted), restore it from the newest
        matching copy in the track's ``.cuepilot-backup`` folder, which
        ``core.metadata.write_metadata`` creates next to the track before
        every tag write.
        """
        result = super().rollback(backup_id)
        backup_dir = self.backups_dir / backup_id
        manifest_path = backup_dir / BACKUP_MANIFEST
        if not manifest_path.is_file():
            return result
        try:
            manifest = BackupManifest.from_file(str(manifest_path))
        except Exception:
            return result
        track_path = manifest.cue_plan.get("track", {}).get("path", "")
        if not track_path or not os.path.isfile(track_path):
            return result
        restored = result.get("restored", [])
        if track_path in restored:
            result["audioRestoredFrom"] = "backup"
            return result
        # Fallback: newest .cuepilot-backup copy for this track stem.
        track = Path(track_path)
        backup_root = track.parent / ".cuepilot-backup"
        candidates = []
        if backup_root.is_dir():
            candidates = sorted(
                (p for p in backup_root.iterdir()
                 if p.is_file() and p.name.startswith(track.stem + ".")),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        if candidates:
            try:
                import shutil

                shutil.copy2(str(candidates[0]), track_path)
                restored = [*restored, track_path]
                result["restored"] = restored
                result["ok"] = True
                result["audioRestoredFrom"] = f".cuepilot-backup:{candidates[0].name}"
                return result
            except OSError as exc:
                result["ok"] = False
                result["error"] = f"audio rollback from .cuepilot-backup failed: {exc}"
                return result
        result["audioRestoredFrom"] = None
        return result

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
        steps.append(
            {
                "action": "serato-hot-cues-unsupported",
                "detail": SERATO_HOT_CUE_NOTE,
            }
        )
        return steps

    def _apply(self, cue_plan: dict[str, Any]) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        errors: list[str] = []
        skipped_fields: list[str] = []

        track = cue_plan.get("track", {})
        path = track.get("path", "")
        if path and os.path.isfile(path):
            self._current_track_path = path
            ext = Path(path).suffix.lower().lstrip(".")
            if ext not in {e.lstrip(".") for e in SUPPORTED_EXTENSIONS}:
                errors.append(f"unsupported audio format: {path}")
            else:
                # Pass raw BPM/key through: core.metadata now guards them
                # (T-6) -- BPM <= 0 / empty key are skipped, never written,
                # and reported in skipped_fields instead of overwriting
                # good tags with 0.00 / "".
                write_result = write_metadata(
                    path,
                    bpm=track.get("bpm"),
                    key=track.get("key"),
                    cues=cue_plan.get("cues", []),
                    analysis={"bpm": track.get("bpm"), "key": track.get("key")},
                )
                skipped_fields.extend(write_result.get("skipped_fields", []))
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
        return {
            "steps": results,
            "seratoHotCuesSupported": False,
            "skipped_fields": skipped_fields,
            "note": SERATO_HOT_CUE_NOTE,
        }

    def _cueplan_target(self, cue_plan: dict[str, Any]) -> str:
        track = cue_plan.get("track", {})
        path = track.get("path", "")
        stem = Path(path).stem if path else "track"
        return str(Path(path).parent / f"{stem}.cueplan.json") if path else os.path.join(os.getcwd(), f"{stem}.cueplan.json")