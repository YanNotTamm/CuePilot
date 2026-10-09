"""Base library adapter + backup/verify/rollback safety protocol.

The following lifecycle is enforced for any library mutation:

    Create Backup → Hash Backup → Apply Changes → Verify → Commit

with rollback if verification fails. Backups include the generated
CuePlan, an original-metadata snapshot, target files, timestamp,
application version, and adapter version.

This module defines the shared contract every DJ adapter (Serato, Rekordbox)
implements. Direct encrypted-database manipulation is deliberately avoided for data safety.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core import __version__ as APP_VERSION

BACKUP_MANIFEST = "backup-manifest.json"


class BackupError(RuntimeError):
    """Raised when a backup/apply/verify/rollback step cannot be completed safely."""


def sha256_file(path: str, chunk: int = 1024 * 1024) -> str:
    """SHA-256 of a file's contents (streaming, large audio/db files safe)."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while block := f.read(chunk):
                h.update(block)
    except OSError:
        return ""
    return h.hexdigest()


@dataclass
class BackupManifest:
    """Immutable record of a library backup."""

    backup_id: str
    adapter: str
    adapter_version: str
    app_version: str
    created_at: str
    target_files: list[dict[str, Any]] = field(default_factory=list)
    cue_plan: dict[str, Any] = field(default_factory=dict)
    metadata_snapshot: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    @classmethod
    def from_file(cls, path: str) -> "BackupManifest":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        field_map = {
            "backupId": "backup_id",
            "adapterVersion": "adapter_version",
            "appVersion": "app_version",
            "createdAt": "created_at",
            "targetFiles": "target_files",
            "cuePlan": "cue_plan",
            "metadataSnapshot": "metadata_snapshot",
        }
        kwargs = {field_map.get(k, k): v for k, v in data.items()}
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backupId": self.backup_id,
            "adapter": self.adapter,
            "adapterVersion": self.adapter_version,
            "appVersion": self.app_version,
            "createdAt": self.created_at,
            "targetFiles": self.target_files,
            "cuePlan": self.cue_plan,
            "metadataSnapshot": self.metadata_snapshot,
            "notes": self.notes,
        }


class LibraryAdapter(ABC):
    """Contract for applying a CuePlan into an external DJ library.

    Subclasses implement `discover`, `_plan_steps`, `_apply`, and `_rollback`.
    The generic lifecycle (`backup` → `apply` → `verify` → `commit`/`rollback`)
    is enforced here so every target gets the same safety guarantees.
    """

    name: str = "adapter"
    version: str = "0.1.0"

    def __init__(self, backups_dir: str | None = None) -> None:
        override = backups_dir or os.environ.get("CUEPILOT_BACKUPS_DIR", "")
        self.backups_dir = Path(override) if override else Path.home() / ".cuepilot" / "backups"
        self.backups_dir.mkdir(parents=True, exist_ok=True)

    # -- discovery -----------------------------------------------------------

    @abstractmethod
    def discover(self) -> dict[str, Any]:
        """Locate the target library and return a status dict."""

    # -- lifecycle -----------------------------------------------------------

    def backup(self, cue_plan: dict[str, Any], metadata_snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
        """Create a timestamped backup with the CuePlan + target-file hashes."""
        backup_id = f"{self.name}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        backup_dir = self.backups_dir / backup_id
        backup_dir.mkdir(parents=True, exist_ok=True)

        target_files: list[dict[str, Any]] = []
        for src in self._target_paths():
            if not os.path.isfile(src):
                continue
            dst = backup_dir / os.path.basename(src)
            try:
                shutil.copy2(src, dst)
            except OSError as exc:
                raise BackupError(f"failed to back up {src}: {exc}") from exc
            target_files.append(
                {"source": src, "backup": str(dst), "sha256": sha256_file(str(dst))}
            )

        manifest = BackupManifest(
            backup_id=backup_id,
            adapter=self.name,
            adapter_version=self.version,
            app_version=APP_VERSION,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            target_files=target_files,
            cue_plan=cue_plan,
            metadata_snapshot=metadata_snapshot or {},
            notes=f"{self.name} pre-apply backup",
        )
        manifest_path = backup_dir / BACKUP_MANIFEST
        manifest_path.write_text(
            json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return {"ok": True, "backupId": backup_id, "backupDir": str(backup_dir), "manifest": manifest.to_dict()}

    def apply(
        self,
        cue_plan: dict[str, Any],
        *,
        dry_run: bool = False,
        backup_id: str | None = None,
        metadata_snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the full safe lifecycle for a CuePlan.

        When `dry_run` is True only `_plan_steps` is executed (no writes,
        no backup). Otherwise: backup → apply → verify → commit.
        """
        if dry_run:
            steps = self._plan_steps(cue_plan)
            return {
                "ok": True,
                "dryRun": True,
                "adapter": self.name,
                "adapterVersion": self.version,
                "steps": steps,
            }

        if backup_id is None:
            created = self.backup(cue_plan, metadata_snapshot=metadata_snapshot)
            backup_id = created["backupId"]

        manifest_path = self.backups_dir / backup_id / BACKUP_MANIFEST
        if not manifest_path.is_file():
            raise BackupError(f"backup {backup_id} not found; refusing to apply without a backup")

        applied = self._apply(cue_plan)
        verification = self.verify(backup_id)

        if not verification.get("ok"):
            rollback = self.rollback(backup_id)
            return {
                "ok": False,
                "applied": applied,
                "verification": verification,
                "rolledBack": rollback,
            }

        self._commit(backup_id)
        return {"ok": True, "backupId": backup_id, "applied": applied, "verification": verification}

    def verify(self, backup_id: str) -> dict[str, Any]:
        """Re-hash backed-up files and confirm the library still matches.

        Subclasses may override `_verify_after_apply` for targets where the
        mutation legitimately changes a library file (e.g. Rekordbox master.db),
        in which case "verification" means confirming the write, not byte-identity.
        """
        backup_dir = self.backups_dir / backup_id
        manifest_path = backup_dir / BACKUP_MANIFEST
        if not manifest_path.is_file():
            return {"ok": False, "error": f"no manifest for backup {backup_id}"}

        manifest = BackupManifest.from_file(str(manifest_path))

        if self._verifies_library_changes():
            return self._verify_after_apply(backup_id, manifest)

        checks: list[dict[str, Any]] = []
        for entry in manifest.target_files:
            source, expected = entry["source"], entry["sha256"]
            current = sha256_file(source)
            checks.append(
                {
                    "source": source,
                    "expectedSha256": expected,
                    "actualSha256": current,
                    "match": current == expected,
                }
            )
        ok = all(c["match"] for c in checks)
        return {"ok": ok, "backupId": backup_id, "checks": checks}

    def _verifies_library_changes(self) -> bool:
        """True when the target is mutated in place by apply (no byte-identity check)."""
        return False

    def _verify_after_apply(self, backup_id: str, manifest: "BackupManifest") -> dict[str, Any]:
        raise NotImplementedError

    def rollback(self, backup_id: str) -> dict[str, Any]:
        """Restore every backed-up target file from the backup copy."""
        backup_dir = self.backups_dir / backup_id
        manifest_path = backup_dir / BACKUP_MANIFEST
        if not manifest_path.is_file():
            return {"ok": False, "error": f"no manifest for backup {backup_id}"}

        manifest = BackupManifest.from_file(str(manifest_path))
        restored: list[str] = []
        for entry in manifest.target_files:
            backup_path = entry["backup"]
            if not os.path.isfile(backup_path):
                continue
            try:
                shutil.copy2(backup_path, entry["source"])
                restored.append(entry["source"])
            except OSError as exc:
                return {"ok": False, "error": f"rollback failed for {entry['source']}: {exc}", "restored": restored}

        return {"ok": True, "backupId": backup_id, "restored": restored}

    # -- hooks ---------------------------------------------------------------

    @abstractmethod
    def _target_paths(self) -> list[str]:
        """Absolute paths of library files that must be backed up before apply."""

    @abstractmethod
    def _plan_steps(self, cue_plan: dict[str, Any]) -> list[dict[str, Any]]:
        """Describe (without executing) the mutations an apply would perform."""

    @abstractmethod
    def _apply(self, cue_plan: dict[str, Any]) -> dict[str, Any]:
        """Execute the mutations described by `_plan_steps`."""

    def _commit(self, backup_id: str) -> None:
        """Mark a backup as committed (e.g. remove backup dir)."""
        shutil.rmtree(self.backups_dir / backup_id, ignore_errors=True)