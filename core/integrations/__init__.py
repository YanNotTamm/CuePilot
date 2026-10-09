"""DJ library integration adapters.

Contains the isolated adapter layer that hands generated CuePlans to external
DJ software (Serato DJ Pro, Pioneer Rekordbox).

Backup/verify/rollback lifecycle semantics live in `base.LibraryAdapter`.
"""

from core.integrations.base import LibraryAdapter, BackupManifest, BackupError
from core.integrations.serato import SeratoAdapter
from core.integrations.rekordbox import RekordboxAdapter

__all__ = [
    "LibraryAdapter",
    "BackupManifest",
    "BackupError",
    "SeratoAdapter",
    "RekordboxAdapter",
]
