"""Regression tests for the INTEGRASI fix batch (K-1, K-2, K-3, K-4, T-6).

- K-2: Rekordbox apply is idempotent per slot (no duplicates on re-apply or
  over a manual cue on the same slot) and verification is set-based.
- K-4: Rekordbox path matching is normalized (case/separators/suffix) and
  a locked DB (wal/shm sidecar) refuses writes with a clear error.
- K-3: commit retains the backup; base backup/rollback handle directories
  recursively; Serato rollback restores audio tags from .cuepilot-backup.
- K-1: Serato results state plainly that hot cues are NOT supported.
- T-6: metadata writes skip BPM <= 0 / empty key and never overwrite a
  good existing tag with an empty value; skipped fields are reported.
"""

import os
import sqlite3

import pytest

from core.integrations.base import BackupError
from core.integrations.rekordbox import (
    RekordboxAdapter,
    is_database_locked,
    normalize_path,
    paths_match,
)
from core.integrations.serato import SERATO_HOT_CUE_NOTE, SeratoAdapter
from core.metadata import write_metadata


def _cue_plan(path: str, bpm: float = 128.0, key: str = "8A") -> dict:
    return {
        "schemaVersion": "1.0",
        "track": {"path": path, "artist": "A", "title": "T", "duration": 100.0, "bpm": bpm, "key": key},
        "cues": [
            {"slot": 1, "type": "INTRO", "time": 0.0, "label": "INTRO", "confidence": 0.97},
            {"slot": 4, "type": "DROP", "time": 64.0, "label": "DROP", "confidence": 0.94},
        ],
    }


def _seed_rekordbox_db(db_path: str, folder: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE DjmdSong (ContentID TEXT, FolderPath TEXT, FileName TEXT, TrackTitle TEXT)")
    conn.execute(
        "CREATE TABLE DjmdHotCue (UUID TEXT, ContentID TEXT, CueNumber INTEGER, CueTiming INTEGER, CueKind INTEGER, Comment TEXT)"
    )
    conn.execute(
        "INSERT INTO DjmdSong (ContentID, FolderPath, FileName, TrackTitle) "
        "VALUES ('S1', ?, 'track.mp3', 'Track')",
        (folder,),
    )
    conn.commit()
    conn.close()


def _hot_cues(db_path: str):
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT CueNumber, CueTiming FROM DjmdHotCue ORDER BY CueNumber").fetchall()
    conn.close()
    return [(int(r[0]), int(r[1])) for r in rows]


def _rk_adapter(tmp_path, db):
    return RekordboxAdapter(
        storage_dir=str(tmp_path), master_db=str(db), backups_dir=str(tmp_path / "backups")
    )


# --------------------------------------------------------------------------
# K-4: path normalization
# --------------------------------------------------------------------------


def test_normalize_path_unifies_separators_and_case():
    assert normalize_path("C:\\Music\\Track.MP3") == normalize_path("c:/music/track.mp3")


def test_paths_match_suffix_parent_and_file():
    # Same file + parent dir, different roots / separators / case.
    assert paths_match("C:\\Music", "Track.mp3", "/home/user/music/track.mp3") is True
    assert paths_match("/Music", "track.mp3", "/Other/track.mp3") is False  # different parent
    assert paths_match("/Music", "other.mp3", "/Music/track.mp3") is False  # different file


def test_rekordbox_resolves_windows_spelled_folder(tmp_path, monkeypatch):
    db = tmp_path / "master.db"
    _seed_rekordbox_db(str(db), folder="C:\\Music")
    monkeypatch.setattr(RekordboxAdapter, "_agent_running", lambda self: False)
    adapter = _rk_adapter(tmp_path, db)
    plan = _cue_plan("/music/track.mp3")
    # FolderPath "C:\Music" + FileName must match the POSIX-spelled plan path
    # via FileName + parent-dir suffix matching.
    assert adapter._resolve_content_id(plan["track"]["path"]) == "S1"


# --------------------------------------------------------------------------
# K-2: dedup + set-based verification
# --------------------------------------------------------------------------


def test_rekordbox_apply_twice_no_duplicates(tmp_path, monkeypatch):
    db = tmp_path / "master.db"
    _seed_rekordbox_db(str(db), folder=str(tmp_path))
    monkeypatch.setattr(RekordboxAdapter, "_agent_running", lambda self: False)
    adapter = _rk_adapter(tmp_path, db)
    plan = _cue_plan(os.path.join(str(tmp_path), "track.mp3"))

    first = adapter.apply(plan)
    assert first["ok"] is True
    second = adapter.apply(plan)
    assert second["ok"] is True

    rows = _hot_cues(str(db))
    assert rows == [(1, 0), (4, 64000)]
    assert second["verification"]["duplicated"] == []


def test_rekordbox_apply_over_manual_cue_replaces_slot_only(tmp_path, monkeypatch):
    db = tmp_path / "master.db"
    _seed_rekordbox_db(str(db), folder=str(tmp_path))
    # Manual cue on planned slot 1 (wrong timing) + manual cue on slot 9.
    conn = sqlite3.connect(str(db))
    conn.execute(
        "INSERT INTO DjmdHotCue VALUES ('MANUAL-1', 'S1', 1, 9999, 0, 'manual')"
    )
    conn.execute(
        "INSERT INTO DjmdHotCue VALUES ('MANUAL-9', 'S1', 9, 5000, 0, 'manual-9')"
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(RekordboxAdapter, "_agent_running", lambda self: False)
    adapter = _rk_adapter(tmp_path, db)
    plan = _cue_plan(os.path.join(str(tmp_path), "track.mp3"))
    result = adapter.apply(plan)
    assert result["ok"] is True

    rows = _hot_cues(str(db))
    # Slot 1 replaced (no duplicate, plan timing wins); slot 9 untouched.
    assert rows == [(1, 0), (4, 64000), (9, 5000)]
    assert (1, 9999) not in rows
    assert result["verification"]["missing"] == []
    assert result["verification"]["duplicated"] == []


def test_rekordbox_deterministic_uuid_stable_across_applies(tmp_path, monkeypatch):
    db = tmp_path / "master.db"
    _seed_rekordbox_db(str(db), folder=str(tmp_path))
    monkeypatch.setattr(RekordboxAdapter, "_agent_running", lambda self: False)
    adapter = _rk_adapter(tmp_path, db)
    plan = _cue_plan(os.path.join(str(tmp_path), "track.mp3"))
    adapter.apply(plan)
    conn = sqlite3.connect(str(db))
    uuids_first = conn.execute("SELECT UUID FROM DjmdHotCue ORDER BY CueNumber").fetchall()
    conn.close()
    adapter.apply(plan)
    conn = sqlite3.connect(str(db))
    uuids_second = conn.execute("SELECT UUID FROM DjmdHotCue ORDER BY CueNumber").fetchall()
    conn.close()
    assert uuids_first == uuids_second


# --------------------------------------------------------------------------
# K-4: lock refusal
# --------------------------------------------------------------------------


def test_is_database_locked_on_wal_sidecar(tmp_path):
    db = tmp_path / "master.db"
    _seed_rekordbox_db(str(db), folder=str(tmp_path))
    assert is_database_locked(str(db))[0] is False
    (tmp_path / "master.db-wal").write_bytes(b"\x00" * 32)
    locked, reason = is_database_locked(str(db))
    assert locked is True
    assert "wal" in reason.lower()


def test_rekordbox_apply_refuses_when_locked(tmp_path, monkeypatch):
    db = tmp_path / "master.db"
    _seed_rekordbox_db(str(db), folder=str(tmp_path))
    (tmp_path / "master.db-wal").write_bytes(b"\x00" * 32)
    monkeypatch.setattr(RekordboxAdapter, "_agent_running", lambda self: False)
    adapter = _rk_adapter(tmp_path, db)
    plan = _cue_plan(os.path.join(str(tmp_path), "track.mp3"))
    with pytest.raises(BackupError, match="does not look closed|wal"):
        adapter.apply(plan)
    assert _hot_cues(str(db)) == []  # nothing written


def test_check_rekordbox_closed_is_explicit(tmp_path, monkeypatch):
    db = tmp_path / "master.db"
    _seed_rekordbox_db(str(db), folder=str(tmp_path))
    monkeypatch.setattr(RekordboxAdapter, "_agent_running", lambda self: False)
    adapter = _rk_adapter(tmp_path, db)
    assert adapter.check_rekordbox_closed()["closed"] is True
    (tmp_path / "master.db-shm").write_bytes(b"\x00" * 32)
    closed = adapter.check_rekordbox_closed()
    assert closed["closed"] is False
    assert closed["reasons"]


# --------------------------------------------------------------------------
# K-3: backup retention + directory backup/rollback
# --------------------------------------------------------------------------


def test_commit_retains_backup(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "database V2").write_text("serato-db-bytes", encoding="utf-8")
    adapter = SeratoAdapter(library_dir=str(lib), backups_dir=str(tmp_path / "backups"))
    result = adapter.apply(_cue_plan(str(tmp_path / "missing.mp3")))
    # Track file missing -> only cueplan export runs; apply still succeeds
    # and the backup must be retained, not deleted.
    assert result["ok"] is True
    assert result["backupRetained"] is True
    backup_dir = tmp_path / "backups" / result["backupId"]
    assert backup_dir.is_dir()
    assert (backup_dir / "backup-manifest.json").is_file()
    assert result["backupDir"] == str(backup_dir)


def test_backup_and_rollback_handle_subcrates_directory(tmp_path):
    lib = tmp_path / "_Serato_"
    subcrates = lib / "Subcrates"
    subcrates.mkdir(parents=True)
    (subcrates / "crate1.crate").write_text("crate-bytes", encoding="utf-8")
    adapter = SeratoAdapter(library_dir=str(lib), backups_dir=str(tmp_path / "backups"))
    backup_id = adapter.backup(_cue_plan("x.mp3"))["backupId"]
    manifest_dir = tmp_path / "backups" / backup_id
    assert (manifest_dir / "Subcrates" / "crate1.crate").is_file()

    (subcrates / "crate1.crate").write_text("tampered", encoding="utf-8")
    result = adapter.rollback(backup_id)
    assert result["ok"] is True
    assert (subcrates / "crate1.crate").read_text(encoding="utf-8") == "crate-bytes"


def test_serato_rollback_restores_audio_from_cuepilot_backup(tmp_path):
    # Simulate an old backup that did NOT include the audio file, then a
    # .cuepilot-backup copy made by write_metadata next to the track.
    audio = tmp_path / "track.mp3"
    audio.write_bytes(b"ORIGINAL-AUDIO-BYTES")
    backup_root = tmp_path / ".cuepilot-backup"
    backup_root.mkdir()
    (backup_root / "track.20260101-000000.abcdef.mp3").write_bytes(b"ORIGINAL-AUDIO-BYTES")

    lib = tmp_path / "_Serato_"
    lib.mkdir()
    adapter = SeratoAdapter(library_dir=str(lib), backups_dir=str(tmp_path / "backups"))
    backup_id = adapter.backup(_cue_plan(str(audio)))["backupId"]
    # Remove the audio entry from the manifest to mimic a pre-fix backup.
    import json

    manifest_path = tmp_path / "backups" / backup_id / "backup-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["targetFiles"] = [t for t in manifest["targetFiles"] if t["source"] != str(audio)]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    audio.write_bytes(b"TAGGED-AUDIO-BYTES")
    result = adapter.rollback(backup_id)
    assert result["ok"] is True
    assert audio.read_bytes() == b"ORIGINAL-AUDIO-BYTES"
    assert result.get("audioRestoredFrom", "").startswith(".cuepilot-backup:")


# --------------------------------------------------------------------------
# K-1: Serato honesty
# --------------------------------------------------------------------------


def test_serato_dry_run_states_hot_cues_unsupported(tmp_path):
    adapter = SeratoAdapter(library_dir=str(tmp_path / "_Serato_"), backups_dir=str(tmp_path / "backups"))
    result = adapter.apply(_cue_plan(str(tmp_path / "t.mp3")), dry_run=True)
    actions = [s["action"] for s in result["steps"]]
    assert "serato-hot-cues-unsupported" in actions
    unsupported = [s for s in result["steps"] if s["action"] == "serato-hot-cues-unsupported"][0]
    assert "BELUM didukung" in unsupported["detail"]
    assert "NOT yet supported" in unsupported["detail"]


def test_serato_apply_result_states_hot_cues_unsupported(tmp_path):
    adapter = SeratoAdapter(library_dir=str(tmp_path / "_Serato_"), backups_dir=str(tmp_path / "backups"))
    result = adapter.apply(_cue_plan(str(tmp_path / "missing.mp3")))
    assert result["applied"]["seratoHotCuesSupported"] is False
    assert "BELUM didukung" in result["applied"]["note"]
    assert SERATO_HOT_CUE_NOTE in result["applied"]["note"]


# --------------------------------------------------------------------------
# T-6: metadata guard (mutagen ID3 path)
# --------------------------------------------------------------------------


def _write_good_tags(path: str) -> None:
    from mutagen.id3 import ID3, TKEY, TBPM

    tags = ID3()
    tags.add(TBPM(encoding=3, text=["130.00"]))
    tags.add(TKEY(encoding=3, text=["12A"]))
    tags.save(path)


def test_metadata_guard_skips_bad_bpm_and_empty_key(tmp_path):
    from mutagen.id3 import ID3

    mp3 = tmp_path / "t.mp3"
    mp3.write_bytes(b"")
    _write_good_tags(str(mp3))
    result = write_metadata(str(mp3), bpm=0.0, key="", cues=[], analysis=None)
    assert result["ok"] is True
    assert result["skipped_fields"] == ["bpm", "key"]
    tags = ID3(str(mp3))
    # Good existing tags must NOT be overwritten with 0.00 / "".
    assert tags.getall("TBPM")[0].text[0] == "130.00"
    assert tags.getall("TKEY")[0].text[0] == "12A"


def test_metadata_guard_negative_bpm_and_blank_key_dry_run(tmp_path):
    mp3 = tmp_path / "t.mp3"
    mp3.write_bytes(b"")
    result = write_metadata(str(mp3), bpm=-5, key="   ", cues=[], analysis=None, dry_run=True)
    assert result["skipped_fields"] == ["bpm", "key"]
    assert "TBPM" not in result["tags"]
    assert "TKEY" not in result["tags"]


def test_metadata_guard_valid_values_still_written(tmp_path):
    from mutagen.id3 import ID3

    mp3 = tmp_path / "t.mp3"
    mp3.write_bytes(b"")
    result = write_metadata(str(mp3), bpm=128.0, key="8A", cues=[], analysis=None)
    assert result["ok"] is True
    assert result["skipped_fields"] == []
    tags = ID3(str(mp3))
    assert tags.getall("TBPM")[0].text[0] == "128.00"
    assert tags.getall("TKEY")[0].text[0] == "8A"
