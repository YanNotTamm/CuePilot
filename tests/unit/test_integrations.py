"""Tests for library integration adapters.

Covers the shared backup/verify/rollback lifecycle plus Serato (tag/export path)
and Rekordbox (SQLite DjmdHotCue path) adapter specifics.
"""

import json
import os
import sqlite3

import pytest

from core.integrations.base import BackupError
from core.integrations.serato import SeratoAdapter
from core.integrations.rekordbox import RekordboxAdapter


def _cue_plan(path: str) -> dict:
    return {
        "schemaVersion": "1.0",
        "track": {"path": path, "artist": "A", "title": "T", "duration": 100.0, "bpm": 128.0, "key": "8A"},
        "cues": [
            {"slot": 1, "type": "INTRO", "time": 0.0, "label": "INTRO", "confidence": 0.97},
            {"slot": 4, "type": "DROP", "time": 64.0, "label": "DROP", "confidence": 0.94},
        ],
    }


# --------------------------------------------------------------------------
# Shared lifecycle
# --------------------------------------------------------------------------


class _FakeAdapter(SeratoAdapter):
    """Minimal concrete adapter with no external dependencies for lifecycle tests."""

    name = "fake"
    version = "1.2.3"

    def _apply(self, cue_plan):  # noqa: ANN001
        return {"steps": []}


def test_backup_creates_manifest_with_required_fields(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    db = lib / "database V2"
    db.write_text("serato-db-bytes", encoding="utf-8")

    adapter = _FakeAdapter(library_dir=str(lib), backups_dir=str(tmp_path / "backups"))
    result = adapter.backup(_cue_plan("x.mp3"), metadata_snapshot={"bpm": 128.0})

    assert result["ok"] is True
    manifest = result["manifest"]
    for key in ("backupId", "adapter", "adapterVersion", "appVersion", "createdAt",
                "targetFiles", "cuePlan", "metadataSnapshot", "notes"):
        assert key in manifest
    assert manifest["adapterVersion"] == "1.2.3"
    assert manifest["cuePlan"]["track"]["bpm"] == 128.0
    assert manifest["metadataSnapshot"]["bpm"] == 128.0
    assert len(manifest["targetFiles"]) == 1
    assert manifest["targetFiles"][0]["sha256"]


def test_verify_detects_modified_library(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    db = lib / "database V2"
    db.write_text("original-bytes", encoding="utf-8")

    adapter = _FakeAdapter(library_dir=str(lib), backups_dir=str(tmp_path / "backups"))
    backup_id = adapter.backup(_cue_plan("x.mp3"))["backupId"]

    check = adapter.verify(backup_id)
    assert check["ok"] is True

    db.write_text("tampered-bytes", encoding="utf-8")
    check = adapter.verify(backup_id)
    assert check["ok"] is False
    assert any(c["match"] is False for c in check["checks"])


def test_rollback_restores_original_bytes(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    db = lib / "database V2"
    db.write_text("original-bytes", encoding="utf-8")

    adapter = _FakeAdapter(library_dir=str(lib), backups_dir=str(tmp_path / "backups"))
    backup_id = adapter.backup(_cue_plan("x.mp3"))["backupId"]

    db.write_text("tampered-bytes", encoding="utf-8")
    result = adapter.rollback(backup_id)
    assert result["ok"] is True
    assert db.read_text(encoding="utf-8") == "original-bytes"


def test_apply_without_backup_refuses(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    adapter = _FakeAdapter(library_dir=str(lib), backups_dir=str(tmp_path / "backups"))
    with pytest.raises(BackupError, match="refusing to apply without a backup"):
        adapter.apply(_cue_plan("x.mp3"), backup_id="missing")


# --------------------------------------------------------------------------
# Serato adapter
# --------------------------------------------------------------------------


def test_serato_discover_missing_library(tmp_path):
    adapter = SeratoAdapter(library_dir=str(tmp_path / "nope"), backups_dir=str(tmp_path / "backups"))
    info = adapter.discover()
    assert info["ok"] is False
    assert "DatabaseV2 is encrypted" in info["note"]


def test_serato_discover_finds_database(tmp_path):
    lib = tmp_path / "_Serato_"
    lib.mkdir()
    (lib / "database V2").write_text("bytes", encoding="utf-8")

    adapter = SeratoAdapter(library_dir=str(lib), backups_dir=str(tmp_path / "backups"))
    info = adapter.discover()
    assert info["ok"] is True
    assert info["databaseV2"] == str(lib / "database V2")
    assert info["databaseV2Readable"] is False  # random bytes aren't SQLite


def test_serato_dry_run_describes_steps(tmp_path):
    audio = tmp_path / "t.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 40)

    adapter = SeratoAdapter(library_dir=str(tmp_path / "_Serato_"), backups_dir=str(tmp_path / "backups"))
    result = adapter.apply(_cue_plan(str(audio)), dry_run=True)

    assert result["dryRun"] is True
    actions = [s["action"] for s in result["steps"]]
    assert "write-metadata-tags" in actions
    assert "export-cueplan" in actions
    assert "no-database-edit" in actions


def test_serato_apply_writes_tags_and_cueplan(tmp_path):
    import soundfile as sf
    import numpy as np

    wav = tmp_path / "t.wav"
    sf.write(str(wav), np.zeros(22050, dtype=np.float32), 22050, subtype="PCM_16")

    adapter = SeratoAdapter(library_dir=str(tmp_path / "_Serato_"), backups_dir=str(tmp_path / "backups"))
    result = adapter.apply(_cue_plan(str(wav)))

    assert result["ok"] is True
    assert result["verification"]["ok"] is True
    assert (tmp_path / "t.cueplan.json").is_file()

    payload = json.loads((tmp_path / "t.cueplan.json").read_text(encoding="utf-8"))
    assert payload["cues"][1]["time"] == 64.0

    from mutagen.wave import WAVE

    tags = WAVE(str(wav)).tags
    assert tags is not None
    tbpm = tags.getall("TBPM")
    assert tbpm and tbpm[0].text[0] == "128.00"


def test_serato_apply_fails_on_unsupported_extension(tmp_path):
    mp4 = tmp_path / "t.mp4"
    mp4.write_bytes(b"\x00" * 64)

    adapter = SeratoAdapter(library_dir=str(tmp_path / "_Serato_"), backups_dir=str(tmp_path / "backups"))
    with pytest.raises(BackupError, match="unsupported audio format"):
        adapter.apply(_cue_plan(str(mp4)))


# --------------------------------------------------------------------------
# Rekordbox adapter
# --------------------------------------------------------------------------


def _seed_rekordbox_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE DjmdSong (ContentID TEXT, FolderPath TEXT, FileName TEXT, TrackTitle TEXT)")
    conn.execute(
        "CREATE TABLE DjmdHotCue (UUID TEXT, ContentID TEXT, CueNumber INTEGER, CueTiming INTEGER, CueKind INTEGER, Comment TEXT)"
    )
    conn.execute(
        "INSERT INTO DjmdSong (ContentID, FolderPath, FileName, TrackTitle) "
        "VALUES ('S1', ?, 'track.mp3', 'Track')",
        (os.path.dirname(db_path).replace("/", "\\"),),
    )
    conn.commit()
    conn.close()


def test_rekordbox_discover_finds_tables(tmp_path):
    db = tmp_path / "master.db"
    _seed_rekordbox_db(str(db))

    adapter = RekordboxAdapter(
        storage_dir=str(tmp_path), master_db=str(db), backups_dir=str(tmp_path / "backups")
    )
    info = adapter.discover()
    assert info["ok"] is True
    assert "DjmdSong" in info["tables"]
    assert "DjmdHotCue" in info["tables"]


def test_rekordbox_dry_run_lists_insert_sql(tmp_path):
    db = tmp_path / "master.db"
    _seed_rekordbox_db(str(db))

    plan = _cue_plan(os.path.join(str(tmp_path), "track.mp3"))
    adapter = RekordboxAdapter(
        storage_dir=str(tmp_path), master_db=str(db), backups_dir=str(tmp_path / "backups")
    )
    result = adapter.apply(plan, dry_run=True)
    inserts = [s for s in result["steps"] if s["action"] == "insert-hot-cue"]
    assert len(inserts) == 2
    assert "INSERT INTO DjmdHotCue" in inserts[0]["sql"]
    assert "CueTiming" in inserts[0]["sql"]


def test_rekordbox_apply_writes_hot_cues(tmp_path, monkeypatch):
    db = tmp_path / "master.db"
    _seed_rekordbox_db(str(db))
    monkeypatch.setattr(RekordboxAdapter, "_agent_running", lambda self: False)

    plan = _cue_plan(os.path.join(str(tmp_path), "track.mp3"))
    adapter = RekordboxAdapter(
        storage_dir=str(tmp_path), master_db=str(db), backups_dir=str(tmp_path / "backups")
    )
    result = adapter.apply(plan)

    assert result["ok"] is True
    assert result["verification"]["ok"] is True

    conn = sqlite3.connect(str(db))
    rows = conn.execute("SELECT CueNumber, CueTiming, CueKind, Comment FROM DjmdHotCue ORDER BY CueNumber").fetchall()
    conn.close()
    assert len(rows) == 2
    assert rows[0][0] == 1 and rows[0][1] == 0  # slot 1 at 0.0s
    assert rows[1][0] == 4 and rows[1][1] == 64000  # slot 4 at 64.0s
    assert rows[0][2] == 0
    assert rows[0][3] == "INTRO"


def test_rekordbox_apply_track_not_in_library(tmp_path, monkeypatch):
    db = tmp_path / "master.db"
    _seed_rekordbox_db(str(db))
    monkeypatch.setattr(RekordboxAdapter, "_agent_running", lambda self: False)

    plan = _cue_plan(os.path.join(str(tmp_path), "missing.mp3"))
    adapter = RekordboxAdapter(
        storage_dir=str(tmp_path), master_db=str(db), backups_dir=str(tmp_path / "backups")
    )
    with pytest.raises(BackupError, match="track not found in Rekordbox library"):
        adapter.apply(plan)


def test_rekordbox_apply_refuses_when_agent_running(tmp_path, monkeypatch):
    db = tmp_path / "master.db"
    _seed_rekordbox_db(str(db))
    monkeypatch.setattr(RekordboxAdapter, "_agent_running", lambda self: True)

    plan = _cue_plan(os.path.join(str(tmp_path), "track.mp3"))
    adapter = RekordboxAdapter(
        storage_dir=str(tmp_path), master_db=str(db), backups_dir=str(tmp_path / "backups")
    )
    with pytest.raises(BackupError, match="agent is running"):
        adapter.apply(plan)