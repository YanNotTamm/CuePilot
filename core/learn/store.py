"""Learning store for personalization and cue correction tracking.

Persists DJ hot-cue corrections into a local SQLite database so CuePilot can
learn the user's preferred cue placement per genre + slot and apply it to
future analyses of tracks the user has not touched yet.

Data model
----------
`cue_feedback` stores one row per (track_path, slot) — the user's final
approved cue layout for that track (upsert). From these rows we aggregate a
`position_ratio` prior (cue time / track duration) per (genre, slot). The
prior is only trusted once enough distinct tracks have been saved.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

MIN_SAMPLES = 3
"""Minimum distinct tracks per (genre, slot) before a learned prior is used."""


def default_data_dir() -> Path:
    """Data directory for user-generated files (learning DB, backups)."""
    env = os.environ.get("CUEPILOT_DATA_DIR")
    if env:
        return Path(env)
    return Path.home() / ".cuepilot"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS cue_feedback (
    track_path       TEXT NOT NULL,
    genre            TEXT NOT NULL,
    slot             INTEGER NOT NULL,
    section_type     TEXT NOT NULL DEFAULT '',
    cue_role         TEXT NOT NULL DEFAULT '',
    time_seconds     REAL NOT NULL,
    track_duration   REAL NOT NULL,
    position_ratio   REAL NOT NULL,
    created_at       TEXT NOT NULL,
    PRIMARY KEY (track_path, slot)
);
CREATE INDEX IF NOT EXISTS idx_feedback_genre_slot ON cue_feedback (genre, slot);
"""


class LearningStore:
    """Thread-safe SQLite store for learned cue patterns."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = str(db_path or (default_data_dir() / "learning.db"))
        self._lock = threading.Lock()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(_SCHEMA)
                conn.commit()
            finally:
                conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    # ---- writing ----------------------------------------------------------

    def record_cue_feedback(
        self,
        track_path: str,
        genre: str,
        slot: int,
        time_seconds: float,
        track_duration: float,
        section_type: str = "",
        cue_role: str = "",
    ) -> None:
        """Upsert the user-approved cue position for (track_path, slot)."""
        duration = max(0.001, float(track_duration))
        ratio = max(0.0, min(1.0, float(time_seconds) / duration))
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO cue_feedback
                        (track_path, genre, slot, section_type, cue_role,
                         time_seconds, track_duration, position_ratio, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(track_path, slot) DO UPDATE SET
                        genre=excluded.genre,
                        section_type=excluded.section_type,
                        cue_role=excluded.cue_role,
                        time_seconds=excluded.time_seconds,
                        track_duration=excluded.track_duration,
                        position_ratio=excluded.position_ratio,
                        created_at=excluded.created_at
                    """,
                    (
                        track_path,
                        genre,
                        int(slot),
                        section_type,
                        cue_role,
                        float(time_seconds),
                        duration,
                        ratio,
                        now,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def record_batch(self, rows: Iterable[dict[str, Any]]) -> int:
        """Upsert several feedback rows at once. Returns rows written."""
        n = 0
        for row in rows:
            self.record_cue_feedback(
                track_path=str(row["trackPath"]),
                genre=str(row.get("genre", "")),
                slot=int(row["slot"]),
                time_seconds=float(row["time"]),
                track_duration=float(row.get("trackDuration", 0.0)),
                section_type=str(row.get("sectionType", "")),
                cue_role=str(row.get("cueRole", "")),
            )
            n += 1
        return n

    # ---- reading ----------------------------------------------------------

    def patterns_for(self, genre: str) -> list[dict[str, Any]]:
        """Aggregated learned priors for one genre, one row per slot."""
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """
                    SELECT slot,
                           AVG(position_ratio) AS avg_position,
                           COUNT(*)            AS samples,
                           GROUP_CONCAT(DISTINCT cue_role) AS roles
                    FROM cue_feedback
                    WHERE genre = ?
                    GROUP BY slot
                    ORDER BY slot
                    """,
                    (genre,),
                )
                rows = [dict(r) for r in cur.fetchall()]
            finally:
                conn.close()
        out = []
        for r in rows:
            out.append(
                {
                    "slot": int(r["slot"]),
                    "avgPosition": float(r["avg_position"]),
                    "samples": int(r["samples"]),
                    "roles": [x for x in (r["roles"] or "").split(",") if x],
                    "trusted": int(r["samples"]) >= MIN_SAMPLES,
                }
            )
        return out

    def pattern_for(self, genre: str, slot: int) -> dict[str, Any] | None:
        for p in self.patterns_for(genre):
            if p["slot"] == slot:
                return p
        return None

    def count_corrections(self, track_path: str) -> int:
        """Number of saved correction rows for one track.

        Each row is one user-approved cue slot for the track, so this counts
        how many manual corrections the DJ has recorded for it. More
        corrections → lower Gig Readiness stability component.
        """
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "SELECT COUNT(*) AS n FROM cue_feedback WHERE track_path = ?",
                    (track_path,),
                )
                return int(dict(cur.fetchone())["n"])
            finally:
                conn.close()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "SELECT COUNT(*) AS total, COUNT(DISTINCT track_path) AS tracks FROM cue_feedback"
                )
                row = dict(cur.fetchone())
                cur2 = conn.execute("SELECT COUNT(DISTINCT genre) AS genres FROM cue_feedback")
                row.update(dict(cur2.fetchone()))
            finally:
                conn.close()
        return row

    def clear(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("DELETE FROM cue_feedback")
                conn.commit()
            finally:
                conn.close()


_store: LearningStore | None = None
_store_lock = threading.Lock()


def get_store(db_path: str | Path | None = None) -> LearningStore:
    """Module-level singleton store (or a dedicated one when db_path given)."""
    global _store
    if db_path is not None:
        return LearningStore(db_path)
    with _store_lock:
        if _store is None:
            _store = LearningStore()
        return _store
