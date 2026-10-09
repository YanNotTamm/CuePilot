"""Folder scanner for recursive audio discovery and deduplication.

Recursively discovers supported audio files, reads metadata (mutagen), hashes
files for duplicate/corruption detection, and reports status per track.
"""

from __future__ import annotations

import hashlib
import os
import time

from .models import ScanResult, Track, TrackStatus

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".aiff", ".aif", ".flac", ".ogg"}

_HASH_CHUNK = 1024 * 1024


def file_hash(path: str) -> str:
    """SHA-256 of file contents (streaming, for large audio files)."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while chunk := f.read(_HASH_CHUNK):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


def _read_metadata(path: str) -> tuple[str, str, str, float]:
    """Return (artist, title, album, duration) via mutagen where possible."""
    try:
        from mutagen import File as MutagenFile

        meta = MutagenFile(path, easy=True)
        if meta is None:
            return "", "", "", 0.0

        def first(key: str, default: str = "") -> str:
            try:
                val = meta.get(key)
                if isinstance(val, list) and val:
                    return str(val[0]).strip()
                if val:
                    return str(val).strip()
            except Exception:
                pass
            return default

        artist = first("artist")
        title = first("title")
        album = first("album")

        duration = 0.0
        try:
            info = meta.info
            if info is not None and getattr(info, "length", 0):
                duration = float(info.length)
        except Exception:
            duration = 0.0

        return artist, title, album, duration
    except Exception:
        return "", "", "", 0.0


def is_supported(path: str) -> bool:
    return path.lower().rsplit(".", 1)[-1] and f".{path.lower().rsplit('.', 1)[-1]}" in SUPPORTED_EXTENSIONS


def scan_folder(
    folder: str,
    recursive: bool = True,
    progress_callback=None,
) -> ScanResult:
    """Scan `folder` for supported audio files.

    Assigns each discovered file a QUEUED status, computing hash, metadata,
    and duration. Files that fail to hash are marked FAILED; supported files
    that mutagen cannot open are still kept with empty metadata (audio may
    still be analyzed by the engine).

    Returns a ScanResult with all tracks and duplicate/error counts.
    """
    start = time.perf_counter()
    tracks: list[Track] = []
    seen_hashes: dict[str, str] = {}

    walker = os.walk(folder) if recursive else [(folder, [], os.listdir(folder))]

    for root, _dirs, files in walker:
        for filename in sorted(files):
            path = os.path.join(root, filename)
            if not is_supported(path):
                continue
            if progress_callback:
                progress_callback(path)

            track_id = hashlib.sha1(path.lower().encode("utf-8")).hexdigest()[:16]
            track = Track(
                id=track_id,
                path=path,
                filename=filename,
                status=TrackStatus.QUEUED,
            )
            try:
                track.file_hash = file_hash(path)
            except Exception:
                track.file_hash = ""
            if not track.file_hash:
                track.status = TrackStatus.FAILED
                track.error = "cannot read file"
            else:
                if track.file_hash in seen_hashes:
                    track.status = TrackStatus.SKIPPED
                    track.error = f"duplicate of {seen_hashes[track.file_hash]}"
                else:
                    seen_hashes[track.file_hash] = path

            if track.status in (TrackStatus.QUEUED,):
                artist, title, album, duration = _read_metadata(path)
                track.artist = artist
                track.title = title
                track.album = album
                track.duration = duration

            tracks.append(track)

    total = len(tracks)
    duplicates = sum(1 for t in tracks if t.status == TrackStatus.SKIPPED)
    failed = sum(1 for t in tracks if t.status == TrackStatus.FAILED)
    elapsed = time.perf_counter() - start

    return ScanResult(
        folder=folder,
        tracks=tracks,
        total=total,
        duplicates=duplicates,
        failed=failed,
        elapsed_seconds=elapsed,
    )
