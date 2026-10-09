"""Analysis cache keyed by audio file hash (SIE.md §46).

Stem separation is the most expensive pipeline stage. Per SIE.md §46-47 the
cache stores per-track artifacts (stem energy envelopes, beat grid, BPM,
feature vectors) keyed by the SHA-256 of the audio file so re-analysis of the
same file skips the expensive steps unless the analysis version changed.

Artifacts are stored under ``~/.cuepilot/cache/{artifact}/{hash}.npz`` (or
``CUEPILOT_DATA_DIR`` when set). Each file embeds its analysis version; a
mismatch invalidates the entry.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np

from .settings import _data_dir

ANALYSIS_VERSION = "1.0.0"


def file_sha256(path: str) -> str:
    """Streaming SHA-256 of a file (SIE.md §46: content hash, not path hash)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _artifact_dir(artifact: str) -> Path:
    d = _data_dir() / "cache" / artifact
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_path(artifact: str, file_hash: str) -> Path:
    return _artifact_dir(artifact) / f"{file_hash}.npz"


def _store(
    artifact: str,
    file_hash: str,
    arrays: dict[str, np.ndarray],
    meta: dict[str, Any] | None = None,
) -> Path:
    path = _cache_path(artifact, file_hash)
    meta = dict(meta or {})
    meta["analysis_version"] = ANALYSIS_VERSION
    np.savez_compressed(path, **arrays, **{f"_meta_{k}": v for k, v in meta.items()})
    return path


def _load(
    artifact: str,
    file_hash: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]] | None:
    path = _cache_path(artifact, file_hash)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            meta = {k[6:]: v for k, v in data.items() if k.startswith("_meta_")}
            if meta.get("analysis_version") != ANALYSIS_VERSION:
                return None
            arrays = {
                k: np.asarray(data[k]) for k in data.files if not k.startswith("_meta_")
            }
            return arrays, meta
    except Exception:
        return None


def get_cached_arrays(artifact: str, file_hash: str):
    """Return cached arrays+meta, or None if absent/invalid/version-mismatched."""
    return _load(artifact, file_hash)


def store_cached_arrays(
    artifact: str,
    file_hash: str,
    arrays: dict[str, np.ndarray],
    meta: dict[str, Any] | None = None,
) -> Path:
    """Persist arrays under `artifact` for `file_hash`."""
    return _store(artifact, file_hash, arrays, meta)


def clear_cache(artifact: str | None = None) -> int:
    """Remove cached artifacts. Returns the number of files removed."""
    if artifact is not None:
        root = _artifact_dir(artifact)
    else:
        root = _data_dir() / "cache"
    removed = 0
    if root.exists():
        for p in root.rglob("*.npz"):
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass
    return removed