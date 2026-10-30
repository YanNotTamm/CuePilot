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
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .settings import _data_dir

ANALYSIS_VERSION = "1.0.0"


def _normalise_params(params: dict[str, Any] | None) -> dict[str, Any] | None:
    """Normalise analysis params to a JSON-comparable dict, or None."""
    if params is None:
        return None
    return json.loads(json.dumps(params, sort_keys=True, default=str))


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
    params: dict[str, Any] | None = None,
) -> Path:
    path = _cache_path(artifact, file_hash)
    meta = dict(meta or {})
    meta["analysis_version"] = ANALYSIS_VERSION
    # Analysis parameters (sample rate, hop length, ...) are part of the
    # cache identity: arrays computed at one sample rate / hop grid are
    # meaningless on another grid, so they must never be reused across
    # parameter changes (fix R-2). Stored as a sorted JSON string so it
    # survives npz round-tripping; callers that pass `params` on load get
    # None (invalid) when the stored params differ or are absent.
    normalised = _normalise_params(params)
    if normalised is not None:
        meta["analysis_params"] = json.dumps(normalised, sort_keys=True)
    np.savez_compressed(path, **arrays, **{f"_meta_{k}": v for k, v in meta.items()})
    return path


def _load(
    artifact: str,
    file_hash: str,
    params: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]] | None:
    path = _cache_path(artifact, file_hash)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            meta = {k[6:]: v for k, v in data.items() if k.startswith("_meta_")}
            if meta.get("analysis_version") != ANALYSIS_VERSION:
                return None
            # Safe migration: an entry written before params were recorded
            # (no `analysis_params` in meta) or with different params is
            # invalid for a caller that specifies params — return None so
            # the caller recomputes instead of silently reusing stale
            # arrays. Callers that pass no params keep the legacy
            # version-only behaviour.
            expected = _normalise_params(params)
            if expected is not None:
                stored_raw = meta.get("analysis_params")
                if stored_raw is None:
                    return None
                try:
                    stored = json.loads(str(stored_raw))
                except Exception:
                    return None
                if stored != expected:
                    return None
                meta["analysis_params"] = stored
            arrays = {
                k: np.asarray(data[k]) for k in data.files if not k.startswith("_meta_")
            }
            return arrays, meta
    except Exception:
        return None


def get_cached_arrays(artifact: str, file_hash: str, params: dict[str, Any] | None = None):
    """Return cached arrays+meta, or None if absent/invalid/version-mismatched.

    Pass ``params`` (e.g. ``{"sample_rate": 22050, "hop_length": 512}``)
    to require an exact parameter match; entries stored without params
    or with different params are treated as misses.
    """
    return _load(artifact, file_hash, params=params)


def store_cached_arrays(
    artifact: str,
    file_hash: str,
    arrays: dict[str, np.ndarray],
    meta: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> Path:
    """Persist arrays under `artifact` for `file_hash`.

    Pass ``params`` to record the analysis parameters in the entry meta;
    they become part of the cache identity on later loads.
    """
    return _store(artifact, file_hash, arrays, meta, params=params)


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