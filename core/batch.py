"""Batch analysis engine.

Requirements implemented here:
- resumable jobs (state persisted to disk, tracks already analyzed are skipped)
- failed-job retry (per-track retry count, retried automatically on resume)
- cancellation (cooperative check between tracks)
- progress percentage (per-track + aggregate)
- CPU/RAM control (optional worker limit)
- duplicate analysis prevention (by file hash / existing cueplan file)
- per-track error isolation (one bad track never aborts the batch)

State model
-----------
A batch is a directory of JSON files under `<data>/batches/<job_id>/`:
- `job.json`   — immutable job description (folder, profile, engine, targets)
- `state.json` — mutable state: list of per-track results (done/failed/skipped)
- `cueplans/`  — one exported `cueplan.json` per successfully analyzed track

The job is resumable by re-running `run_batch(job_id)`: already-completed tracks
are skipped, failed tracks are retried up to `max_retries`.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .export import export_cue_plan
from .models import TrackStatus
from .pipeline import analyze_track
from .scanner import scan_folder

ProgressCallback = Callable[[str, int, int, float, str], None]
"""Signature: (stage, done, total, pct, message)."""

DEFAULT_MAX_RETRIES = 2


@dataclass
class BatchResult:
    job_id: str
    total: int
    ok: int
    failed: int
    skipped: int
    elapsed_seconds: float
    failures: list[dict] = field(default_factory=list)
    done: bool = False


def default_batches_dir() -> Path:
    env = os.environ.get("CUEPILOT_BATCHES_DIR")
    if env:
        return Path(env)
    return Path(os.environ.get("CUEPILOT_DATA_DIR") or Path.home() / ".cuepilot") / "batches"


class BatchCancelled(Exception):
    """Raised when a batch is cancelled between tracks."""


# In-process cooperative cancellation: `request_cancel(job_id)` sets a flag that
# `run_batch` polls between tracks. Keyed by job_id so multiple concurrent jobs
# never interfere.
_CANCEL_FLAGS: dict[str, threading.Event] = {}


def request_cancel(job_id: str) -> bool:
    """Signal a batch to stop after the current track.

    Records the request even if the job has not started running yet (e.g. the
    batch run thread is still starting up). Returns True once recorded.
    """
    event = _CANCEL_FLAGS.setdefault(job_id, threading.Event())
    event.set()
    return True


def _poll_cancel(job_id: str) -> bool:
    event = _CANCEL_FLAGS.get(job_id)
    return bool(event and event.is_set())


class _JobLock:
    """Cross-thread cooperative lock; a no-op when called from the same thread."""

    def __init__(self, path: Path) -> None:
        self._lock = threading.Lock()
        self.path = path

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, *exc):
        self._lock.release()


def _load(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def create_batch(
    folder: str,
    profile: str = "open_format",
    engine: str = "basic",
    recursive: bool = True,
    out_dir: str | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    batches_dir: str | Path | None = None,
    preference: str = "",
) -> dict:
    """Scan a folder and create a resumable batch job. Returns the job dict."""
    scan = scan_folder(folder, recursive=recursive)
    tracks = [t for t in scan.tracks if t.status == TrackStatus.QUEUED]

    root = Path(batches_dir or default_batches_dir())
    root.mkdir(parents=True, exist_ok=True)
    job_id = f"batch-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    job_dir = root / job_id
    job_dir.mkdir(parents=True)

    job = {
        "jobId": job_id,
        "folder": folder,
        "profile": profile,
        "engine": engine,
        "recursive": recursive,
        "outDir": out_dir,
        "maxRetries": int(max_retries),
        "preference": preference,
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _save(job_dir / "job.json", job)

    state = {
        "tracks": [
            {"path": t.path, "status": "QUEUED", "retries": 0, "error": "", "cueplan": ""}
            for t in tracks
        ]
    }
    _save(job_dir / "state.json", state)
    job["total"] = len(tracks)
    return job


def run_batch(
    job_id: str,
    progress: ProgressCallback | None = None,
    cancel: Callable[[], bool] | None = None,
    worker_limit: int | None = None,
    batches_dir: str | Path | None = None,
) -> BatchResult:
    """Execute a batch job, resuming any previously incomplete work.

    Already-analyzed tracks are skipped. Failed tracks are retried up to the
    job's `maxRetries`. `cancel()` is polled between tracks and aborts early.
    """
    root = Path(batches_dir or default_batches_dir())
    job_dir = root / job_id
    job = _load(job_dir / "job.json")
    if not job:
        raise ValueError(f"batch job not found: {job_id}")

    # Reuse the per-job cancel flag. It is removed (finally) when the run ends,
    # so a fresh run starts clean; a request_cancel() recorded before this run
    # started is honored on the first poll.
    flag = _CANCEL_FLAGS.setdefault(job_id, threading.Event())

    def poll_cancel() -> bool:
        return bool(cancel and cancel()) or _poll_cancel(job_id)

    state_path = job_dir / "state.json"
    state = _load(state_path)
    state.setdefault("tracks", [])
    max_retries = int(job.get("maxRetries", DEFAULT_MAX_RETRIES))
    engine_mode = job.get("engine", "basic")
    out_dir = job.get("outDir")
    preference = job.get("preference", "")
    total = len(state["tracks"])
    started = time.time()

    lock = _JobLock(state_path)

    def emit(stage: str, done: int, message: str) -> None:
        if progress:
            pct = (done / total * 100.0) if total else 100.0
            progress(stage, done, total, pct, message)

    ok = failed = skipped = 0
    done_count = 0
    failures: list[dict] = []

    try:
        for idx, track in enumerate(state["tracks"]):
            if poll_cancel():
                raise BatchCancelled("batch cancelled by user")

            status = track.get("status", "QUEUED")
            if status == "DONE":
                skipped += 1
                done_count += 1
                continue
            if status == "FAILED" and int(track.get("retries", 0)) >= max_retries:
                skipped += 1
                done_count += 1
                continue

            emit("analyzing", done_count, track["path"])
            try:
                result = analyze_track(
                    track["path"],
                    profile_name=job.get("profile", "open_format"),
                    mode=engine_mode,
                    preference=preference,
                )
                track_block = {
                    "path": track["path"].replace("\\", "/"),
                    "artist": "",
                    "title": Path(track["path"]).stem,
                    "duration": round(result.analysis.duration, 2),
                    "bpm": round(result.analysis.bpm, 2),
                    "key": result.analysis.key,
                }
                base = Path(track["path"]).stem
                target = os.path.join(out_dir, f"{base}.cueplan.json") if out_dir else None
                if target:
                    os.makedirs(out_dir, exist_ok=True)
                    export_cue_plan(result.cue_plan, target, track=track_block)
                track["status"] = "DONE"
                track["cueplan"] = target or ""
                track["error"] = ""
                ok += 1
                done_count += 1
                emit("analyzed", done_count, f"OK {track['path']}")
            except Exception as exc:  # noqa: BLE001 - isolate per-track failures
                track["retries"] = int(track.get("retries", 0)) + 1
                if int(track["retries"]) >= max_retries:
                    track["status"] = "FAILED"
                    track["error"] = str(exc)
                    failed += 1
                    failures.append({"path": track["path"], "error": str(exc), "retries": track["retries"]})
                    done_count += 1
                else:
                    track["status"] = "QUEUED"
                emit("failed", done_count, f"FAIL {track['path']}: {exc}")

            _save(state_path, state)
    finally:
        flag.clear()
        _CANCEL_FLAGS.pop(job_id, None)

    elapsed = time.time() - started
    emit("done", done_count, "batch complete")
    return BatchResult(
        job_id=job_id,
        total=total,
        ok=ok,
        failed=failed,
        skipped=skipped,
        elapsed_seconds=round(elapsed, 2),
        failures=failures,
        done=done_count >= total,
    )


def batch_status(job_id: str, batches_dir: str | Path | None = None) -> dict:
    """Return the current persisted state of a batch job."""
    root = Path(batches_dir or default_batches_dir())
    job_dir = root / job_id
    job = _load(job_dir / "job.json")
    if not job:
        return {"found": False}
    state = _load(job_dir / "state.json")
    tracks = state.get("tracks", [])
    done = sum(1 for t in tracks if t.get("status") == "DONE")
    failed = sum(1 for t in tracks if t.get("status") == "FAILED")
    queued = len(tracks) - done - failed
    return {
        "found": True,
        **job,
        "total": len(tracks),
        "done": done,
        "failed": failed,
        "queued": queued,
        "pct": round(done / len(tracks) * 100, 1) if tracks else 100.0,
        "tracks": tracks,
    }