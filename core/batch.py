"""Batch analysis engine.

Requirements implemented here:
- resumable jobs (state persisted to disk, tracks already analyzed are skipped)
- failed-job retry (per-track retry count, retried automatically on resume)
- cancellation (cooperative check between tracks)
- progress percentage (per-track + aggregate)
- CPU/RAM control (worker limit: `run_batch(worker_limit=N)` runs at most
  N analyses concurrently; None/1 keeps the deterministic sequential path)
- duplicate analysis prevention (by file hash / existing cueplan file)
- per-track error isolation (one bad track never aborts the batch)
- default cueplan export: when no `out_dir` is given, cueplans are
  exported to `<job_dir>/cueplans/` instead of being silently dropped
  while the job reported DONE (fix T-8).

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
from concurrent.futures import ThreadPoolExecutor
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
    # Tracks skipped in this run, reported explicitly instead of being
    # dropped silently (fix T-8): each entry is {"path", "reason",
    # "status"} with reason "already_done" or "retries_exhausted".
    skipped_tracks: list[dict] = field(default_factory=list)


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
    """Cross-thread cooperative lock, shared per job state file.

    One lock object per state path is kept in a process-wide registry, so
    two `run_batch` calls for the same job in this process actually
    serialise their state writes instead of each holding a private lock
    (the old code created a lock and never used it at all).
    Limitation: this is in-process only — it does not lock across
    processes (that would need a file-lock dependency, which is not a
    project dependency). State writes still go through this lock.
    """

    _registry: dict[str, threading.Lock] = {}
    _registry_guard = threading.Lock()

    def __init__(self, path: Path) -> None:
        self.path = path
        with _JobLock._registry_guard:
            self._lock = _JobLock._registry.setdefault(str(path), threading.Lock())

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
    """Scan a folder and create a resumable batch job. Returns the job dict.

    If `out_dir` is None, cueplans default to ``<job_dir>/cueplans/``
    (recorded in the job as `outDir`) so a successful run always exports
    its cueplans somewhere instead of reporting DONE with no output.
    """
    scan = scan_folder(folder, recursive=recursive)
    tracks = [t for t in scan.tracks if t.status == TrackStatus.QUEUED]

    root = Path(batches_dir or default_batches_dir())
    root.mkdir(parents=True, exist_ok=True)
    job_id = f"batch-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    job_dir = root / job_id
    job_dir.mkdir(parents=True)
    if out_dir is None:
        out_dir = str(job_dir / "cueplans")

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

    `worker_limit`: maximum number of analyses running concurrently.
    None or 1 runs the original deterministic sequential path. A value
    >= 2 runs analyses in a `ThreadPoolExecutor` capped at that many
    workers; state writes and progress events are still applied in the
    original track order, so resume/skip semantics are unchanged, and
    cancellation is honoured between ordered completions. Values < 1
    raise ValueError.

    Jobs created before the default-out_dir fix (no `outDir` recorded)
    fall back to ``<job_dir>/cueplans/`` so their cueplans are exported
    instead of dropped. Skipped tracks are reported in
    `BatchResult.skipped_tracks` (and in `batch_status`'s track list).
    """
    if worker_limit is not None and int(worker_limit) < 1:
        raise ValueError(f"worker_limit must be >= 1, got {worker_limit}")
    max_workers = int(worker_limit) if worker_limit is not None else 1

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
    out_dir = job.get("outDir") or str(job_dir / "cueplans")
    preference = job.get("preference", "")
    profile = job.get("profile", "open_format")
    total = len(state["tracks"])
    started = time.time()

    lock = _JobLock(state_path)

    def save_state() -> None:
        with lock:
            _save(state_path, state)

    def emit(stage: str, done: int, message: str) -> None:
        if progress:
            pct = (done / total * 100.0) if total else 100.0
            progress(stage, done, total, pct, message)

    def analyze_one(track: dict) -> tuple[bool, str, str]:
        """Analyse one track; returns (ok, cueplan_path, error)."""
        result = analyze_track(
            track["path"],
            profile_name=profile,
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
        target = os.path.join(out_dir, f"{base}.cueplan.json")
        os.makedirs(out_dir, exist_ok=True)
        export_cue_plan(result.cue_plan, target, track=track_block)
        return True, target, ""

    ok = failed = skipped = 0
    done_count = 0
    failures: list[dict] = []
    skipped_tracks: list[dict] = []

    def apply_success(track: dict, cueplan: str) -> None:
        nonlocal ok, done_count
        track["status"] = "DONE"
        track["cueplan"] = cueplan
        track["error"] = ""
        ok += 1
        done_count += 1
        emit("analyzed", done_count, f"OK {track['path']}")

    def apply_failure(track: dict, exc: Exception) -> None:
        nonlocal failed, done_count
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

    def apply_skip(track: dict, reason: str) -> None:
        nonlocal skipped, done_count
        skipped += 1
        done_count += 1
        skipped_tracks.append(
            {"path": track["path"], "reason": reason, "status": track.get("status", ""), "error": track.get("error", "")}
        )

    try:
        if max_workers <= 1:
            for track in state["tracks"]:
                if poll_cancel():
                    raise BatchCancelled("batch cancelled by user")
                status = track.get("status", "QUEUED")
                if status == "DONE":
                    apply_skip(track, "already_done")
                    continue
                if status == "FAILED" and int(track.get("retries", 0)) >= max_retries:
                    apply_skip(track, "retries_exhausted")
                    continue
                emit("analyzing", done_count, track["path"])
                try:
                    _, cueplan, _ = analyze_one(track)
                    apply_success(track, cueplan)
                except Exception as exc:  # noqa: BLE001 - isolate per-track failures
                    apply_failure(track, exc)
                save_state()
        else:
            # Concurrent path: submit all pending tracks, then apply
            # results strictly in original order so state/progress stay
            # deterministic. Cancellation is checked before submitting
            # each batch of work and between ordered completions.
            executor = ThreadPoolExecutor(max_workers=max_workers)
            try:
                ordered: list[tuple[str, dict, object | None]] = []
                for track in state["tracks"]:
                    status = track.get("status", "QUEUED")
                    if status == "DONE":
                        ordered.append(("skip_done", track, None))
                    elif status == "FAILED" and int(track.get("retries", 0)) >= max_retries:
                        ordered.append(("skip_failed", track, None))
                    else:
                        if poll_cancel():
                            raise BatchCancelled("batch cancelled by user")
                        ordered.append(("work", track, executor.submit(analyze_one, track)))
                for kind, track, fut in ordered:
                    if kind == "skip_done":
                        apply_skip(track, "already_done")
                    elif kind == "skip_failed":
                        apply_skip(track, "retries_exhausted")
                    else:
                        if poll_cancel():
                            raise BatchCancelled("batch cancelled by user")
                        emit("analyzing", done_count, track["path"])
                        try:
                            _, cueplan, _ = fut.result()  # type: ignore[union-attr]
                            apply_success(track, cueplan)
                        except Exception as exc:  # noqa: BLE001 - isolate per-track failures
                            apply_failure(track, exc)
                    save_state()
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
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
        skipped_tracks=skipped_tracks,
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