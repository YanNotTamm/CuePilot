"""Tests for the batch analysis engine."""

import json
import os

import pytest

from core.batch import (
    BatchCancelled,
    batch_status,
    create_batch,
    request_cancel,
    run_batch,
)

pytest.importorskip("librosa", reason="librosa required for batch analysis")


@pytest.fixture()
def music_dir(tmp_path):
    from tests.fixtures.synth import write_synth_track

    structure = [("INTRO", 0.3), ("DROP", 1.0), ("OUTRO", 0.2)]
    write_synth_track(str(tmp_path / "a.wav"), bpm=128, seconds=8, structure=structure)
    write_synth_track(str(tmp_path / "b.wav"), bpm=126, seconds=8, structure=structure)
    return tmp_path


def test_create_batch(music_dir, tmp_path):
    job = create_batch(str(music_dir), out_dir=str(tmp_path / "out"), batches_dir=tmp_path / "batches")
    assert job["total"] == 2
    assert job["jobId"].startswith("batch-")


def test_run_batch_success(music_dir, tmp_path):
    job = create_batch(str(music_dir), out_dir=str(tmp_path / "out"), batches_dir=tmp_path / "batches")
    result = run_batch(job["jobId"], batches_dir=tmp_path / "batches")
    assert result.ok == 2
    assert result.failed == 0
    assert result.skipped == 0
    assert result.done is True
    out = tmp_path / "out"
    assert (out / "a.cueplan.json").is_file()
    assert (out / "b.cueplan.json").is_file()


def test_run_batch_resume_skips_done(music_dir, tmp_path):
    job = create_batch(str(music_dir), out_dir=str(tmp_path / "out"), batches_dir=tmp_path / "batches")
    first = run_batch(job["jobId"], batches_dir=tmp_path / "batches")
    assert first.ok == 2
    second = run_batch(job["jobId"], batches_dir=tmp_path / "batches")
    assert second.skipped == 2
    assert second.ok == 0


def test_batch_status_after_run(music_dir, tmp_path):
    job = create_batch(str(music_dir), out_dir=str(tmp_path / "out"), batches_dir=tmp_path / "batches")
    run_batch(job["jobId"], batches_dir=tmp_path / "batches")
    status = batch_status(job["jobId"], batches_dir=tmp_path / "batches")
    assert status["found"] is True
    assert status["done"] == 2
    assert status["pct"] == 100.0
    assert len(status["tracks"]) == 2


def test_cancel_stops_batch(music_dir, tmp_path):
    job = create_batch(str(music_dir), out_dir=str(tmp_path / "out"), batches_dir=tmp_path / "batches")
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 1  # cancel after the first track

    with pytest.raises(BatchCancelled):
        run_batch(job["jobId"], cancel=cancel, batches_dir=tmp_path / "batches")


def test_request_cancel_by_id_stops_batch(music_dir, tmp_path):
    job = create_batch(str(music_dir), out_dir=str(tmp_path / "out"), batches_dir=tmp_path / "batches")
    assert request_cancel(job["jobId"]) is True
    with pytest.raises(BatchCancelled):
        run_batch(job["jobId"], batches_dir=tmp_path / "batches")


def test_request_cancel_unknown_job_returns_true():
    assert request_cancel("batch-does-not-exist") is True


def test_request_cancel_resets_between_runs(music_dir, tmp_path):
    """A cancelled flag must not leak into a later (resumed) run."""
    job = create_batch(str(music_dir), out_dir=str(tmp_path / "out"), batches_dir=tmp_path / "batches")
    assert request_cancel(job["jobId"]) is True
    with pytest.raises(BatchCancelled):
        run_batch(job["jobId"], batches_dir=tmp_path / "batches")
    result = run_batch(job["jobId"], batches_dir=tmp_path / "batches")
    assert result.ok == 2


def test_missing_job_raises(tmp_path):
    with pytest.raises(ValueError):
        run_batch("batch-does-not-exist", batches_dir=tmp_path / "batches")
    assert batch_status("batch-does-not-exist", batches_dir=tmp_path / "batches") == {"found": False}


def test_progress_callback(music_dir, tmp_path):
    job = create_batch(str(music_dir), out_dir=str(tmp_path / "out"), batches_dir=tmp_path / "batches")
    events = []

    def progress(stage, done, total, pct, message):
        events.append((stage, done, total))

    run_batch(job["jobId"], progress=progress, batches_dir=tmp_path / "batches")
    assert ("analyzing", 0, 2) in events
    assert ("analyzed", 1, 2) in events
    assert ("done", 2, 2) in events