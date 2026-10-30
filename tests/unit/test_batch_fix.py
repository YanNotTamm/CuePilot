"""Fix-verification tests for batch/CLI (T-8), cache params (R-2).

These tests do NOT need librosa: `analyze_track` and `export_cue_plan`
are stubbed in `core.batch`, and jobs are built from dummy audio files
(`scan_folder` only hashes + reads metadata, which degrades gracefully
without mutagen). The full pipeline tests in test_batch.py still cover
real analysis when librosa is installed.
"""

import json
import os
import threading
import time
from types import SimpleNamespace

import pytest

import core.batch as batch_mod
from core.batch import batch_status, create_batch, run_batch

pytest.importorskip("numpy", reason="numpy required for cache tests")


def _music_dir(tmp_path, names=("a.wav", "b.wav", "c.wav", "d.wav")):
    d = tmp_path / "music"
    d.mkdir()
    for name in names:
        (d / name).write_bytes(b"fake-audio-" + name.encode())
    return d


@pytest.fixture()
def stub_analysis(monkeypatch):
    """Stub analyze/export so run_batch never touches the real pipeline."""
    calls = {"n": 0}

    def fake_analyze(path, profile_name="open_format", mode="basic", preference=""):
        calls["n"] += 1
        analysis = SimpleNamespace(duration=200.0, bpm=128.0, key="Am")
        return SimpleNamespace(analysis=analysis, cue_plan=SimpleNamespace(cues=[]))

    def fake_export(cue_plan, target, track=None):
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump({"track": track}, f)

    monkeypatch.setattr(batch_mod, "analyze_track", fake_analyze)
    monkeypatch.setattr(batch_mod, "export_cue_plan", fake_export)
    return calls


def test_default_out_dir_exports_cueplans(tmp_path, stub_analysis):
    music = _music_dir(tmp_path)
    job = create_batch(str(music), batches_dir=tmp_path / "batches")
    # No out_dir given: the job must record a default inside the job dir.
    assert job["outDir"].endswith("cueplans")
    result = run_batch(job["jobId"], batches_dir=tmp_path / "batches")
    assert result.ok == 4 and result.done is True
    cueplans = tmp_path / "batches" / job["jobId"] / "cueplans"
    assert (cueplans / "a.cueplan.json").is_file()
    assert (cueplans / "d.cueplan.json").is_file()


def test_resume_reports_skipped_tracks(tmp_path, stub_analysis):
    music = _music_dir(tmp_path, names=("a.wav", "b.wav"))
    job = create_batch(str(music), out_dir=str(tmp_path / "out"), batches_dir=tmp_path / "batches")
    run_batch(job["jobId"], batches_dir=tmp_path / "batches")
    second = run_batch(job["jobId"], batches_dir=tmp_path / "batches")
    assert second.skipped == 2 and second.ok == 0
    assert len(second.skipped_tracks) == 2
    assert all(s["reason"] == "already_done" for s in second.skipped_tracks)
    assert {s["path"] for s in second.skipped_tracks} == {
        str(music / "a.wav"), str(music / "b.wav"),
    }


def test_retries_exhausted_reported_as_skipped(tmp_path, monkeypatch):
    music = _music_dir(tmp_path, names=("a.wav",))
    job = create_batch(
        str(music), out_dir=str(tmp_path / "out"), max_retries=1,
        batches_dir=tmp_path / "batches",
    )

    def always_fail(path, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(batch_mod, "analyze_track", always_fail)
    first = run_batch(job["jobId"], batches_dir=tmp_path / "batches")
    assert first.failed == 1
    assert first.failures[0]["path"] == str(music / "a.wav")
    second = run_batch(job["jobId"], batches_dir=tmp_path / "batches")
    assert second.skipped == 1
    assert second.skipped_tracks[0]["reason"] == "retries_exhausted"


def test_worker_limit_validated_and_enforced(tmp_path, monkeypatch):
    music = _music_dir(tmp_path)
    job = create_batch(str(music), out_dir=str(tmp_path / "out"), batches_dir=tmp_path / "batches")
    with pytest.raises(ValueError):
        run_batch(job["jobId"], worker_limit=0, batches_dir=tmp_path / "batches")

    state = {"current": 0, "peak": 0}
    guard = threading.Lock()

    def slow_analyze(path, **kwargs):
        with guard:
            state["current"] += 1
            state["peak"] = max(state["peak"], state["current"])
        time.sleep(0.05)
        with guard:
            state["current"] -= 1
        analysis = SimpleNamespace(duration=200.0, bpm=128.0, key="Am")
        return SimpleNamespace(analysis=analysis, cue_plan=SimpleNamespace(cues=[]))

    def fake_export(cue_plan, target, track=None):
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump({}, f)

    monkeypatch.setattr(batch_mod, "analyze_track", slow_analyze)
    monkeypatch.setattr(batch_mod, "export_cue_plan", fake_export)
    result = run_batch(job["jobId"], worker_limit=2, batches_dir=tmp_path / "batches")
    assert result.ok == 4
    assert state["peak"] <= 2
    assert state["peak"] == 2  # limit is real: parallelism actually happened


def test_batch_cli_create_run_status(tmp_path, stub_analysis, monkeypatch, capsys):
    pytest.importorskip("librosa", reason="CLI imports the analyzer engine")
    monkeypatch.setenv("CUEPILOT_BATCHES_DIR", str(tmp_path / "batches"))
    from cli.main import main

    music = _music_dir(tmp_path, names=("a.wav",))
    assert main(["batch", "create", str(music)]) == 0
    out = capsys.readouterr().out
    job_id = [line for line in out.splitlines() if line.startswith("batch: ")][0].split()[1]
    assert main(["batch", "status", job_id]) == 0
    assert main(["batch", "run", job_id]) == 0
    out = capsys.readouterr().out
    assert "ok=1" in out
    # Legacy form still works.
    assert main(["batch", str(music), "--create-only"]) == 0


def test_cache_params_invalidate(tmp_path, monkeypatch):
    import numpy as np

    import core.cache as cache

    monkeypatch.setenv("CUEPILOT_DATA_DIR", str(tmp_path))
    arrays = {"x": np.array([1.0, 2.0])}
    params = {"sample_rate": 22050, "hop_length": 512}
    cache.store_cached_arrays("stem_features", "hash1", arrays, params=params)
    loaded = cache.get_cached_arrays("stem_features", "hash1", params=params)
    assert loaded is not None and loaded[0]["x"].tolist() == [1.0, 2.0]
    # Different analysis parameters must NOT reuse the entry.
    assert cache.get_cached_arrays("stem_features", "hash1", params={"sample_rate": 44100, "hop_length": 512}) is None
    # Legacy entry (stored without params) is a miss for a params caller,
    # but still loads for a caller that specifies no params.
    cache.store_cached_arrays("stem_features", "hash2", arrays)
    assert cache.get_cached_arrays("stem_features", "hash2", params=params) is None
    assert cache.get_cached_arrays("stem_features", "hash2") is not None
