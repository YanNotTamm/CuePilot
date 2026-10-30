"""Fix-verification tests for the PIPELINE & BENCHMARK module (T-3, T-7, T-1).

Run with the real dependencies when available::

    python -m pytest tests/unit/test_pipeline_fix.py -v

When librosa / soundfile are missing (this sandbox), the import guard below
installs minimal stubs so the pure-logic tests in this file still run: none
of them need a real decoder — they exercise genre resolution, stem-backend
reporting, cue clamping, key detection, and the benchmark cue matcher.
Tests that DO need a real decoder live in test_decode_fix.py and are
skipped/expected to fail honestly there when librosa is absent.
"""

import sys
import types
import warnings

import numpy as np
import pytest

# --- optional-dependency guard (stub only when the real module is absent) ---
try:
    import librosa  # noqa: F401
except ImportError:  # pragma: no cover - sandbox without librosa
    librosa = types.ModuleType("librosa")
    sys.modules.setdefault("librosa", librosa)
try:
    import soundfile  # noqa: F401
except ImportError:  # pragma: no cover - sandbox without soundfile
    sys.modules.setdefault("soundfile", types.ModuleType("soundfile"))

from core.analyzer.base import AudioData
from core.analyzer.key import MAJOR_PROFILE, MINOR_PROFILE, detect_key
from core.models import Bar, Beat, Cue, CuePlan, CueType, Phrase, Section, SectionType
from core.pipeline import (
    _analyze_stems,
    _clamp_cues_to_duration,
    _clip_sections_to_duration,
    _resolve_genre_profile,
    analyze_track,
)
from core.stems.features import StemFeatures

# ---------------------------------------------------------------------------
# Key detection (core/analyzer/key.py) — previously had NO direct test.
# ---------------------------------------------------------------------------


def _chroma_of(mean_profile):
    """(12, 8) chromagram whose per-pitch-class mean is `mean_profile`."""
    return np.tile(np.asarray(mean_profile, dtype=float)[:, None], (1, 8))


def test_key_c_major_profile_is_8b():
    assert detect_key(_chroma_of(MAJOR_PROFILE)) == "8B"


def test_key_a_minor_profile_is_8a():
    assert detect_key(_chroma_of(MINOR_PROFILE)) == "8A"


def test_key_g_major_profile_is_9b():
    assert detect_key(_chroma_of(np.roll(MAJOR_PROFILE, 7))) == "9B"


def test_key_d_minor_profile_is_10a():
    assert detect_key(_chroma_of(np.roll(MINOR_PROFILE, 2))) == "10A"


def test_key_empty_and_degenerate_fall_back_to_8a():
    assert detect_key(np.zeros((12, 0))) == "8A"
    assert detect_key(np.zeros((12, 8))) == "8A"
    assert detect_key(np.zeros((11, 8))) == "8A"


# ---------------------------------------------------------------------------
# T-3: genre honesty — profile is a prior, never a detection.
# ---------------------------------------------------------------------------


def test_resolve_known_profile_no_fallback():
    profile, fell_back = _resolve_genre_profile("edm")
    assert profile.name == "edm"
    assert fell_back is False


def test_resolve_alias_and_normalization_no_fallback():
    profile, fell_back = _resolve_genre_profile("hiphop")
    assert profile.name == "hip_hop"
    assert fell_back is False
    profile, fell_back = _resolve_genre_profile("Tech-House")
    assert profile.name == "tech_house"
    assert fell_back is False


def test_resolve_unknown_profile_falls_back_loudly():
    with pytest.warns(UserWarning, match="Unknown genre profile"):
        profile, fell_back = _resolve_genre_profile("not_a_genre_xyz")
    assert profile.name == "open_format"
    assert fell_back is True


def test_resolve_unknown_profile_warning_names_open_format_fallback():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _resolve_genre_profile("definitely_not_a_genre")
    assert any("open_format" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# Cue / section duration clamping (short-track safety net).
# ---------------------------------------------------------------------------


def _cue(slot, time):
    return Cue(
        slot=slot,
        type=CueType.DROP,
        time=time,
        label="DROP",
        confidence=0.9,
        color="#E74C3C",
    )


def test_clamp_drops_cues_outside_duration_but_keeps_boundary():
    cues = [_cue(1, -0.5), _cue(2, 4.9), _cue(3, 5.0), _cue(4, 5.1)]
    kept = _clamp_cues_to_duration(cues, 5.0)
    assert [c.slot for c in kept] == [2, 3]


def test_clip_sections_to_duration():
    sections = [
        Section(type=SectionType.INTRO, start=0.0, end=4.0, confidence=0.9),
        Section(type=SectionType.DROP, start=4.0, end=9.0, confidence=0.9),
        Section(type=SectionType.OUTRO, start=9.0, end=12.0, confidence=0.9),
    ]
    clipped = _clip_sections_to_duration(sections, 7.0)
    assert len(clipped) == 2
    assert clipped[1].end == pytest.approx(7.0)
    assert clipped[1].start == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Shared fake harness for analyze_track wiring tests (no DSP, no decoder).
# ---------------------------------------------------------------------------


class _FakeEngine:
    name = "fake"
    version = "0.0-test"

    def __init__(self, duration=60.0):
        self._duration = duration

    def load(self, path):
        return AudioData(
            samples=np.zeros((1, int(self._duration * 22050)), dtype=np.float32),
            sample_rate=22050,
            channels=1,
            duration=self._duration,
        )

    def extract(self, audio, progress=None):
        from core.analyzer.base import AnalysisFeatures

        n = 64
        return AnalysisFeatures(
            hop_length=512,
            sample_rate=22050,
            frame_times=np.linspace(0.0, self._duration, n),
            onset_env=np.zeros(n),
            rms=np.full(n, 0.5),
            spectral_centroid=np.zeros(n),
            spectral_flux=np.zeros(n),
            chroma=np.zeros((12, n)),
            harmonic=np.zeros(n),
            percussive=np.zeros(n),
            mfcc=np.zeros((13, n)),
            subbass_energy=np.zeros(n),
            bpm_estimate=128.0,
        )


class _FakeGrid:
    bpm = 128.0
    confidence = 0.92
    beats = [Beat(time=float(t), is_downbeat=(i % 4 == 0), index=i) for i, t in enumerate(range(0, 60, 1))]
    bars = [Bar(index=0, downbeat_time=0.0, beat_count=4)]
    phrases = [Phrase(index=0, start_time=0.0, end_time=60.0, bar_count=4)]
    downbeat_times = [0.0, 4.0, 8.0]
    subdivision = "four"
    subdivision_confidence = 0.9
    tempo_verification = {}


class _FakeQc:
    def to_dict(self):
        return {}


class _FakeReadiness:
    def to_dict(self):
        return {}


class _FakeStore:
    def count_corrections(self, path):
        return 0


def _patch_pipeline_wiring(monkeypatch, cues, sections, grid=None):
    """Replace the heavy DSP stages with canned outputs."""
    import core.learn
    import core.pipeline as pipeline

    monkeypatch.setattr(pipeline, "detect_beatgrid", lambda feats, **kw: grid or _FakeGrid())
    monkeypatch.setattr(pipeline, "detect_structure", lambda feats, g, prof: sections)
    monkeypatch.setattr(pipeline, "generate_cueplan", lambda *a, **kw: CuePlan(cues=list(cues), sections=list(sections)))
    monkeypatch.setattr(pipeline, "compute_qc_report", lambda audio, feats: _FakeQc())
    monkeypatch.setattr(pipeline, "detect_loops", lambda *a, **kw: [])
    monkeypatch.setattr(pipeline, "compute_readiness", lambda **kw: _FakeReadiness())
    monkeypatch.setattr(pipeline, "adaptive_weights", lambda name, base_weights=None: dict(base_weights or {}))
    monkeypatch.setattr(pipeline, "apply_learned_patterns", lambda cues_, duration, genre, **kw: cues_)
    monkeypatch.setattr(pipeline, "load_edits", lambda path: None)
    monkeypatch.setattr(core.learn, "get_store", lambda: _FakeStore())


def _run_pipeline(monkeypatch, tmp_path, profile_name="edm", duration=60.0, cues=None, mode="basic"):
    sections = [Section(type=SectionType.INTRO, start=0.0, end=10.0, confidence=0.9)]
    if cues is None:
        cues = [_cue(5, 20.0)]
    _patch_pipeline_wiring(monkeypatch, cues, sections)
    return analyze_track(
        str(tmp_path / "track.wav"),
        profile_name=profile_name,
        engine=_FakeEngine(duration=duration),
        apply_edits=False,
        mode=mode,
    )


def test_pipeline_genre_fields_are_honest(monkeypatch, tmp_path):
    result = _run_pipeline(monkeypatch, tmp_path, profile_name="edm")
    # The old code wrote genre_detected="edm" and genre_confidence=0.92
    # (the beatgrid confidence). Neither claim is true: there is no genre
    # detector, so the deprecated fields must stay empty / 0.
    assert result.cue_plan.genre_detected == ""
    assert result.cue_plan.genre_confidence == 0.0
    assert result.cue_plan.genre_profile_used == "edm"
    genre = result.analysis.extras["genre"]
    assert genre["profileUsed"] == "edm"
    assert genre["profileRequested"] == "edm"
    assert genre["profileFallback"] is False
    assert genre["detected"] is None
    assert genre["confidence"] is None
    assert genre["detector"] is None


def test_pipeline_unknown_profile_warns_and_records_fallback(monkeypatch, tmp_path):
    with pytest.warns(UserWarning, match="Unknown genre profile"):
        result = _run_pipeline(monkeypatch, tmp_path, profile_name="nope_genre")
    assert result.profile.name == "open_format"
    assert result.analysis.extras["genre"]["profileFallback"] is True
    assert result.analysis.extras["genre"]["profileUsed"] == "open_format"
    assert result.cue_plan.genre_detected == ""


def test_pipeline_short_track_has_no_cue_outside_duration(monkeypatch, tmp_path):
    cues = [_cue(1, 0.0), _cue(5, 6.5), _cue(7, 12.0)]
    result = _run_pipeline(monkeypatch, tmp_path, duration=7.0, cues=cues)
    assert all(0.0 <= c.time <= 7.0 for c in result.cue_plan.cues)
    assert all(s.end <= 7.0 for s in result.sections)


# ---------------------------------------------------------------------------
# T-7: stem backend honesty.
# ---------------------------------------------------------------------------


class _FailingOnnxSeparator:
    name = "onnx"

    def available(self):
        return True

    def separate(self, audio, progress=None):
        raise RuntimeError("simulated ONNX model failure")


def test_analyze_stems_onnx_failure_uses_dsp_and_says_so(monkeypatch):
    import core.pipeline as pipeline

    monkeypatch.setattr(pipeline, "get_separator", lambda preferred: _FailingOnnxSeparator())
    monkeypatch.setattr(pipeline, "compute_fused_energy", lambda feats, **kw: np.zeros(8))
    monkeypatch.setattr(
        pipeline,
        "detect_stem_sections",
        lambda feats, grid, profile, fused_energy=None: types.SimpleNamespace(
            sections=[Section(type=SectionType.INTRO, start=0.0, end=2.0, confidence=0.9)]
        ),
    )
    from core.analyzer.base import AnalysisFeatures

    audio = AudioData(
        samples=np.random.default_rng(0).normal(0, 0.1, (1, 44100)).astype(np.float32),
        sample_rate=22050,
        channels=1,
        duration=2.0,
    )
    features = AnalysisFeatures(
        hop_length=512,
        sample_rate=22050,
        frame_times=np.linspace(0.0, 2.0, 8),
        onset_env=np.zeros(8),
        rms=np.full(8, 0.5),
        spectral_centroid=np.zeros(8),
        spectral_flux=np.zeros(8),
        chroma=np.zeros((12, 8)),
        harmonic=np.zeros(8),
        percussive=np.zeros(8),
        mfcc=np.zeros((13, 8)),
        subbass_energy=np.zeros(8),
    )
    sections, stem_feats = _analyze_stems(
        path="/nonexistent/track.wav",
        audio=audio,
        grid=_FakeGrid(),
        features=features,
        profile=None,
        progress=None,
        base_pct=55,
        preferred_backend="onnx",
        enabled_stems=["vocal", "drums", "bass", "other"],
    )
    assert stem_feats is not None
    assert stem_feats.backend == "dsp-fallback"
    assert stem_feats.status == "fallback"
    assert "onnx" in (getattr(stem_feats, "fallback_reason", "") or "").lower()


def test_pipeline_onnx_mode_never_reports_engine_onnx_for_dsp(monkeypatch, tmp_path):
    import core.pipeline as pipeline

    stem_feats = StemFeatures(
        frame_times=np.linspace(0.0, 60.0, 64),
        hop_length=512,
        sample_rate=22050,
        vocal_energy=np.full(64, 0.5),
        drum_energy=np.full(64, 0.5),
        bass_energy=np.full(64, 0.5),
        melodic_energy=np.full(64, 0.5),
        backend="dsp-fallback",
        status="fallback",
    )
    sections = [Section(type=SectionType.INTRO, start=0.0, end=10.0, confidence=0.9)]
    _patch_pipeline_wiring(monkeypatch, [_cue(5, 20.0)], sections)
    monkeypatch.setattr(pipeline, "_analyze_stems", lambda **kw: (sections, stem_feats))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = analyze_track(
            str(tmp_path / "track.wav"),
            profile_name="edm",
            engine=_FakeEngine(duration=60.0),
            apply_edits=False,
            mode="onnx",
        )
    extras = result.analysis.extras
    assert extras["engine"] == "dsp-fallback"
    assert extras["backend"] == "dsp-fallback"
    assert extras["status"] == "fallback"
    assert extras["fallback"] is True
    assert extras["backendMismatch"] is True
    assert extras["fallbackFrom"] == "onnx"
    assert extras["requestedEngine"] == "onnx"
    assert result.analysis.engine == "dsp-fallback"
    assert result.analysis.engine != "onnx"


# ---------------------------------------------------------------------------
# T-1: benchmark cue matcher — label-collapse mapping (proven metric bug).
# ---------------------------------------------------------------------------


def _benchmark():
    import benchmarks.benchmark as bench

    return bench


def test_benchmark_break_cue_matches_breakdown_segment():
    bench = _benchmark()
    structure = [("INTRO", 0.25), ("BUILD", 0.55), ("DROP", 1.0), ("BREAKDOWN", 0.15), ("OUTRO", 0.2)]
    # Internal boundaries only: starts of segments 1..4 (segment 0 starts at 0).
    gt_starts = [10.0, 25.0, 40.0, 50.0]
    matched = bench._match_cue_to_ground("BREAK", 40.4, structure, gt_starts, set())
    assert matched is not None
    err, j = matched
    assert j == 3
    assert err == pytest.approx(0.4)


def test_benchmark_vocal_cue_matches_chorus_segment():
    bench = _benchmark()
    structure = [("INTRO", 0.35), ("VOCAL", 0.8), ("CHORUS", 0.95), ("OUTRO", 0.4)]
    # Internal boundaries: segment 1 starts at 10, segment 2 (CHORUS) at 25,
    # segment 3 (OUTRO) at 40. The VOCAL segment (index 1) is already used,
    # so the VOCAL cue must match the CHORUS segment at 25, not the OUTRO.
    gt_starts = [10.0, 25.0, 40.0]
    matched = bench._match_cue_to_ground("VOCAL", 25.2, structure, gt_starts, {1})
    assert matched is not None
    err, j = matched
    assert j == 2
    assert err == pytest.approx(0.2)


def test_benchmark_skips_intro_outro_and_respects_used_segments():
    bench = _benchmark()
    structure = [("INTRO", 0.25), ("DROP", 1.0), ("OUTRO", 0.2)]
    gt_starts = [10.0, 25.0]
    assert bench._match_cue_to_ground("INTRO", 0.1, structure, gt_starts, set()) is None
    assert bench._match_cue_to_ground("OUTRO", 25.0, structure, gt_starts, set()) is None
    assert bench._match_cue_to_ground("DROP", 10.2, structure, gt_starts, {1}) is None
    matched = bench._match_cue_to_ground("DROP", 10.2, structure, gt_starts, set())
    assert matched is not None and matched[1] == 1
