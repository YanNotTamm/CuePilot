"""Regression tests for the STEM/QC fix batch (T-7).

Covers: DSP stem clipping (incl. the percussive-boost >1 gain), the
explicit dsp-fallback label (never "onnx" for DSP output), and the
documented DSP fallback confidence penalty in section detection.
"""

import numpy as np

from core.beatgrid import BeatGrid
from core.models import Beat, SectionType
from core.stems.backends import DspSeparator, get_separator
from core.stems.features import StemFeatures
from core.stems.sections import (
    DSP_FALLBACK_CONFIDENCE_PENALTY,
    _apply_fallback_confidence_penalty,
    detect_stem_sections,
)


def _loud_audio(sr=22050, seconds=2.0):
    t = np.arange(int(sr * seconds)) / sr

    class _Audio:
        sample_rate = sr
        mono = (0.9 * np.sin(2 * np.pi * 60 * t) + 0.6 * np.sin(2 * np.pi * 1200 * t)).astype(np.float32)

    return _Audio()


def test_dsp_stems_all_clipped_to_valid_range(monkeypatch):
    # Force a loud percussive envelope so the old bass formula
    # bass * (1 + 0.4*|percussive|) exceeds 1.0 before clipping.
    import core.stems.backends as backends

    def fake_hpss(y, margin=1.0):
        return y, np.full_like(y, 2.0)

    monkeypatch.setattr(backends, "_hpss", fake_hpss)
    stems = DspSeparator(sr=22050).separate(_loud_audio())
    assert stems.backend == "dsp-fallback"
    assert stems.backend != "onnx"
    assert stems.status == "fallback"
    for name in ("vocals", "drums", "bass", "other", "instrumental"):
        arr = getattr(stems, name)
        assert np.all(arr >= -1.0), name
        assert np.all(arr <= 1.0), name
    # The boost really would have exceeded 1.0 unclipped: bass hits the rail.
    assert float(np.max(np.abs(stems.bass))) <= 1.0


def test_dsp_never_labelled_onnx_when_model_missing():
    # models/stems/ is empty in the repo, so a requested "onnx" backend
    # must fall back to DSP — and report itself as dsp-fallback.
    sep = get_separator("onnx")
    assert sep.name == "dsp-fallback"


def test_fallback_penalty_applied_once():
    from core.models import Section

    secs = [Section(type=SectionType.DROP, start=0.0, end=4.0, confidence=0.8)]
    feats = StemFeatures(backend="dsp-fallback", status="fallback")
    out = _apply_fallback_confidence_penalty(secs, feats)
    assert abs(out[0].confidence - (0.8 - DSP_FALLBACK_CONFIDENCE_PENALTY)) < 1e-9
    assert DSP_FALLBACK_CONFIDENCE_PENALTY == 0.15
    # Non-fallback features are untouched.
    feats_ml = StemFeatures(backend="onnx", status="ok")
    out_ml = _apply_fallback_confidence_penalty(secs, feats_ml)
    assert out_ml[0].confidence == 0.8
    # Penalty floors at 0.
    low = [Section(type=SectionType.INTRO, start=0.0, end=4.0, confidence=0.05)]
    assert _apply_fallback_confidence_penalty(low, feats)[0].confidence == 0.0


def _synthetic_feats(backend, status):
    n = 200
    times = np.linspace(0.0, 40.0, n)
    # Energy rises halfway (a "drop" region) then stays high.
    ramp = np.concatenate([np.linspace(0.1, 0.3, 100), np.linspace(0.8, 0.9, 100)])
    return StemFeatures(
        frame_times=times,
        vocal_presence=ramp * 0.5,
        vocal_energy=ramp * 0.5,
        drum_energy=ramp,
        kick_activity=ramp,
        bass_energy=ramp,
        percussion_density=ramp * 0.5,
        harmonic_density=ramp * 0.3,
        melodic_energy=ramp * 0.4,
        onset_density=ramp * 0.3,
        backend=backend,
        status=status,
    )


def _grid():
    beats = [Beat(time=float(t), is_downbeat=(i % 4 == 0), index=i) for i, t in enumerate(np.arange(0.0, 40.0, 0.5))]
    downbeats = [float(t) for t in np.arange(0.0, 40.0, 2.0)]
    return BeatGrid(bpm=120.0, beats=beats, downbeat_times=downbeats, bars=[], phrases=[], confidence=0.9)


def test_section_detection_penalises_fallback_vs_ml():
    ml = detect_stem_sections(_synthetic_feats("onnx", "ok"), _grid())
    fb = detect_stem_sections(_synthetic_feats("dsp-fallback", "fallback"), _grid())
    assert ml.sections and fb.sections
    assert len(ml.sections) == len(fb.sections)
    for s_ml, s_fb in zip(ml.sections, fb.sections):
        assert s_fb.type == s_ml.type
        expected = max(0.0, s_ml.confidence - DSP_FALLBACK_CONFIDENCE_PENALTY)
        assert abs(s_fb.confidence - expected) < 1e-6
