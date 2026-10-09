"""Tests for the Stem Intelligence Engine (SIE.md)."""

from __future__ import annotations

import numpy as np
import pytest

from core.stems import ENERGY_WEIGHTS, StemSet
from core.stems.backends import DspSeparator, available_backends, get_separator
from core.stems.features import extract_stem_features
from core.stems.sections import compute_fused_energy


def _dummy_audio(sr: int = 22050, seconds: float = 4.0):
    class _Audio:
        sample_rate = sr
        channels = 1
        duration = seconds
        mono = np.zeros(int(sr * seconds), dtype=np.float32)
        # A little content so HPSS/band filters don't divide by zero.
        mono[:] = 0.01
        t = np.arange(len(mono)) / sr
        mono += 0.2 * np.sin(2 * np.pi * 60 * t)  # bass
        mono += 0.15 * np.sin(2 * np.pi * 1200 * t)  # upper
        mono[: int(0.5 * sr)] = 0.0  # silence intro
        samples = mono

    return _Audio()


def _stems_from_dsp() -> StemSet:
    audio = _dummy_audio()
    return DspSeparator(sr=audio.sample_rate).separate(audio)


def test_separator_fallback_available():
    # No ML model installed in the repo -> DSP fallback must be available.
    assert DspSeparator().available()
    assert "dsp-fallback" in available_backends()
    assert get_separator().name in ("onnx", "dsp-fallback")


def test_dsp_separator_outputs_all_stems():
    stems = _stems_from_dsp()
    assert stems.status == "fallback"
    for name in ("vocals", "drums", "bass", "other", "instrumental"):
        arr = getattr(stems, name)
        assert arr is not None
        assert arr.ndim == 1
        assert arr.size > 0


def test_extract_stem_features_curves():
    stems = _stems_from_dsp()
    times = np.linspace(0, stems.instrumental.size / stems.sample_rate, 200)
    feats = extract_stem_features(stems, times, hop_length=512)
    assert feats.frame_times.size == times.size
    for curve in (
        feats.vocal_energy,
        feats.drum_energy,
        feats.bass_energy,
        feats.kick_activity,
        feats.percussion_density,
        feats.bass_onset,
        feats.melodic_energy,
    ):
        assert curve.size == times.size
        assert np.all(curve >= 0.0)
        assert np.all(curve <= 1.0 + 1e-6)


def test_fused_energy_normalized():
    stems = _stems_from_dsp()
    times = np.linspace(0, stems.instrumental.size / stems.sample_rate, 200)
    feats = extract_stem_features(stems, times, hop_length=512)
    energy = compute_fused_energy(feats)
    assert energy.size == times.size
    assert np.all(energy >= 0.0)
    assert np.all(energy <= 1.0 + 1e-6)
    assert float(np.max(energy)) > 0.5  # the synthetic tone is loud


def test_fused_energy_weights_configurable():
    stems = _stems_from_dsp()
    times = np.linspace(0, stems.instrumental.size / stems.sample_rate, 200)
    feats = extract_stem_features(stems, times, hop_length=512)
    custom = {**ENERGY_WEIGHTS, "drums": 0.0, "bass": 0.9}
    e1 = compute_fused_energy(feats, weights=custom)
    e2 = compute_fused_energy(feats, weights=ENERGY_WEIGHTS)
    assert e1.size == e2.size


def test_section_detection_returns_sections():
    """Full stem-aware section detection on a tiny synthetic signal."""
    from core.analyzer.librosa_engine import LibrosaEngine
    from core.beatgrid import detect_beatgrid
    from core.stems.sections import detect_stem_sections

    audio = _dummy_audio()
    engine = LibrosaEngine()
    features = engine.extract(audio)
    grid = detect_beatgrid(features)
    stems = get_separator().separate(audio)
    stem_feats = extract_stem_features(stems, features.frame_times, features.hop_length)
    result = detect_stem_sections(stem_feats, grid)
    # Either empty (too short) or a valid list of ordered sections.
    assert isinstance(result.sections, list)
    for s in result.sections:
        assert s.end > s.start
        assert 0.0 <= s.confidence <= 1.0
    # Every evidence entry must be JSON-serializable (no numpy scalars).
    for ev in result.evidence:
        assert isinstance(ev.get("dropScore", 0), float)
        assert isinstance(ev.get("phraseBoundary", False), bool)


@pytest.mark.parametrize("mode", ["basic", "intellistem", "onnx"])
def test_pipeline_mode_switches(mode):
    """analyze_track accepts both engine modes without error (dummy audio)."""
    import tempfile
    import wave

    sr = 22050
    audio = _dummy_audio(sr=sr)
    path = tempfile.mktemp(suffix=".wav")
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        data = (np.clip(audio.mono, -1, 1) * 32767).astype(np.int16)
        w.writeframes(data.tobytes())

    from core.pipeline import analyze_track

    try:
        result = analyze_track(path, profile_name="open_format", mode=mode, apply_edits=False)
        assert result.analysis is not None
        assert result.cue_plan is not None
        assert result.analysis.bpm > 0 or result.analysis.duration > 0
    finally:
        import os

        os.unlink(path)


def test_intellistem_extras_populated():
    """Intellistem analysis marks the engine/backend/status in extras."""
    import tempfile
    import wave

    sr = 22050
    audio = _dummy_audio(sr=sr)
    path = tempfile.mktemp(suffix=".wav")
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(audio.mono, -1, 1) * 32767).astype(np.int16).tobytes())

    from core.pipeline import analyze_track

    try:
        result = analyze_track(path, profile_name="open_format", mode="intellistem", apply_edits=False)
        assert result.analysis.engine == "intellistem"
        assert result.analysis.extras.get("engine") == "intellistem"
        assert result.analysis.extras.get("backend") in ("onnx", "dsp-fallback")
    finally:
        import os

        os.unlink(path)
