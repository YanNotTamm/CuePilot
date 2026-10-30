"""Tests for Audio QC and Gig Readiness."""

import numpy as np

from core.analyzer.base import AnalysisFeatures, AudioData
from core.models import Cue, CueType, Section, SectionType
from core.quality import compute_qc_report, compute_readiness, readiness_bucket
from core.quality.qc import _dbfs_rms
from core.quality.readiness import WEIGHTS


def _audio(samples: np.ndarray, sr: int = 22050, channels: int = 1) -> AudioData:
    if samples.ndim == 1:
        samples = samples[np.newaxis, :]
    return AudioData(samples=samples, sample_rate=sr, channels=channels, duration=len(samples[0]) / sr)


def _features(rms_value: float = 0.5, size: int = 100) -> AnalysisFeatures:
    return AnalysisFeatures(
        hop_length=512,
        sample_rate=22050,
        frame_times=np.linspace(0.0, 10.0, size),
        rms=np.full(size, rms_value),
        onset_env=np.full(size, 0.5),
    )


def _cue(confidence: float) -> Cue:
    return Cue(slot=1, type=CueType.DROP, time=5.0, label="D", confidence=confidence, color="#fff")


def _sections() -> list[Section]:
    return [
        Section(type=SectionType.INTRO, start=0.0, end=2.0, confidence=0.9),
        Section(type=SectionType.DROP, start=4.0, end=6.0, confidence=0.9),
        Section(type=SectionType.OUTRO, start=8.0, end=10.0, confidence=0.9),
    ]


# -- QC report ------------------------------------------------------------


def test_qc_clean_audio_scores_high():
    t = np.linspace(0.0, 10.0, 22050 * 10)
    audio = _audio(0.5 * np.sin(2 * np.pi * 440 * t))
    report = compute_qc_report(audio, _features())
    assert report.score >= 0.7
    assert report.ok


def test_qc_detects_clipping():
    t = np.linspace(0.0, 10.0, 22050 * 10)
    audio = _audio(np.clip(0.5 * np.sin(2 * np.pi * 440 * t) * 4.0, -1.0, 1.0))
    report = compute_qc_report(audio, _features())
    codes = [i.code for i in report.issues]
    assert "clipping" in codes
    assert report.score < 1.0


def test_qc_mono_compatibility_warning():
    t = np.linspace(0.0, 10.0, 22050 * 10)
    # Anti-phase stereo → near-zero correlation.
    left = np.sin(2 * np.pi * 440 * t)
    right = -left
    audio = _audio(np.stack([left, right]), channels=2)
    report = compute_qc_report(audio, _features())
    codes = [i.code for i in report.issues]
    assert "mono_compat" in codes


def test_qc_loudness_inconsistency_flagged_with_library():
    t = np.linspace(0.0, 10.0, 22050 * 10)
    audio = _audio(0.5 * np.sin(2 * np.pi * 440 * t))
    loud = _dbfs_rms(audio.mono)
    report = compute_qc_report(audio, _features(), library_dbfs_rms=loud + 10.0)
    codes = [i.code for i in report.issues]
    assert "loudness" in codes
    # The metric is dBFS RMS, never LUFS.
    assert "dbfsRms" in report.metrics
    assert "lufs" not in report.metrics


def test_qc_edge_silence_detected():
    size = 200
    rms = np.zeros(size)
    rms[40:160] = 0.5
    feats = AnalysisFeatures(
        hop_length=512,
        sample_rate=22050,
        frame_times=np.linspace(0.0, 10.0, size),
        rms=rms,
        onset_env=np.full(size, 0.5),
    )
    t = np.linspace(0.0, 10.0, 22050 * 10)
    audio = _audio(0.5 * np.sin(2 * np.pi * 440 * t))
    report = compute_qc_report(audio, feats)
    assert report.metrics.get("startSilence", 0.0) > 1.5
    assert report.metrics.get("endSilence", 0.0) > 1.5


def test_qc_high_freq_uses_whole_track_not_just_intro():
    # Bright content only AFTER 30s: the old first-12s window would report
    # almost no high-frequency energy; the whole-track windows must see it.
    from core.quality.qc import _high_freq_rolloff

    sr = 22050
    t = np.linspace(0.0, 60.0, sr * 60)
    sig = np.zeros_like(t)
    sig[sr * 40 :] = 0.5 * np.sin(2 * np.pi * 12000 * t[sr * 40 :])
    assert _high_freq_rolloff(sig, sr) > 0.05


def test_qc_edge_silence_symmetric_when_nothing_audible():
    from core.quality.qc import _edge_silence

    feats = AnalysisFeatures(
        hop_length=512,
        sample_rate=22050,
        frame_times=np.linspace(0.0, 10.0, 100),
        rms=np.zeros(100),
        onset_env=np.zeros(100),
    )
    start, end = _edge_silence(feats)
    assert start == end == 10.0  # was (10.0, 0.0) — asymmetric


def test_qc_report_to_dict_serializable():
    t = np.linspace(0.0, 10.0, 22050 * 10)
    report = compute_qc_report(_audio(np.sin(2 * np.pi * 440 * t)), _features())
    data = report.to_dict()
    assert isinstance(data["score"], float)
    assert isinstance(data["issues"], list)
    assert isinstance(data["metrics"], dict)


# -- Gig Readiness --------------------------------------------------------


def test_readiness_not_analyzed():
    r = compute_readiness()
    assert r.bucket == "not_analyzed"
    assert r.score == 0.0


def test_readiness_ready_bucket():
    # Consolidation (T-5): with the new verification policy, an
    # unverified plan scores 0.8 on the verification component, so
    # this borderline case (QC 0.58: clipping + transcode warnings on a
    # full-amplitude sine) scores 76.0 and is correctly needs_review
    # without human verification. A human-verified version of the same
    # plan (4 corrections -> verification 1.0) scores exactly 80.0 and
    # is ready. See test_readiness_unverified_borderline_needs_review.
    t = np.linspace(0.0, 10.0, 22050 * 10)
    qc = compute_qc_report(_audio(np.sin(2 * np.pi * 440 * t)), _features())
    cues = [_cue(0.95), _cue(0.85)]
    r = compute_readiness(cues=cues, sections=_sections(), qc=qc, correction_count=4)
    assert r.bucket == "ready"
    assert r.score >= READY_MIN


def test_readiness_unverified_borderline_needs_review():
    # Same borderline plan, unverified: must NOT be labelled ready.
    t = np.linspace(0.0, 10.0, 22050 * 10)
    qc = compute_qc_report(_audio(np.sin(2 * np.pi * 440 * t)), _features())
    cues = [_cue(0.95), _cue(0.85)]
    r = compute_readiness(cues=cues, sections=_sections(), qc=qc, correction_count=0)
    assert r.bucket == "needs_review"
    assert r.score < READY_MIN


READY_MIN = 80.0


def test_readiness_corrections_raise_score():
    # Policy change (consolidation, T-5): human corrections no longer
    # penalise readiness. No corrections = 0.8 verification component
    # (DSP only, unverified); each correction adds 0.05 up to 1.0, so a
    # human-verified plan must score at least as high as the unverified
    # one. This replaces the old test_readiness_corrections_reduce_score,
    # which asserted the previous (punitive) behaviour.
    t = np.linspace(0.0, 10.0, 22050 * 10)
    qc = compute_qc_report(_audio(np.sin(2 * np.pi * 440 * t)), _features())
    cues = [_cue(0.95), _cue(0.85)]
    base = compute_readiness(cues=cues, sections=_sections(), qc=qc, correction_count=0)
    corrected = compute_readiness(cues=cues, sections=_sections(), qc=qc, correction_count=4)
    assert corrected.score >= base.score


def test_readiness_component_weights_sum():
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


def test_readiness_bucket_thresholds():
    assert readiness_bucket(90.0) == "ready"
    assert readiness_bucket(60.0) == "needs_review"
    assert readiness_bucket(10.0) == "not_analyzed"


def test_readiness_to_dict():
    r = compute_readiness(cues=[_cue(0.9)], sections=_sections())
    data = r.to_dict()
    assert data["score"] >= 0.0
    assert data["bucket"] in ("ready", "needs_review", "not_analyzed")
    assert "components" in data and "reasons" in data