"""Audio Quality / QC Report.

Non-destructive checks computed from the same analysis pass:

- clipping detection (peak / near-full-scale sample ratio)
- loudness inconsistency (whole-signal dBFS RMS vs library average —
  this is NOT LUFS: there is no K-weighting and no gating, so it must
  never be labelled or reported as LUFS)
- mono-compatibility issue (L/R correlation)
- extremely low bitrate / transcode artifact (high-frequency roll-off)
- silence at start/end (mismatch with intro/outro cue expectation)

Everything is computed in-memory from the already-decoded audio + features,
so the report is effectively free to produce during a normal analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..analyzer.base import AnalysisFeatures, AudioData


@dataclass
class QcIssue:
    """A single non-destructive quality finding."""

    code: str
    severity: str  # "error" | "warning" | "info"
    label: str
    detail: str

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "severity": self.severity,
            "label": self.label,
            "detail": self.detail,
        }


@dataclass
class QcReport:
    """Result of the audio QC pass."""

    score: float
    issues: list[QcIssue] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.score >= 0.75

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 3),
            "ok": self.ok,
            "issues": [i.to_dict() for i in self.issues],
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
        }


# Severity weight pulled from the 0..1 score.
_SEVERITY_PENALTY = {"error": 0.30, "warning": 0.12, "info": 0.03}

_EDGE_SILENCE_THRESHOLD_DB = -46.0
"""RMS below this (dBFS) counts as silence for intro/outro checks."""


def _dbfs_rms(mono: np.ndarray) -> float:
    """Whole-signal RMS level in dBFS (NOT LUFS).

    This is a plain full-band RMS mapped to dBFS: no K-weighting, no
    gating, no absolute-gate loudness integration as in BS.1770. It is
    stable enough for library-level consistency comparison, but it must
    never be presented as LUFS / LUFS-I. Formerly named ``_lufs_approx``;
    renamed because that name mislabelled the measurement.
    """
    if mono.size == 0:
        return -90.0
    rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))
    if rms <= 1e-12:
        return -90.0
    return 20.0 * np.log10(rms)


def _clipping_ratio(mono: np.ndarray, threshold: float = 0.999) -> float:
    if mono.size == 0:
        return 0.0
    return float(np.mean(np.abs(mono) >= threshold))


def _correlation(l: np.ndarray, r: np.ndarray) -> float:
    """Normalized cross-correlation between two stereo channels."""
    if l.size == 0 or r.size == 0:
        return 1.0
    l = l - l.mean()
    r = r - r.mean()
    denom = float(np.sqrt(np.dot(l, l) * np.dot(r, r)))
    if denom <= 1e-12:
        return 1.0
    return float(np.clip(np.dot(l, r) / denom, -1.0, 1.0))


def _high_freq_rolloff(mono: np.ndarray, sample_rate: int) -> float:
    """Fraction of spectral energy above 9 kHz (transcode/bitrate proxy).

    Heavily compressed / low-bitrate files typically have a sharp high-
    frequency roll-off. Previously this only looked at ~the first 12
    seconds, so a track whose intro is dull (or bright) was misjudged.
    Now up to 32 FFT windows are spread evenly across the WHOLE track
    and their magnitude spectra are summed, so the result represents
    the full song, not just its opening.
    """
    if mono.size < 512 or sample_rate <= 0:
        return 0.0
    window = np.hanning(2048)
    max_windows = 32
    last_start = mono.size - 2048
    if last_start <= 0:
        starts = [0]
    else:
        n_windows = min(max_windows, last_start // 512 + 1)
        if n_windows <= 1:
            starts = [0]
        else:
            starts = [int(round(i * last_start / (n_windows - 1))) for i in range(n_windows)]
    spec_sum: np.ndarray | None = None
    for start in starts:
        seg = mono[start : start + 2048]
        if seg.size < 2048:
            break
        mag = np.abs(np.fft.rfft(seg * window))
        spec_sum = mag if spec_sum is None else spec_sum + mag
    if spec_sum is None:
        return 0.0
    freqs = np.fft.rfftfreq(2048, d=1.0 / sample_rate)
    total = float(np.sum(spec_sum))
    if total <= 1e-12:
        return 0.0
    high = float(np.sum(spec_sum[freqs >= 9000.0]))
    return high / total


def _edge_silence(
    features: AnalysisFeatures,
) -> tuple[float, float]:
    """Return (start_silence, end_silence) seconds where RMS < threshold.

    Symmetric by design: if NO frame is audible, the whole track counts
    as silence at both edges, so both values equal the track duration
    (last frame time). Previously this returned (duration, 0.0), which
    asymmetrically hid trailing silence. Empty features (no frames at
    all) still return (0.0, 0.0) because no duration is known.
    """
    rms = features.rms
    times = features.frame_times
    if rms.size == 0 or times.size == 0:
        return 0.0, 0.0
    db = 20.0 * np.log10(rms + 1e-12)
    audible = db >= _EDGE_SILENCE_THRESHOLD_DB
    if not np.any(audible):
        duration = float(times[-1])
        return duration, duration
    idx = np.where(audible)[0]
    first = times[idx[0]]
    last = times[idx[-1]]
    duration = float(times[-1])
    return max(0.0, first), max(0.0, duration - last)


def compute_qc_report(
    audio: AudioData,
    features: AnalysisFeatures | None = None,
    library_dbfs_rms: float | None = None,
    library_lufs: float | None = None,
) -> QcReport:
    """Run all non-destructive QC checks on decoded audio + features.

    Args:
        audio: decoded audio (any channel count).
        features: optional extracted features (enables edge-silence checks).
        library_dbfs_rms: optional library-average dBFS RMS to compare
            against (loudness inconsistency check). When omitted, the
            check reports the value but never flags an inconsistency.
        library_lufs: DEPRECATED alias for ``library_dbfs_rms``, kept
            only so old callers do not crash; the value is dBFS RMS,
            not LUFS, despite the parameter name. New code must use
            ``library_dbfs_rms``.
    """
    issues: list[QcIssue] = []
    metrics: dict[str, float] = {}

    mono = audio.mono
    if mono.size == 0:
        return QcReport(score=0.0, issues=issues, metrics=metrics)

    # -- clipping ----------------------------------------------------------
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    clip_ratio = _clipping_ratio(mono)
    metrics["peak"] = peak
    metrics["clippingRatio"] = clip_ratio
    if clip_ratio > 0.001:
        issues.append(
            QcIssue(
                "clipping",
                "error",
                "Clipping detected",
                f"{clip_ratio * 100:.2f}% of samples saturate near full scale",
            )
        )
    elif peak >= 0.999:
        issues.append(
            QcIssue("clipping", "warning", "Near-full-scale peaks", f"peak {peak:.3f}")
        )

    # -- loudness (dBFS RMS — NOT LUFS) -----------------------------------
    import warnings as _warnings

    if library_lufs is not None and library_dbfs_rms is None:
        _warnings.warn(
            "library_lufs is deprecated; pass library_dbfs_rms instead "
            "(the value is dBFS RMS, not LUFS).",
            DeprecationWarning,
            stacklevel=2,
        )
        library_dbfs_rms = library_lufs
    dbfs_rms = _dbfs_rms(mono)
    metrics["dbfsRms"] = dbfs_rms
    if library_dbfs_rms is not None and abs(dbfs_rms - library_dbfs_rms) > 4.0:
        issues.append(
            QcIssue(
                "loudness",
                "warning",
                "Loudness inconsistent with library",
                f"{dbfs_rms:.1f} dBFS RMS vs library {library_dbfs_rms:.1f} dBFS RMS",
            )
        )

    # -- mono compatibility -----------------------------------------------
    if audio.channels >= 2 and audio.samples.ndim == 2:
        corr = _correlation(audio.samples[0], audio.samples[1])
        metrics["stereoCorrelation"] = corr
        if corr < 0.3:
            issues.append(
                QcIssue(
                    "mono_compat",
                    "error",
                    "Mono-compatibility issue",
                    f"L/R correlation {corr:.2f} — collapses to mono",
                )
            )
        elif corr < 0.6:
            issues.append(
                QcIssue(
                    "mono_compat",
                    "warning",
                    "Weak mono compatibility",
                    f"L/R correlation {corr:.2f}",
                )
            )

    # -- transcode / low bitrate ------------------------------------------
    high_ratio = _high_freq_rolloff(mono, audio.sample_rate)
    metrics["highFreqRatio"] = high_ratio
    if high_ratio < 0.005:
        issues.append(
            QcIssue(
                "transcode",
                "warning",
                "Low bitrate / transcode artifact",
                f"almost no energy above 9 kHz ({high_ratio * 100:.2f}%)",
            )
        )
    elif high_ratio < 0.02:
        issues.append(
            QcIssue(
                "transcode",
                "info",
                "Reduced high-frequency content",
                f"{high_ratio * 100:.2f}% of energy above 9 kHz",
            )
        )

    # -- edge silence ------------------------------------------------------
    if features is not None:
        start_sil, end_sil = _edge_silence(features)
        metrics["startSilence"] = start_sil
        metrics["endSilence"] = end_sil
        if start_sil > 2.0:
            issues.append(
                QcIssue(
                    "edge_silence",
                    "info",
                    "Silence at start",
                    f"{start_sil:.1f}s before audio begins",
                )
            )
        if end_sil > 2.0:
            issues.append(
                QcIssue(
                    "edge_silence",
                    "info",
                    "Silence at end",
                    f"{end_sil:.1f}s of trailing silence",
                )
            )

    # -- score -------------------------------------------------------------
    score = 1.0
    for issue in issues:
        score -= _SEVERITY_PENALTY.get(issue.severity, 0.0)
    score = float(np.clip(score, 0.0, 1.0))

    return QcReport(score=round(score, 3), issues=issues, metrics=metrics)