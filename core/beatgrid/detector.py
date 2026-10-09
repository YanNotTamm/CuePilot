"""Beatgrid detection and phrase alignment.

Detects beats, downbeats, bars, and phrases from the low-level features
produced by the analyzer. Downbeat estimation uses the classic "pick the
4-beat phase whose onsets are strongest" heuristic applied to a fixed
beats-per-bar count (default 4/4), which is robust for the electronic genres
CuePilot targets. Cue timing then snaps to the musical grid (Section 9).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..analyzer import AnalysisFeatures
from ..models import Bar, Beat, Phrase
from .subdivision import classify_subdivision

UNKNOWN_SUBDIVISION = "unknown"

BEATS_PER_BAR = 4
BARS_PER_PHRASE = 4


@dataclass
class BeatGrid:
    bpm: float
    beats: list[Beat]
    downbeat_times: list[float]
    bars: list[Bar]
    phrases: list[Phrase]
    confidence: float
    subdivision: str = "straight"
    subdivision_confidence: float = 0.0
    tempo_verification: dict = field(default_factory=dict)

    @property
    def beat_times(self) -> list[float]:
        return [b.time for b in self.beats]


def _snap_time_to_grid(
    time: float,
    grid_times: list[float],
    before: bool = False,
) -> float:
    """Snap a timestamp to the nearest (or previous) grid time."""
    if not grid_times:
        return time
    arr = np.asarray(grid_times, dtype=float)
    idx = np.searchsorted(arr, time)
    if before:
        idx = max(0, idx - 1)
        return float(arr[idx])
    if idx == 0:
        return float(arr[0])
    if idx >= len(arr):
        return float(arr[-1])
    left = arr[idx - 1]
    right = arr[idx]
    return float(left if (time - left) <= (right - time) else right)


def _estimate_downbeat_phase(beat_onsets: np.ndarray, beats_per_bar: int) -> int:
    """Choose the beat index (0..beats_per_bar-1) that maximizes mean onset.

    Onsets at the first beat of each bar (the downbeat) are typically the
    strongest, so the phase with the highest average onset is the downbeat.
    """
    if len(beat_onsets) < beats_per_bar:
        return 0
    best_phase = 0
    best_mean = -1.0
    for phase in range(beats_per_bar):
        subset = beat_onsets[phase::beats_per_bar]
        if len(subset) == 0:
            continue
        mean = float(np.mean(subset))
        if mean > best_mean:
            best_mean = mean
            best_phase = phase
    return best_phase


def detect_beatgrid(
    features: AnalysisFeatures,
    beats_per_bar: int = BEATS_PER_BAR,
    bars_per_phrase: int = BARS_PER_PHRASE,
    subdivision_hint: str | None = None,
) -> BeatGrid:
    """Detect the musical grid from analysis features.

    Args:
        features: output of an AnalyzerEngine.
        beats_per_bar: assumed meter (default 4/4).
        bars_per_phrase: number of bars per musical phrase (default 4).
        subdivision_hint: optional genre hint (e.g. "baile", "reggaeton",
            "uk_garage", "jersey_club") to bias subdivision classification
            (Genre_pattern.md §717).

    Returns:
        A BeatGrid with beats, downbeats, bars, phrases, subdivision and
        kick-snare tempo verification.
    """
    hop_length = features.hop_length
    sr = float(features.sample_rate)
    onset_env = features.onset_env
    bpm_estimate = features.bpm_estimate

    try:
        import librosa

        tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=hop_length,
            units="frames",
            bpm=None,
            trim=False,
        )
        bpm = float(np.atleast_1d(tempo)[0])
        if bpm <= 0 or not np.isfinite(bpm):
            bpm = bpm_estimate
    except Exception:
        bpm = bpm_estimate
        beat_frames = np.array([], dtype=int)

    beat_frames = np.asarray(beat_frames, dtype=int)

    if len(beat_frames) < 4:
        # Degenerate / nearly empty audio: fall back to a bare grid.
        beat_times: list[float] = []
        downbeat_times: list[float] = []
        bars: list[Bar] = []
        phrases: list[Phrase] = []
        confidence = 0.0
        if len(beat_frames) > 0:
            beat_times = [
                float(librosa.frames_to_time(f, sr=sr, hop_length=hop_length))
                for f in beat_frames
            ]
            downbeat_times = [beat_times[0]]
            bars = [Bar(index=0, downbeat_time=beat_times[0], beat_count=len(beat_times))]
        return BeatGrid(
            bpm=float(bpm) if bpm > 0 else 0.0,
            beats=[
                Beat(time=t, is_downbeat=(i in (0,) and len(beat_times) > 0), index=i)
                for i, t in enumerate(beat_times)
            ],
            downbeat_times=downbeat_times,
            bars=bars,
            phrases=phrases,
            confidence=confidence,
            subdivision=UNKNOWN_SUBDIVISION,
            subdivision_confidence=0.0,
            tempo_verification={"feel": "straight", "confidence": 0.0},
        )

    beat_times = [
        float(librosa.frames_to_time(f, sr=sr, hop_length=hop_length)) for f in beat_frames
    ]

    # Re-estimate BPM from the detected beat intervals: librosa's global tempo
    # estimator carries a systematic bias (its median is quantized to the
    # onset-frame grid), while the beat grid itself is precise. Take the mean
    # of intervals within ±20% of the median to average out frame-quantization
    # jitter while excluding doubled/missed beats.
    if len(beat_times) >= 2:
        intervals = np.diff(np.asarray(beat_times))
        valid = intervals[intervals > 0]
        if valid.size > 0:
            med = float(np.median(valid))
            core = valid[np.abs(valid - med) <= 0.20 * med]
            if core.size > 0:
                refined_bpm = 60.0 / float(np.mean(core))
                if 40.0 <= refined_bpm <= 240.0:
                    bpm = refined_bpm

    beat_onsets = np.asarray([onset_env[f] for f in beat_frames], dtype=float)
    phase = _estimate_downbeat_phase(beat_onsets, beats_per_bar)

    downbeat_indices = list(range(phase, len(beat_frames), beats_per_bar))
    downbeat_times = [beat_times[i] for i in downbeat_indices]

    bars: list[Bar] = []
    for bar_idx, db_time in enumerate(downbeat_times):
        start_idx = downbeat_indices[bar_idx]
        end_idx = (
            downbeat_indices[bar_idx + 1] if bar_idx + 1 < len(downbeat_indices) else len(beat_frames)
        )
        bars.append(
            Bar(index=bar_idx, downbeat_time=db_time, beat_count=max(0, end_idx - start_idx))
        )

    phrase_bounds: list[int] = []
    for i in range(0, len(downbeat_indices), bars_per_phrase):
        phrase_bounds.append(downbeat_indices[i])
    if len(downbeat_indices) > 0 and phrase_bounds[-1] != downbeat_indices[-1]:
        phrase_bounds.append(downbeat_indices[-1])

    phrases: list[Phrase] = []
    for p_idx in range(len(phrase_bounds) - 1):
        start_i = phrase_bounds[p_idx]
        end_i = phrase_bounds[p_idx + 1]
        end_time = (
            beat_times[end_i] if end_i < len(beat_times) else beat_times[-1]
        )
        phrases.append(
            Phrase(
                index=p_idx,
                start_time=beat_times[start_i],
                end_time=end_time,
                bar_count=max(0, end_i - start_i) // beats_per_bar,
            )
        )

    is_downbeat_set = set(downbeat_indices)
    beats = [
        Beat(time=t, is_downbeat=(i in is_downbeat_set), index=i)
        for i, t in enumerate(beat_times)
    ]

    confidence = min(1.0, max(0.0, len(beat_frames) / 64.0))
    if bpm <= 0 or not np.isfinite(bpm):
        confidence *= 0.5

    subdivision, sub_conf = classify_subdivision(features, beat_times, hint=subdivision_hint)
    tempo_verification = verify_kick_snare_tempo(features, beat_times, bpm)

    return BeatGrid(
        bpm=float(bpm),
        beats=beats,
        downbeat_times=downbeat_times,
        bars=bars,
        phrases=phrases,
        confidence=confidence,
        subdivision=subdivision,
        subdivision_confidence=sub_conf,
        tempo_verification=tempo_verification,
    )


def verify_kick_snare_tempo(
    features: AnalysisFeatures,
    beat_times: list[float],
    bpm: float,
) -> dict:
    """Verify whether the detected BPM matches the kick-snare pattern.

    Half-time/double-time ambiguity (Genre_pattern.md §717-723, Trap/Dubstep/
    Future Bass/trap-hip-hop): auto-BPM software commonly doubles or halves
    the written tempo. We probe the per-bar onset phases (kick on beats 1&3
    vs four-on-the-floor across all 4 beats) and the sub-beat pulse.

    Returns a dict describing the most consistent feel:
    ``{"feel": "straight"|"half_time"|"double_time", "confidence": 0..1}``.
    """
    bt = np.asarray(beat_times, dtype=float)
    if bt.size < 8 or bpm <= 0:
        return {"feel": "straight", "confidence": 0.0}

    onset = np.asarray(features.onset_env, dtype=float)
    times = np.asarray(features.frame_times, dtype=float)
    if onset.size == 0 or times.size == 0:
        return {"feel": "straight", "confidence": 0.0}

    period = float(np.median(np.diff(bt))) if bt.size > 1 else 60.0 / bpm
    if period <= 0:
        return {"feel": "straight", "confidence": 0.0}

    def probe(offset_frac: float) -> float:
        pts = bt[:-1] + offset_frac * period
        idx = np.clip(np.searchsorted(times, pts), 0, onset.size - 1)
        return float(np.mean(onset[idx]))

    # Per-bar onset phases: kick strength on each of the 4 beats in the bar.
    idx = np.clip(np.searchsorted(times, bt), 0, onset.size - 1)
    vals = onset[idx]
    if vals.size < 8:
        return {"feel": "straight", "confidence": 0.0}
    phases = np.arange(vals.size) % 4
    p = [float(np.mean(vals[phases == k])) if np.any(phases == k) else 0.0 for k in range(4)]
    m = max(1e-9, *p)
    norm = [x / m for x in p]

    # Straight 4/4: all four beat phases carry similar energy (kick on every
    # beat, or kick 1&3 + snare backbeat on 2&4).
    spread = max(norm) - min(norm)
    straight = float(np.clip(1.0 - spread, 0.0, 1.0))
    # Half-time (trap/dubstep): kick on 1&3 only, beat 2 weak, beat 4 weak.
    half_time = float(np.clip((norm[0] + norm[2]) / 2.0 - (norm[1] + norm[3]) / 2.0, 0.0, 1.0))
    # Double-time: strong pulse on the quarter-beat too (kick every 8th).
    sub = probe(0.25)
    double_time = float(np.clip(sub / m, 0.0, 1.0)) if m > 1e-9 else 0.0

    scores = {"straight": straight, "half_time": half_time, "double_time": double_time}
    feel = max(scores, key=lambda k: scores[k])
    conf = float(np.clip(scores[feel] - 0.5, 0.0, 0.5) * 2.0)
    return {"feel": feel, "confidence": round(conf, 3)}


def snap_to_beat(time: float, grid: BeatGrid, before: bool = False) -> float:
    """Snap a time to the nearest beat in the grid."""
    return _snap_time_to_grid(time, grid.beat_times, before=before)


def snap_to_downbeat(time: float, grid: BeatGrid, before: bool = False) -> float:
    """Snap a time to the nearest downbeat in the grid."""
    return _snap_time_to_grid(time, grid.downbeat_times, before=before)
