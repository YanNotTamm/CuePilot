"""Beatgrid detection and phrase alignment.

Detects beats, downbeats, bars, and phrases from the low-level features
produced by the analyzer. Downbeat estimation uses the classic "pick the
4-beat phase whose onsets are strongest" heuristic applied to a fixed
beats-per-bar count (default 4/4), which is robust for the electronic genres
CuePilot targets. Cue timing then snaps to the musical grid (Section 9).

KNOWN LIMITATIONS (T-4, stated rather than papered over):
- Meter is ASSUMED, not detected: `beats_per_bar` is caller-supplied
  (default 4) and BeatGrid.meter_detected is always False. A 3/4, 6/8 or
  other-meter track will get a wrong 4/4 grid unless the caller passes
  the right beats_per_bar.
- Tempo is GLOBAL, not tracked: one BPM is refined from the median beat
  interval and applied to the whole track. There is NO variable-tempo
  tracking; a track whose tempo drifts will have a grid that drifts
  with it. BeatGrid.tempo_model records this ("global-single-tempo").
- Subdivision is INFORMATIONAL only: classify_subdivision() runs and its
  result is stored on the grid, but snapping (snap_to_beat /
  snap_to_downbeat) uses the beat/downbeat grid only and does not adapt
  to triplet/tresillo/etc. grids. See subdivision.py.
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
    # T-4: explicit scope fields. meter is the assumed meter string (e.g.
    # "4/4 (assumed, not detected)"); meter_detected is always False
    # because no meter detection is implemented. tempo_model is
    # "global-single-tempo": one BPM for the whole track, no
    # variable-tempo tracking.
    meter: str = "4/4 (assumed, not detected)"
    meter_detected: bool = False
    tempo_model: str = "global-single-tempo"

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


def beatgrid_confidence(
    beat_times: list[float] | np.ndarray,
    beat_onset_values: list[float] | np.ndarray | None = None,
    onset_max: float | None = None,
) -> float:
    """Beatgrid confidence from interval consistency + onset strength (T-5).

    Replaces the old ``min(1, len(beat_frames) / 64)`` formula, which
    rewarded merely having many beats regardless of whether they formed
    a steady grid. Returns a documented 0..1 value:

    - interval consistency: 1 - CV of the beat intervals, where CV is
      std/mean over ALL positive intervals. Deliberately not the
      median-core subset used for BPM refinement: core filtering would
      score an alternating 0.4s/0.6s jitter as perfect by discarding
      half the intervals, whereas a steady grid must be steady on
      every interval. A perfectly regular grid scores 1; wildly
      irregular intervals score near 0. Fewer than 2 usable intervals
      score 0 (no regularity evidence at all).
    - onset strength: mean onset at beat positions divided by the peak
      onset (``onset_max`` if given, else the max of the beat values),
      clipped to 0..1. Strong, clear beats score higher.
    - combination: 0.7 * consistency + 0.3 * onset strength when onset
      values are available, otherwise consistency alone (no invented
      onset evidence).

    Pure function of its inputs: testable on synthetic beat times
    without audio or librosa.
    """
    times = np.asarray(beat_times, dtype=float)
    if times.size < 3:
        return 0.0
    intervals = np.diff(times)
    intervals = intervals[intervals > 0]
    if intervals.size < 2:
        return 0.0
    mean = float(np.mean(intervals))
    if mean <= 0:
        return 0.0
    cv = float(np.std(intervals)) / mean
    consistency = float(np.clip(1.0 - cv, 0.0, 1.0))
    if beat_onset_values is None:
        return consistency
    onsets = np.asarray(beat_onset_values, dtype=float)
    if onsets.size == 0:
        return consistency
    peak = float(onset_max) if onset_max is not None else float(np.max(onsets))
    if peak <= 1e-9:
        return consistency
    strength = float(np.clip(float(np.mean(onsets)) / peak, 0.0, 1.0))
    return float(np.clip(0.7 * consistency + 0.3 * strength, 0.0, 1.0))


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
        beats_per_bar: ASSUMED meter (default 4/4). This is an assumption
            supplied by the caller, not a detected value; see the module
            docstring's KNOWN LIMITATIONS. No variable-tempo tracking is
            performed: a single global BPM is estimated for the track.
        bars_per_phrase: number of bars per musical phrase (default 4).
        subdivision_hint: optional genre hint (e.g. "baile", "reggaeton",
            "uk_garage", "jersey_club") to bias subdivision classification
            (Genre_pattern.md §717). The hint only affects the stored
            subdivision label; it does not change beat/downbeat snapping.

    Returns:
        A BeatGrid with beats, downbeats, bars, phrases, subdivision and
        kick-snare tempo verification. Its confidence is the documented
        interval-consistency metric from beatgrid_confidence(), not a
        beat count.
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
            meter=f"{beats_per_bar}/4 (assumed, not detected)",
            meter_detected=False,
            tempo_model="global-single-tempo",
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

    confidence = beatgrid_confidence(beat_times, beat_onsets)
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
        meter=f"{beats_per_bar}/4 (assumed, not detected)",
        meter_detected=False,
        tempo_model="global-single-tempo",
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
