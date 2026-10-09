"""Synthetic audio fixture generator for CuePilot tests.

Generates a WAV with a known BPM, beatgrid, and section layout so the
analysis pipeline can be verified deterministically.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf

SAMPLE_RATE = 22050


def _allocate_bars(n_bars: int, n_segments: int) -> list[int]:
    """Split `n_bars` into `n_segments` even (whole-bar) counts.

    Uses cumulative rounding so segments stay roughly equal and the rounding
    error is spread across all segments instead of dumped into the last one.
    """
    counts: list[int] = []
    for i in range(1, n_segments + 1):
        counts.append(round(n_bars * i / n_segments) - round(n_bars * (i - 1) / n_segments))
    return counts


def synth_track(
    bpm: float = 128.0,
    seconds: float = 96.0,
    structure: list[tuple[str, float]] | None = None,
) -> tuple[np.ndarray, float]:
    """Synthesize a mono track at `bpm` with a four-on-the-floor kick.

    `structure` is a list of (label, relative_energy) applied in order across
    the track; each segment gets its own RMS amplitude so section detection
    has clear energy differences. Segments are aligned to whole bars so the
    layout is phrase-coherent (real music changes sections on bar boundaries).
    Returns (samples, sr).
    """
    if structure is None:
        structure = [
            ("INTRO", 0.25),
            ("BUILD", 0.55),
            ("DROP", 1.00),
            ("BREAKDOWN", 0.15),
            ("BUILD", 0.60),
            ("SECOND_DROP", 1.00),
            ("OUTRO", 0.20),
        ]

    beat_period = 60.0 / bpm
    n_beats = int(seconds / beat_period)
    n_bars = n_beats // 4
    if n_bars < len(structure):
        raise ValueError("track too short for the requested structure")

    # Allocate bars to segments (proportional, whole bars only).
    bar_counts = _allocate_bars(n_bars, len(structure))
    seg_starts_beats = [0]
    for c in bar_counts:
        seg_starts_beats.append(seg_starts_beats[-1] + c * 4)

    n_samples = int(seconds * SAMPLE_RATE)
    t = np.arange(n_samples) / SAMPLE_RATE
    out = np.zeros(n_samples, dtype=np.float64)

    for seg_idx, (label, amp) in enumerate(structure):
        start = seg_starts_beats[seg_idx] * beat_period
        end = seg_starts_beats[seg_idx + 1] * beat_period
        start_s = int(start * SAMPLE_RATE)
        end_s = min(int(end * SAMPLE_RATE), n_samples)
        seg_t = t[start_s:end_s]

        # Real builds ramp upward toward the drop (which is still the energy
        # peak), so render an upward slope that stays below drop level.
        if label in ("BUILD",):
            seg_amp = np.linspace(0.5 * amp, 0.95 * amp, end_s - start_s)
        else:
            seg_amp = amp

        kick = _kick_train(seg_t, beat_period, seg_amp)
        # Downbeat accent: every 4th kick is louder + has a sub-bass thump so
        # the bar phase is unambiguous ground truth for downbeat detection.
        bar_phase = _bar_phase(seg_t, beat_period)
        accent = 1.0 + 0.5 * bar_phase
        sub = 0.22 * seg_amp * bar_phase * np.sin(2 * np.pi * 55.0 * seg_t)
        hat = 0.12 * seg_amp * _offbeat_hats(seg_t, beat_period)
        tone = 0.08 * seg_amp * np.sin(2 * np.pi * 220.0 * seg_t)
        out[start_s:end_s] = kick * accent + sub + hat + tone

    # 20ms fade in/out to avoid clicks.
    fade = int(0.02 * SAMPLE_RATE)
    out[:fade] *= np.linspace(0.0, 1.0, fade)
    out[-fade:] *= np.linspace(1.0, 0.0, fade)
    peak = np.max(np.abs(out)) or 1.0
    out = out / peak * 0.9
    return out.astype(np.float32), float(SAMPLE_RATE)


def _kick_train(t: np.ndarray, beat_period: float, amp: float) -> np.ndarray:
    phase = np.mod(t, beat_period)
    kick = np.exp(-phase * 60.0)
    return amp * kick


def _offbeat_hats(t: np.ndarray, beat_period: float) -> np.ndarray:
    half = beat_period / 2.0
    phase = np.mod(t + half, beat_period)
    hat = np.exp(-phase * 120.0)
    return hat


def _bar_phase(t: np.ndarray, beat_period: float) -> np.ndarray:
    """1.0 on every 4th beat (downbeat), 0.0 elsewhere."""
    beat_idx = np.floor(t / beat_period).astype(int)
    return np.where(np.mod(beat_idx, 4) == 0, 1.0, 0.0)


def structure_boundaries(
    bpm: float, seconds: float, structure: list[tuple[str, float]]
) -> list[float]:
    """Ground-truth section start times (seconds), aligned to whole bars.

    Mirrors the bar allocation in `synth_track` so tests/benchmarks can assert
    exact boundary positions. Returns the internal boundaries only
    (len(structure) - 1 values), excluding t=0 and the end of the track.
    """
    beat_period = 60.0 / bpm
    n_beats = int(seconds / beat_period)
    n_bars = n_beats // 4
    bar_counts = _allocate_bars(n_bars, len(structure))
    starts = [0.0]
    for c in bar_counts:
        starts.append(starts[-1] + c * 4 * beat_period)
    return starts[1:-1]


def write_synth_track(
    path: str,
    bpm: float = 128.0,
    seconds: float = 96.0,
    structure=None,
) -> str:
    samples, sr = synth_track(bpm=bpm, seconds=seconds, structure=structure)
    sf.write(path, samples, int(sr), subtype="PCM_16")
    return path
