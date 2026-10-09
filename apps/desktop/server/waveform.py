"""Waveform peak computation for the desktop UI.

Produces min/max peak pairs per time bin from the raw audio, so the WebView
can draw a Serato-style stereo waveform without shipping raw samples.
"""

from __future__ import annotations

import numpy as np

from core.analyzer.librosa_engine import LibrosaEngine

_engine = LibrosaEngine()


def compute_waveform(path: str, bins: int = 3000) -> dict:
    audio = _engine.load(path)
    y = audio.mono
    n = y.shape[0]
    if n == 0:
        return {"duration": 0.0, "peaks": [], "sampleRate": audio.sample_rate}

    bin_size = max(1, n // bins)
    usable = (n // bin_size) * bin_size
    yc = y[:usable].reshape(-1, bin_size)
    mins = yc.min(axis=1).astype(float)
    maxs = yc.max(axis=1).astype(float)

    times = np.arange(mins.shape[0]) * (bin_size / float(audio.sample_rate))

    return {
        "duration": audio.duration,
        "sampleRate": audio.sample_rate,
        "peaks": [
            {"t": float(t), "min": float(mi), "max": float(mx)}
            for t, mi, mx in zip(times, mins, maxs)
        ],
    }
