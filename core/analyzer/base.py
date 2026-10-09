"""Modular audio analysis engine interface.

The analyzer is deliberately engine-agnostic: CuePilot computes a set of
low-level features once, then the beatgrid / structure / cue modules consume
those features. Swapping the DSP backend (e.g. librosa -> custom C++ or an
ONNX model) only requires a new `AnalyzerEngine` implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class AudioData:
    """Raw decoded audio for a track."""

    samples: np.ndarray
    sample_rate: int
    channels: int
    duration: float

    @property
    def mono(self) -> np.ndarray:
        if self.samples.ndim == 1:
            return self.samples
        return np.mean(self.samples, axis=0)


@dataclass
class AnalysisFeatures:
    """Low-level features extracted by the engine (input to beatgrid/structure).

    Every array is aligned to the same hop_length grid as `onset_env`, except
    `frames` which carries frame-center times in seconds.
    """

    hop_length: int
    sample_rate: int = 22050
    frame_times: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    onset_env: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    rms: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    spectral_centroid: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    spectral_flux: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    chroma: np.ndarray = field(default_factory=lambda: np.zeros((12, 0), dtype=float))
    harmonic: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    percussive: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    mfcc: np.ndarray = field(default_factory=lambda: np.zeros((13, 0), dtype=float))
    subbass_energy: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    bpm_estimate: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)


class AnalyzerEngine(ABC):
    """Interface every audio analyzer must implement."""

    name: str = "base"
    version: str = "0.0.0"

    @abstractmethod
    def load(self, path: str) -> AudioData:
        """Decode the audio file at `path` into raw samples."""

    @abstractmethod
    def extract(self, audio: AudioData, progress=None) -> AnalysisFeatures:
        """Extract low-level features from decoded audio.

        `progress` is an optional callback `(stage, message, pct)` fired as
        expensive sub-steps complete, so callers can surface live progress.
        """

    def supports(self, path: str) -> bool:
        return True
