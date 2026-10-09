"""Stem Intelligence Engine (SIE.md) - separation layer.

Implements SIE.md sections 4-10: a pluggable stem separator whose output
feeds the feature / energy / section / hotcue engines. Backends are not
hard-coded; the runtime auto-detects the best available accelerator and
model (ONNX / Demucs / UVR), and always degrades gracefully to a DSP-based
approximation so CuePilot stays usable without an ML model (SIE section 50).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class StemSet:
    """The four minimum stems (SIE section 5), each mono and time-aligned."""

    vocals: np.ndarray
    drums: np.ndarray
    bass: np.ndarray
    other: np.ndarray
    instrumental: Optional[np.ndarray] = None
    sample_rate: int = 22050
    backend: str = "dsp-fallback"
    status: str = "ok"  # "ok" | "failed" | "fallback"

    @property
    def names(self) -> list[str]:
        return ["vocals", "drums", "bass", "other", "instrumental"]


class StemSeparator:
    """Interface every stem separation backend must implement (SIE section 6)."""

    name: str = "base"

    def available(self) -> bool:
        """Whether this backend can run right now (model present, deps installed)."""
        return False

    def separate(self, audio, progress: Optional[Callable] = None) -> StemSet:
        """Separate raw audio into a StemSet (SIE section 6)."""
        raise NotImplementedError


class Acceleration:
    """Detected hardware acceleration (SIE section 8)."""

    provider = "cpu"
    device = "CPU"

    @classmethod
    def detect(cls) -> "Acceleration":
        acc = cls()
        try:
            import onnxruntime as ort  # type: ignore

            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in providers:
                acc.provider = "cuda"
                acc.device = "CUDA GPU"
            elif "DmlExecutionProvider" in providers:
                acc.provider = "dml"
                acc.device = "DirectML"
        except Exception:
            pass
        return acc


def get_acceleration() -> Acceleration:
    return Acceleration.detect()


# Tunable fusion weights (SIE section 17).
ENERGY_WEIGHTS: dict[str, float] = {
    "rms": 0.20,
    "spectral": 0.10,
    "drums": 0.25,
    "bass": 0.20,
    "onset": 0.10,
    "stem_activity": 0.15,
}
