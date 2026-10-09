"""Stem feature extraction (SIE sections 11-15).

Computes time-varying features for every stem: vocal presence/energy/density,
drum energy + kick/snare/hat activity, bass energy/presence/re-entry, and
melodic/harmonic density for "other". Every curve is aligned to the same
hop grid as the analyzer features so downstream fusion is element-wise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import StemSet

# Sensor names (SIE section 55: "stems are sensors").
SENSORS = ["vocal", "drums", "bass", "other"]


@dataclass
class StemFeatures:
    """Time-aligned feature curves for all stems."""

    frame_times: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    hop_length: int = 512
    sample_rate: int = 22050

    # vocals (SIE section 12)
    vocal_presence: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    vocal_energy: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    vocal_density: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    vocal_onset: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))

    # drums (SIE section 13)
    drum_energy: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    kick_activity: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    snare_activity: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    hat_activity: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    percussion_density: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    onset_density: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))

    # bass (SIE section 14)
    bass_energy: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    bass_presence: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    bass_onset: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))

    # other / instrumental (SIE section 15)
    melodic_energy: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    harmonic_density: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))

    backend: str = "dsp-fallback"
    status: str = "ok"

    _ARRAY_FIELDS: tuple[str, ...] = (
        "frame_times",
        "vocal_presence",
        "vocal_energy",
        "vocal_density",
        "vocal_onset",
        "drum_energy",
        "kick_activity",
        "snare_activity",
        "hat_activity",
        "percussion_density",
        "onset_density",
        "bass_energy",
        "bass_presence",
        "bass_onset",
        "melodic_energy",
        "harmonic_density",
    )

    def to_arrays(self) -> dict[str, np.ndarray]:
        """All feature arrays keyed by field name (for caching, SIE §46)."""
        return {name: np.asarray(getattr(self, name), dtype=float) for name in self._ARRAY_FIELDS}

    @classmethod
    def from_arrays(
        cls,
        arrays: dict[str, np.ndarray],
        hop_length: int = 512,
        sample_rate: int = 22050,
        backend: str = "dsp-fallback",
        status: str = "ok",
    ) -> "StemFeatures":
        """Rebuild a StemFeatures from cached arrays (SIE §46)."""
        kwargs = {name: np.asarray(arrays.get(name, []), dtype=float) for name in cls._ARRAY_FIELDS}
        return cls(
            **kwargs,
            hop_length=hop_length,
            sample_rate=sample_rate,
            backend=backend,
            status=status,
        )

    def to_dict(self) -> dict:
        """Compact dict suitable for debugging / the analysis payload."""

        def _arr(a: np.ndarray) -> list[float]:
            return [round(float(x), 4) for x in np.asarray(a).reshape(-1)]

        return {
            "backend": self.backend,
            "status": self.status,
            "frameCount": int(np.asarray(self.frame_times).size),
            "vocalEnergy": _arr(self.vocal_energy),
            "vocalPresence": _arr(self.vocal_presence),
            "drumEnergy": _arr(self.drum_energy),
            "bassEnergy": _arr(self.bass_energy),
            "melodicEnergy": _arr(self.melodic_energy),
            "percussionDensity": _arr(self.percussion_density),
        }


def _frame_rms(y: np.ndarray, hop_length: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if y.size == 0:
        return np.array([], dtype=float)
    n = max(1, 1 + (y.size - 1) // hop_length)
    frames = np.zeros(n, dtype=float)
    for i in range(n):
        seg = y[i * hop_length : (i + 1) * hop_length]
        if seg.size:
            frames[i] = np.sqrt(np.mean(seg**2))
    return frames


def _band_energy(y: np.ndarray, sr: int, hop_length: int, lo: float, hi: float) -> np.ndarray:
    """RMS of the signal band-passed to [lo, hi] Hz (per-frame, hop-aligned)."""
    from scipy.signal import butter, sosfiltfilt

    nyq = sr / 2
    freqs = [max(1e-6, lo / nyq), min(0.999, hi / nyq)]
    if freqs[0] >= freqs[1]:
        return _frame_rms(y, hop_length)
    try:
        sos = butter(4, freqs, btype="bandpass", output="sos")
        filtered = sosfiltfilt(sos, np.asarray(y, dtype=np.float64).reshape(-1))
        return _frame_rms(filtered, hop_length)
    except Exception:
        return _frame_rms(y, hop_length)


def _onset_curve(y: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    """Per-frame onset strength of a stem."""
    try:
        import librosa

        env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        return np.asarray(env, dtype=float)
    except Exception:
        return _frame_rms(y, hop_length)


def _sliding_density(curve: np.ndarray, window: int = 12) -> np.ndarray:
    """Local sparsity: fraction of recent frames above a noise floor."""
    if curve.size == 0:
        return curve
    w = max(1, min(window, curve.size))
    kernel = np.ones(w) / w
    thresh = max(float(np.percentile(curve, 85)), 1e-6)
    active = (np.asarray(curve) > thresh * 0.4).astype(float)
    return np.convolve(active, kernel, mode="same")


def _resample(a: np.ndarray, n: int) -> np.ndarray:
    """Block-average resample a curve to `n` samples (keep aligned)."""
    a = np.asarray(a, dtype=float)
    if a.ndim != 1 or a.size == 0:
        return np.zeros(n, dtype=float)
    if n <= 0:
        return a.copy()
    if a.size == n:
        return a.copy()
    out = np.zeros(n, dtype=float)
    for i in range(n):
        lo = int(i * a.size / n)
        hi = int((i + 1) * a.size / n)
        seg = a[lo:max(hi, lo + 1)]
        out[i] = float(np.mean(seg))
    return out


def extract_stem_features(
    stems: StemSet,
    frame_times: np.ndarray,
    hop_length: int = 512,
) -> StemFeatures:
    """Derive time-aligned sensor curves from a StemSet (SIE 11-15)."""
    sr = stems.sample_rate or 22050
    v = stems.vocals
    d = stems.drums
    b = stems.bass
    o = stems.other

    vocal_rms = _frame_rms(v, hop_length)
    drum_rms = _frame_rms(d, hop_length)
    bass_rms = _frame_rms(b, hop_length)
    other_rms = _frame_rms(o, hop_length)
    mix_rms = _frame_rms(v + d + b + o, hop_length)

    n = int(np.asarray(frame_times).size)
    if n == 0:
        n = int(mix_rms.size)

    def _norm(x: np.ndarray) -> np.ndarray:
        x = _resample(x, n)
        m = float(np.max(x)) if x.size else 0.0
        if m <= 1e-9:
            return np.zeros(n, dtype=float)
        return np.clip(x / m, 0.0, 1.0)

    # Percussive activity by band (kick ~40-90Hz, snare ~150-400Hz, hats ~6-10k).
    kick = _norm(_band_energy(d, sr, hop_length, 40, 90))
    snare = _norm(_band_energy(d, sr, hop_length, 150, 400))
    hat = _norm(_band_energy(d, sr, hop_length, 6000, 10000))

    drum_onset = _norm(_onset_curve(d, sr, hop_length))
    bass_onset = _norm(_onset_curve(b, sr, hop_length))
    vocal_onset = _norm(_onset_curve(v, sr, hop_length))

    onset_density = _sliding_density(_norm(mix_rms))
    percussion_density = _sliding_density(_norm(drum_rms))
    vocal_density = _sliding_density(_norm(vocal_rms))
    harmonic_density = _sliding_density(_norm(other_rms))

    # Bass re-entry signal: first derivative of bass presence (SIE 14).
    def _pad_diff(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.size < 2:
            return np.zeros_like(x)
        diff = np.diff(x)
        return np.concatenate([[0.0], diff])

    return StemFeatures(
        frame_times=_resample(frame_times, n),
        hop_length=hop_length,
        sample_rate=sr,
        vocal_presence=_norm(vocal_rms),
        vocal_energy=_norm(vocal_rms),
        vocal_density=vocal_density,
        vocal_onset=_norm(vocal_onset),
        drum_energy=_norm(drum_rms),
        kick_activity=kick,
        snare_activity=snare,
        hat_activity=hat,
        percussion_density=percussion_density,
        onset_density=onset_density,
        bass_energy=_norm(bass_rms),
        bass_presence=_norm(bass_rms),
        bass_onset=_norm(bass_onset),
        melodic_energy=_norm(other_rms),
        harmonic_density=harmonic_density,
        backend=stems.backend,
        status=stems.status,
    )
