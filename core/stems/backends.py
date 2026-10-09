"""Stem separation backends (SIE sections 6-8, 10).

A `StemSeparator` can be swapped in/out: Demucs, MDX-family, UVR-compatible,
ONNX models, or the built-in DSP approximation. `get_separator()` picks the
first available backend and falls back to `DspSeparator` (SIE section 50) so
analysis never aborts when no ML model is installed.

The DSP backend deliberately does NOT rely on a naive
`frequency < X = bass` rule alone (SIE section 2): it combines harmonic /
percussive separation with spectral filtering and envelopes, then marks the
result as `fallback` so downstream confidence can be adjusted.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import StemSeparator, StemSet

_DSP_FALLBACK = "dsp-fallback"


def _hpss(y: np.ndarray, margin: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Harmonic / percussive separation via median filtering (librosa)."""
    try:
        import librosa

        harmonic, percussive = librosa.effects.hpss(y, margin=margin)
        return harmonic, percussive
    except Exception:
        return y, np.zeros_like(y)


class DspSeparator(StemSeparator):
    """Lightweight DSP stem approximation (no external model required).

    Produces time-aligned mono stems aligned to the input mix. Used as the
    always-available fallback (SIE section 50) when no ML backend is present.
    """

    name = _DSP_FALLBACK

    def __init__(self, sr: int = 22050) -> None:
        self._sr = sr

    def available(self) -> bool:
        return True

    def separate(self, audio, progress: Optional[Callable] = None) -> StemSet:
        def emit(pct: float):
            if progress:
                progress(pct)

        y = audio.mono
        if y.ndim > 1:
            y = np.mean(y, axis=0)
        y = np.asarray(y, dtype=np.float64)

        emit(0.1)
        harmonic, percussive = _hpss(y, margin=1.2)
        emit(0.5)

        # Bass: energy concentrated in the low band of the harmonic content
        # (kick & sub in the mix band). Combined low-pass + envelope.
        from scipy.signal import butter, sosfiltfilt

        sos = butter(4, 140 / (self._sr / 2), btype="lowpass", output="sos")
        bass = sosfiltfilt(sos, y)
        bass = bass * (1.0 + 0.4 * np.abs(percussive))
        emit(0.7)

        # Vocals: mid-band harmonic content (rough, best-effort without an ML
        # model; the feature layer treats this as a weak signal).
        sos_mid = butter(4, [140 / (self._sr / 2), 5000 / (self._sr / 2)], btype="bandpass", output="sos")
        vocals = sosfiltfilt(sos_mid, harmonic)
        emit(0.85)

        other = np.clip(y - bass - vocals, -1.0, 1.0)
        instrumental = np.clip(y - vocals, -1.0, 1.0)

        def _shape(x: np.ndarray) -> np.ndarray:
            return np.asarray(x, dtype=np.float32).reshape(-1)

        return StemSet(
            vocals=_shape(vocals),
            drums=_shape(percussive),
            bass=_shape(bass),
            other=_shape(other),
            instrumental=_shape(instrumental),
            sample_rate=self._sr,
            backend=self.name,
            status="fallback",
        )


class OnnxStemSeparator(StemSeparator):
    """ONNX stem separator (UVR/MDX-NET family via onnxruntime, no torch).

    Model discovery: looks for onnx models under `models/stems/` in the repo,
    or a path given via env `CUEPILOT_STEM_MODEL`. The MDX-NET architecture
    operates on a 4-channel STFT representation ``[L_real, L_imag, R_real,
    R_imag]``; we reproduce the reference inference (chunked, overlap-add)
    with numpy + librosa only, so no PyTorch install is required.

    Output mapping (SIE section 5): the model yields ``instrumental`` (and
    ``vocals = mix - instrumental``). Drums / bass / other are derived from
    the instrumental via the DSP band/HPSS helpers so all four minimum stems
    are always present (SIE section 50: "percussion not isolated by model").
    """

    name = "onnx"

    # MDX-NET architecture constants (n_fft=6144, hop=1024, dim_t=8 => 256 frames).
    _MODEL_SR = 44100
    _N_FFT = 6144
    _HOP = 1024
    _DIM_F = 2560
    _DIM_T = 256
    _N_BINS = _N_FFT // 2 + 1

    def __init__(self, model_path: str | None = None) -> None:
        self._model_path = model_path

    def available(self) -> bool:
        try:
            import onnxruntime  # noqa: F401

            from pathlib import Path

            if not self._model_path:
                import os

                p = os.environ.get("CUEPILOT_STEM_MODEL", "")
                self._model_path = p or str(Path(__file__).resolve().parents[2] / "models" / "stems")
            path = Path(self._model_path)
            return path.exists() and (
                path.is_file() or any(path.glob("*.onnx"))
            )
        except Exception:
            return False

    # --- reference STFT helpers (mirrors torch.stft/istft center=True) ----
    @classmethod
    def _stft_4ch(cls, chunk: np.ndarray, window: np.ndarray) -> np.ndarray:
        """(2, n) mono-pair -> (4, n_bins, n_frames) real tensor [Lr,Li,Rr,Ri]."""
        import librosa

        frames = []
        for c in range(2):
            s = librosa.stft(
                chunk[c],
                n_fft=cls._N_FFT,
                hop_length=cls._HOP,
                window=window,
                center=True,
                pad_mode="reflect",
            )
            frames.append(s)  # (n_bins, n_frames) complex
        l, r = frames
        return np.stack(
            [l.real, l.imag, r.real, r.imag], axis=0
        )[:, : cls._DIM_F]  # (4, dim_f, n_frames)

    @classmethod
    def _istft_4ch(cls, spec: np.ndarray, window: np.ndarray, length: int) -> np.ndarray:
        """(4, dim_f, n_frames) -> (2, length) mono-pair."""
        import librosa

        n_frames = spec.shape[-1]
        pad = np.zeros((4, cls._N_BINS - cls._DIM_F, n_frames), dtype=spec.dtype)
        spec = np.concatenate([spec, pad], axis=1)  # (4, n_bins, n_frames)
        out = []
        for c in range(2):
            real = spec[2 * c]
            imag = spec[2 * c + 1]
            s = real + 1j * imag
            y = librosa.istft(
                s,
                hop_length=cls._HOP,
                window=window,
                center=True,
                length=length,
            )
            out.append(y)
        return np.stack(out, axis=0)

    @staticmethod
    def _hann(n_fft: int) -> np.ndarray:
        try:
            from scipy.signal.windows import hann

            return hann(n_fft, sym=False).astype(np.float32)
        except Exception:
            import numpy as _np

            return _np.hanning(n_fft + 1)[:-1].astype(np.float32)

    @classmethod
    def _demix(cls, session, mix: np.ndarray, progress: Optional[Callable]) -> np.ndarray:
        """Chunked MDX-NET inference (reference algorithm, numpy port).

        `mix` is (2, n) float32 at _MODEL_SR. Returns the instrumental (2, n).
        """
        n = mix.shape[1]
        trim = cls._N_FFT // 2
        gen_size = cls._HOP * (cls._DIM_T - 1) - 2 * trim
        pad = (-n) % gen_size
        mix_p = np.zeros((2, trim + n + pad + trim), dtype=np.float32)
        mix_p[:, trim : trim + n] = mix

        total = 0
        n_chunks = max(1, (n + pad + gen_size - 1) // gen_size)
        window = cls._hann(cls._N_FFT)
        outputs = []

        i = 0
        while i < n + pad:
            seg = mix_p[:, i : i + cls._HOP * (cls._DIM_T - 1)]
            if seg.shape[1] < cls._HOP * (cls._DIM_T - 1):
                seg = np.pad(
                    seg,
                    ((0, 0), (0, cls._HOP * (cls._DIM_T - 1) - seg.shape[1])),
                    mode="constant",
                )
            spec = cls._stft_4ch(seg, window)  # (4, dim_f, 256)
            pred = session.run(None, {"input": spec[np.newaxis].astype(np.float32)})[0]
            out = cls._istft_4ch(pred[0], window, seg.shape[1])
            outputs.append(out)
            i += gen_size
            total += 1
            if progress:
                progress(min(0.95, 0.05 + 0.9 * total / n_chunks))

        joined = np.concatenate(outputs, axis=1)[:, :n]
        return joined

    def separate(self, audio, progress: Optional[Callable] = None) -> StemSet:
        def emit(pct: float):
            if progress:
                progress(pct)

        import numpy as np

        try:
            import librosa
            import onnxruntime as ort
        except Exception:
            raise RuntimeError("onnxruntime / librosa not available")

        path = Path(self._model_path) if self._model_path else None
        if path is None:
            p = os.environ.get("CUEPILOT_STEM_MODEL", "")
            path = Path(p) if p else Path(__file__).resolve().parents[2] / "models" / "stems"
        if path.is_dir():
            onnx_files = list(path.glob("*.onnx"))
            if not onnx_files:
                raise RuntimeError("no .onnx model under models/stems/")
            path = onnx_files[0]

        emit(0.02)
        from . import get_acceleration
        acc = get_acceleration()
        providers = []
        if acc.provider == "cuda":
            providers.append("CUDAExecutionProvider")
        elif acc.provider == "dml":
            providers.append("DmlExecutionProvider")
        providers.append("CPUExecutionProvider")
        session = ort.InferenceSession(str(path), providers=providers)

        # --- load -> 44100 stereo (mono duplicated) ---
        src_sr = int(audio.sample_rate)
        y = audio.samples if getattr(audio, "samples", None) is not None else audio.mono
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y[np.newaxis, :]
        if y.shape[0] == 1:
            y = np.repeat(y, 2, axis=0)
        if src_sr != self._MODEL_SR:
            y = librosa.resample(
                y, orig_sr=src_sr, target_sr=self._MODEL_SR, axis=-1, res_type="kaiser_best"
            )
        mix = y.astype(np.float32)  # (2, n) @44100

        emit(0.06)
        instrumental = self._demix(session, mix, progress)  # (2, n)
        emit(0.96)
        vocals = mix - instrumental

        # --- collapse to mono at the analyzer sample rate ---
        sr_out = src_sr

        def _mono(x: np.ndarray) -> np.ndarray:
            m = np.mean(x, axis=0)
            if m.shape[0] != mix.shape[1]:
                m = librosa.resample(
                    m, orig_sr=self._MODEL_SR, target_sr=sr_out, axis=-1, res_type="kaiser_best"
                )
            return np.asarray(m, dtype=np.float32).reshape(-1)

        inst_mono = _mono(instrumental)
        vocal_mono = _mono(vocals)
        mix_mono = _mono(mix)

        # --- derive drums/bass/other from the instrumental (SIE 50) ---
        from scipy.signal import butter, sosfiltfilt

        sos = butter(4, 140 / (sr_out / 2), btype="lowpass", output="sos")
        bass = sosfiltfilt(sos, inst_mono)

        try:
            harmonic, percussive = librosa.effects.hpss(inst_mono, margin=1.2)
        except Exception:
            percussive = inst_mono - vocal_mono * 0.0
            harmonic = inst_mono - percussive
        drums = np.asarray(percussive, dtype=np.float32)
        other = np.clip(inst_mono - bass - drums, -1.0, 1.0).astype(np.float32)

        return StemSet(
            vocals=vocal_mono,
            drums=drums,
            bass=np.asarray(bass, dtype=np.float32),
            other=other,
            instrumental=inst_mono,
            sample_rate=sr_out,
            backend=self.name,
            status="ok",
        )


_SEPARATORS: list[StemSeparator] = [OnnxStemSeparator(), DspSeparator()]


def get_separator(preferred: str | None = None) -> StemSeparator:
    """Return the first available separator (SIE section 6) or preferred if available."""
    if preferred:
        for sep in _SEPARATORS:
            if sep.name == preferred and sep.available():
                return sep
    for sep in _SEPARATORS:
        if sep.available():
            return sep
    return DspSeparator()


def available_backends() -> list[str]:
    return [sep.name for sep in _SEPARATORS if sep.available()]
