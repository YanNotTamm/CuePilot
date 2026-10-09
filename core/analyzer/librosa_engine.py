"""Librosa-based analyzer engine for audio feature extraction."""

from __future__ import annotations

import warnings

import librosa
import numpy as np

from .base import AnalysisFeatures, AnalyzerEngine, AudioData

_SUPPORTED_EXT = {".mp3", ".wav", ".aiff", ".aif", ".flac", ".ogg"}


def band_limited_rms(
    y: np.ndarray,
    sr: float,
    hop_length: int,
    lo_hz: float,
    hi_hz: float,
) -> np.ndarray:
    """Per-frame RMS of the band-passed signal (aligned to the hop grid).

    Used for band-limited novelty (e.g. sub-bass 20-120 Hz drop detection,
    logic_new.md §6.5). Returns zeros if band-passing is unavailable.
    """
    from scipy.signal import butter, sosfiltfilt

    nyq = max(1e-6, sr / 2.0)
    freqs = [max(1e-6, lo_hz / nyq), min(0.9999, hi_hz / nyq)]
    if freqs[0] >= freqs[1]:
        return np.zeros(_frame_count(y, hop_length), dtype=float)
    try:
        sos = butter(4, freqs, btype="bandpass", output="sos")
        filtered = sosfiltfilt(sos, np.asarray(y, dtype=np.float64).reshape(-1))
        return librosa.feature.rms(y=filtered, hop_length=hop_length)[0]
    except Exception:
        return np.zeros(_frame_count(y, hop_length), dtype=float)


def _frame_count(y: np.ndarray, hop_length: int) -> int:
    return int(1 + (y.size - 1) // hop_length) if y.size else 0


class LibrosaEngine(AnalyzerEngine):
    """Default engine backed by librosa + soundfile (libsndfile).

    `load` uses soundfile/librosa with `mono=False` so channel count and
    true sample rate are preserved; downstream analysis uses the mono mix.
    """

    name = "librosa"
    version = "1.0.0"

    def load(self, path: str) -> AudioData:
        y, sr = librosa.load(path, sr=None, mono=False)
        if y.ndim == 1:
            channels = 1
            samples = y[np.newaxis, :]
        else:
            channels = y.shape[0]
            samples = y
        duration = samples.shape[1] / float(sr)
        if sr > 22050:
            samples = librosa.resample(
                samples,
                orig_sr=sr,
                target_sr=22050,
                axis=-1,
                res_type="kaiser_fast" if (22050 / float(sr)) <= 0.5 else "polyphase",
            )
            sr = 22050.0
        return AudioData(samples=samples, sample_rate=int(sr), channels=channels, duration=duration)

    def supports(self, path: str) -> bool:
        ext = path.lower().rsplit(".", 1)[-1]
        return f".{ext}" in _SUPPORTED_EXT

    def extract(self, audio: AudioData, progress=None) -> AnalysisFeatures:
        y = audio.mono
        sr = float(audio.sample_rate)
        hop_length = 512

        def emit(stage: str, msg: str, pct: float) -> None:
            if progress:
                progress(stage, msg, pct)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            emit("extract", "Computing onset strength", 25)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
            frame_times = librosa.frames_to_time(
                np.arange(len(onset_env)), sr=sr, hop_length=hop_length
            )
            emit("extract", "Computing RMS energy", 35)
            rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
            emit("extract", "Computing spectral centroid", 45)
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
            emit("extract", "Computing spectral flux", 55)
            flux = librosa.onset.onset_strength(
                S=np.abs(librosa.stft(y, hop_length=hop_length)),
                sr=sr,
                hop_length=hop_length,
            )
            emit("extract", "Computing chroma (key analysis)", 65)
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
            emit("extract", "Separating harmonic & percussive", 75)
            harmonic = librosa.effects.harmonic(y)
            percussive = librosa.effects.percussive(y)
            harm_rms = librosa.feature.rms(y=harmonic, hop_length=hop_length)[0]
            perc_rms = librosa.feature.rms(y=percussive, hop_length=hop_length)[0]
            emit("extract", "Computing MFCC", 85)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
            emit("extract", "Computing sub-bass energy", 88)
            subbass_energy = band_limited_rms(y, sr, hop_length, 20.0, 120.0)

        bpm = 0.0
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tempo, _ = librosa.beat.beat_track(
                    onset_envelope=onset_env, sr=sr, hop_length=hop_length, units="frames", trim=False
                )
            bpm = float(np.atleast_1d(tempo)[0])
        except Exception:
            bpm = 0.0

        flux = np.clip(flux - np.min(flux), 0.0, None)

        return AnalysisFeatures(
            hop_length=hop_length,
            sample_rate=audio.sample_rate,
            frame_times=frame_times,
            onset_env=np.asarray(onset_env, dtype=float),
            rms=np.asarray(rms, dtype=float),
            spectral_centroid=np.asarray(centroid, dtype=float),
            spectral_flux=np.asarray(flux, dtype=float),
            chroma=np.asarray(chroma, dtype=float),
            harmonic=np.asarray(harm_rms, dtype=float),
            percussive=np.asarray(perc_rms, dtype=float),
            mfcc=np.asarray(mfcc, dtype=float),
            subbass_energy=np.asarray(subbass_energy, dtype=float),
            bpm_estimate=bpm,
        )
