"""Fix-verification tests for audio decoding (S-6).

Covers: corrupt files, empty files, missing files, unsupported formats,
mono vs stereo, sample rates 44100 / 48000, decode errors surfaced by the
pipeline with file + reason + suggestion, and very short tracks (no cue
outside the track duration).

The load-validation tests do NOT need a real decoder (they fail before
librosa is called). The mono/stereo/sample-rate tests fake ``librosa.load``
/ ``librosa.resample`` so they run with or without the real librosa; with
the real librosa installed they still exercise the engine's own channel /
sample-rate / duration bookkeeping, not the decoder itself.
"""

import sys
import types

import numpy as np
import pytest

try:
    import librosa  # noqa: F401
except ImportError:  # pragma: no cover - sandbox without librosa
    librosa = types.ModuleType("librosa")
    sys.modules.setdefault("librosa", librosa)
try:
    import soundfile  # noqa: F401
except ImportError:  # pragma: no cover - sandbox without soundfile
    sys.modules.setdefault("soundfile", types.ModuleType("soundfile"))

from core.analyzer.base import AudioDecodeError
from core.analyzer.librosa_engine import LibrosaEngine
from core.pipeline import analyze_track


def _write(path, data: bytes):
    path.write_bytes(data)
    return str(path)


def test_load_missing_file_raises_decode_error_with_path(tmp_path):
    missing = str(tmp_path / "gone.mp3")
    with pytest.raises(AudioDecodeError) as excinfo:
        LibrosaEngine().load(missing)
    assert "gone.mp3" in str(excinfo.value)
    assert excinfo.value.path == missing
    assert excinfo.value.suggestion


def test_load_empty_file_raises_decode_error(tmp_path):
    empty = _write(tmp_path / "empty.wav", b"")
    with pytest.raises(AudioDecodeError) as excinfo:
        LibrosaEngine().load(empty)
    assert "empty" in str(excinfo.value).lower()
    assert "0 bytes" in str(excinfo.value)


def test_load_unsupported_extension_raises_decode_error(tmp_path):
    txt = _write(tmp_path / "track.txt", b"hello, not audio")
    with pytest.raises(AudioDecodeError) as excinfo:
        LibrosaEngine().load(txt)
    assert "unsupported" in str(excinfo.value).lower()
    assert ".wav" in str(excinfo.value).lower()


def test_load_corrupt_file_wraps_decoder_error_with_context(tmp_path, monkeypatch):
    corrupt = _write(tmp_path / "corrupt.wav", b"RIFF....not real audio data" * 10)

    def _boom(path, sr=None, mono=False):
        raise RuntimeError("simulated libsndfile failure")

    monkeypatch.setattr(librosa, "load", _boom, raising=False)
    with pytest.raises(AudioDecodeError) as excinfo:
        LibrosaEngine().load(corrupt)
    message = str(excinfo.value)
    assert "corrupt.wav" in message
    assert "simulated libsndfile failure" in message
    assert "re-exporting" in message


def test_load_decoder_returning_no_samples_raises_decode_error(tmp_path, monkeypatch):
    wav = _write(tmp_path / "nosamples.wav", b"x" * 100)
    monkeypatch.setattr(librosa, "load", lambda path, sr=None, mono=False: (np.zeros(0), 22050), raising=False)
    with pytest.raises(AudioDecodeError):
        LibrosaEngine().load(wav)


def _fake_librosa_for(y, sr, monkeypatch):
    """Point the engine's librosa at canned decode output."""
    # NOTE: the fake must not let librosa.load's own `sr=None` argument
    # shadow the canned sample rate — bind them as defaults instead.
    monkeypatch.setattr(
        librosa, "load",
        lambda path, sr=None, mono=False, _y=y, _sr=sr: (_y, _sr),
        raising=False,
    )
    # Resample fake: the engine only checks the returned array's shape via
    # duration (computed pre-resample) and stores sr=22050 afterwards, so
    # returning the input unchanged is enough to test the bookkeeping.
    monkeypatch.setattr(librosa, "resample", lambda samples, **kwargs: samples, raising=False)


def test_load_mono_22050_preserves_channels_and_duration(tmp_path, monkeypatch):
    wav = _write(tmp_path / "mono.wav", b"x" * 100)
    y = np.zeros(22050, dtype=np.float32)  # 1.0 s mono
    _fake_librosa_for(y, 22050, monkeypatch)
    audio = LibrosaEngine().load(wav)
    assert audio.channels == 1
    assert audio.sample_rate == 22050
    assert audio.duration == pytest.approx(1.0)
    assert audio.samples.shape == (1, 22050)


def test_load_stereo_44100_reports_two_channels_and_resampled_rate(tmp_path, monkeypatch):
    wav = _write(tmp_path / "stereo.wav", b"x" * 100)
    y = np.zeros((2, 44100), dtype=np.float32)  # 1.0 s stereo @44100
    _fake_librosa_for(y, 44100, monkeypatch)
    audio = LibrosaEngine().load(wav)
    assert audio.channels == 2
    assert audio.sample_rate == 22050
    # Duration is computed from the ORIGINAL sample rate, before resampling.
    assert audio.duration == pytest.approx(1.0)


def test_load_stereo_48000_reports_two_channels_and_resampled_rate(tmp_path, monkeypatch):
    wav = _write(tmp_path / "stereo48.wav", b"x" * 100)
    y = np.zeros((2, 48000), dtype=np.float32)  # 1.0 s stereo @48000
    _fake_librosa_for(y, 48000, monkeypatch)
    audio = LibrosaEngine().load(wav)
    assert audio.channels == 2
    assert audio.sample_rate == 22050
    assert audio.duration == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Pipeline-level decode handling + very short tracks.
# ---------------------------------------------------------------------------


class _BoomEngine:
    name = "boom"
    version = "0.0-test"

    def load(self, path):
        raise RuntimeError("simulated decoder crash")

    def extract(self, audio, progress=None):
        raise AssertionError("extract must not run after a failed load")


def test_pipeline_wraps_engine_decode_failure_with_filename(tmp_path):
    with pytest.raises(AudioDecodeError) as excinfo:
        analyze_track(
            str(tmp_path / "broken.mp3"),
            engine=_BoomEngine(),
            apply_edits=False,
        )
    assert "broken.mp3" in str(excinfo.value)
    assert excinfo.value.path.endswith("broken.mp3")
    assert excinfo.value.suggestion


def test_pipeline_short_track_5_to_10_seconds_has_no_cue_outside_duration(
    monkeypatch, tmp_path
):
    """A 7-second track: delivered cues must all lie within [0, 7]."""
    import core.learn
    import core.pipeline as pipeline
    from core.models import Bar, Beat, Cue, CuePlan, CueType, Phrase, Section, SectionType
    from core.analyzer.base import AnalysisFeatures, AudioData

    class FakeEngine:
        name = "fake"
        version = "0.0-test"

        def load(self, path):
            return AudioData(
                samples=np.zeros((1, 7 * 22050), dtype=np.float32),
                sample_rate=22050,
                channels=1,
                duration=7.0,
            )

        def extract(self, audio, progress=None):
            n = 32
            return AnalysisFeatures(
                hop_length=512,
                sample_rate=22050,
                frame_times=np.linspace(0.0, 7.0, n),
                onset_env=np.zeros(n),
                rms=np.full(n, 0.5),
                spectral_centroid=np.zeros(n),
                spectral_flux=np.zeros(n),
                chroma=np.zeros((12, n)),
                harmonic=np.zeros(n),
                percussive=np.zeros(n),
                mfcc=np.zeros((13, n)),
                subbass_energy=np.zeros(n),
                bpm_estimate=128.0,
            )

    class FakeGrid:
        bpm = 128.0
        confidence = 0.9
        beats = [Beat(time=float(t), is_downbeat=(i % 4 == 0), index=i) for i, t in enumerate(range(7))]
        bars = [Bar(index=0, downbeat_time=0.0, beat_count=4)]
        phrases = [Phrase(index=0, start_time=0.0, end_time=7.0, bar_count=4)]
        downbeat_times = [0.0, 4.0]
        subdivision = "four"
        subdivision_confidence = 0.9
        tempo_verification = {}

    def cue(slot, time):
        return Cue(slot=slot, type=CueType.DROP, time=time, label="DROP",
                   confidence=0.9, color="#E74C3C")

    # One cue inside, one exactly at the end (kept), two outside (dropped).
    cues = [cue(1, 0.0), cue(5, 6.5), cue(7, 7.0), cue(8, 9.5)]
    sections = [Section(type=SectionType.INTRO, start=0.0, end=9.0, confidence=0.9)]

    class FakeQc:
        def to_dict(self):
            return {}

    class FakeReadiness:
        def to_dict(self):
            return {}

    class FakeStore:
        def count_corrections(self, path):
            return 0

    monkeypatch.setattr(pipeline, "detect_beatgrid", lambda feats, **kw: FakeGrid())
    monkeypatch.setattr(pipeline, "detect_structure", lambda feats, g, prof: sections)
    monkeypatch.setattr(pipeline, "generate_cueplan", lambda *a, **kw: CuePlan(cues=list(cues), sections=list(sections)))
    monkeypatch.setattr(pipeline, "compute_qc_report", lambda audio, feats: FakeQc())
    monkeypatch.setattr(pipeline, "detect_loops", lambda *a, **kw: [])
    monkeypatch.setattr(pipeline, "compute_readiness", lambda **kw: FakeReadiness())
    monkeypatch.setattr(pipeline, "adaptive_weights", lambda name, base_weights=None: dict(base_weights or {}))
    monkeypatch.setattr(pipeline, "apply_learned_patterns", lambda cues_, duration, genre, **kw: cues_)
    monkeypatch.setattr(pipeline, "load_edits", lambda path: None)
    monkeypatch.setattr(core.learn, "get_store", lambda: FakeStore())

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = analyze_track(
            str(tmp_path / "short.wav"),
            profile_name="edm",
            engine=FakeEngine(),
            apply_edits=False,
        )
    assert all(0.0 <= c.time <= 7.0 for c in result.cue_plan.cues)
    assert [c.slot for c in result.cue_plan.cues] == [1, 5, 7]
    assert all(s.end <= 7.0 for s in result.sections)
    assert result.analysis.duration == pytest.approx(7.0)
