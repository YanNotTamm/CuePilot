"""Analysis pipeline.

Orchestrates: scan/load -> analyze (DSP) -> beatgrid -> structure -> cues
-> canonical CuePlan.

Two engines are supported:

* ``mode="basic"`` — the classic DSP pipeline (librosa features + structure
  heuristics).
* ``mode="intellistem"`` — the Stem Intelligence Engine: stems act
  as sensors (vocal/drums/bass/other), a fused energy curve drives
  section/drop detection, and cues are scored from stem evidence. Degrades
  to the DSP fallback separator when no ML model is installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Optional

from .analyzer import AnalysisFeatures, AnalyzerEngine, AudioData, detect_key, get_engine
from .beatgrid import BeatGrid, detect_beatgrid
from .cues import generate_cueplan
from .edits import apply_edits_to_cues, load_edits
from .learn import apply_learned_patterns, adaptive_weights, get_preference_store, apply_preference
from .models import Analysis, Bar, Beat, CuePlan, EnergyPoint, GenreProfile, Phrase, Section
from .profiles.genre import get_profile
from .quality import compute_qc_report, compute_readiness
from .loops import detect_loops
from .stems.backends import get_separator
from .stems.features import StemFeatures, extract_stem_features
from .stems.sections import compute_fused_energy, detect_stem_sections
from .structure import detect_structure
from .structure.router import strategy_for_profile

ProgressCallback = Callable[[str, str, float], None]
"""Signature: (stage_key, message, pct_done)."""

ENGINE_MODES = ("basic", "intellistem", "onnx")


@dataclass
class PipelineResult:
    track_id: str
    audio: AudioData = None  # type: ignore[assignment]
    features: AnalysisFeatures = None  # type: ignore[assignment]
    analysis: Analysis = None  # type: ignore[assignment]
    grid: BeatGrid = None  # type: ignore[assignment]
    sections: list[Section] = field(default_factory=list)
    cue_plan: CuePlan = None  # type: ignore[assignment]
    profile: GenreProfile = None  # type: ignore[assignment]

    def to_dict(self) -> dict:
        return {
            "trackId": self.track_id,
            "analysis": self.analysis.to_dict() if self.analysis else None,
            "sections": [s.to_dict() for s in self.sections],
            "cuePlan": self.cue_plan.to_dict() if self.cue_plan else None,
            "profile": self.profile.name if self.profile else None,
        }


def analyze_track(
    path: str,
    track_id: str = "",
    profile_name: str = "open_format",
    engine: AnalyzerEngine | None = None,
    apply_edits: bool = True,
    progress: ProgressCallback | None = None,
    mode: str = "basic",
    preference: str = "",
    enabled_stems: list[str] | None = None,
) -> PipelineResult:
    """Run the full analysis pipeline for one audio file.

    Args:
        path: path to an audio file (MP3/WAV/AIFF/FLAC/OGG).
        track_id: optional stable id; defaults to the file stem.
        profile_name: genre profile to bias structure/cue generation.
        engine: analyzer engine; defaults to the registered default (librosa).
        apply_edits: when True, re-apply persisted human edits (locked cues keep
            their position, deleted cues stay removed) after generating cues.
        progress: optional callback (stage, message, pct) invoked at each stage.
        mode: "basic" (classic DSP pipeline) or "intellistem" (SIE stem-aware).
        preference: optional DJ preference profile name to blend into scoring
            (adaptive scoring / preference profiles).

    Returns:
        A PipelineResult with analysis, beatgrid, sections, and cue plan.
    """
    if mode not in ENGINE_MODES:
        mode = "basic"
    if engine is None:
        engine = get_engine()

    if enabled_stems is None:
        from .settings import get_enabled_stems

        enabled_stems = get_enabled_stems()

    def emit(stage: str, message: str, pct: float) -> None:
        if progress:
            progress(stage, message, pct)

    emit("load", "Reading & decoding audio file", 5)
    audio = engine.load(path)
    emit("extract", "Extracting spectral features & waveform", 20)
    features = engine.extract(audio, progress=progress)
    profile = get_profile(profile_name)
    emit("profile", f"Loading genre profile: {profile.display_name}", 35)

    key = detect_key(features.chroma)
    emit("key", "Detecting musical key", 45)
    grid = detect_beatgrid(features, subdivision_hint=profile.name)
    emit("beatgrid", "Analyzing beatgrid (BPM, downbeats)", 55)

    if mode in ("intellistem", "onnx"):
        sections, stem_feats = _analyze_stems(
            path=path,
            audio=audio,
            grid=grid,
            features=features,
            profile=profile,
            progress=progress,
            base_pct=55,
            preferred_backend="onnx" if mode == "onnx" else "dsp-fallback",
            enabled_stems=enabled_stems,
        )
    else:
        stem_feats = None
        sections = detect_structure(features, grid, profile)
    emit("structure", "Detecting song structure sections", 65)

    bpm = grid.bpm if grid.bpm > 0 else features.bpm_estimate

    beat_models = [
        Beat(time=b.time, is_downbeat=b.is_downbeat, index=b.index) for b in grid.beats
    ]
    bar_models = [
        Bar(index=bar.index, downbeat_time=bar.downbeat_time, beat_count=bar.beat_count)
        for bar in grid.bars
    ]
    phrase_models = [
        Phrase(
            index=p.index,
            start_time=p.start_time,
            end_time=p.end_time,
            bar_count=p.bar_count,
        )
        for p in grid.phrases
    ]

    if mode in ("intellistem", "onnx") and stem_feats is not None:
        energy_values = _normalized_stem_energy(stem_feats)
    else:
        energy_values = features.rms

    energy_curve = [
        EnergyPoint(time=float(t), energy=float(e))
        for t, e in zip(features.frame_times, energy_values)
    ]

    stem_info: dict = {}
    if stem_feats is not None:
        stem_info = {
            "engine": mode,
            "backend": stem_feats.backend,
            "status": stem_feats.status,
            "enabledStems": list(enabled_stems or []),
            "stems": _stem_curves(stem_feats),
        }

    # Audio QC report and the
    # Gig Readiness Score, computed from the same analysis pass.
    qc_report = compute_qc_report(audio, features)
    extras = dict(stem_info)
    extras["quality"] = qc_report.to_dict()

    # Seamless Saved Loop candidates from the beatgrid.
    extras["loops"] = [
        l.to_dict()
        for l in detect_loops(grid, energy_curve, features=features, duration=audio.duration)
    ]

    analysis = Analysis(
        track_id=track_id or _stem(path),
        duration=audio.duration,
        sample_rate=audio.sample_rate,
        channels=audio.channels,
        bpm=float(bpm),
        key=key,
        beats=beat_models,
        downbeats=grid.downbeat_times,
        bars=bar_models,
        phrases=phrase_models,
        energy_curve=energy_curve,
        spectral_features=_spectral_summary(features),
        engine=mode if mode in ("intellistem", "onnx") else engine.name,
        engine_version=engine.version,
        confidence=grid.confidence,
        extras=extras,
    )

    emit("cues", "Generating & scoring hot cues", 75)
    # Personalization: blend the genre's default weights with the DJ's
    # learned sub-score pressure, then layer any named preference profile on top.
    cue_weights = adaptive_weights(profile.name, base_weights=dict(profile.cue_weights))
    if preference:
        pref_profile = get_preference_store().get(preference)
        if pref_profile is not None:
            cue_weights = apply_preference(pref_profile, profile.name, cue_weights)
    cue_plan = generate_cueplan(analysis, features, grid, sections, profile, cue_weights=cue_weights)

    # logic_new.md §7: provenance block for the canonical CuePlan — which
    # detection strategy was used and which genre profile it was detected under.
    cue_plan.strategy_used = strategy_for_profile(profile)
    cue_plan.genre_detected = profile.name
    cue_plan.genre_confidence = round(grid.confidence, 3)

    # Personalization: nudge auto cues toward the DJ's learned
    # per-genre patterns before re-applying per-track manual edits.
    cue_plan.cues = apply_learned_patterns(cue_plan.cues, analysis.duration, profile.name)
    emit("learn", "Applying learned patterns from your edits", 85)

    if apply_edits:
        edits = load_edits(path)
        if edits is not None:
            cue_plan.cues = apply_edits_to_cues(cue_plan.cues, edits)
    emit("done", "Hot cues ready", 100)

    # Gig Readiness Score from cue confidence, QC,
    # structure completeness and manual-correction history.
    try:
        from .learn import get_store

        corrections = get_store().count_corrections(os.path.abspath(path))
    except Exception:
        corrections = 0
    readiness = compute_readiness(
        cues=cue_plan.cues,
        sections=sections,
        qc=qc_report,
        correction_count=corrections,
    )
    extras["readiness"] = readiness.to_dict()

    # logic_new.md §6.3: which detection strategy the DSP structure detector
    # routed to for this genre, plus the beatgrid subdivision and kick-snare
    # tempo verification (Genre_pattern.md §717, §702-723).
    extras["structure"] = {
        "strategy": strategy_for_profile(profile),
        "subdivision": grid.subdivision,
        "subdivisionConfidence": grid.subdivision_confidence,
        "tempoVerification": grid.tempo_verification,
    }

    return PipelineResult(
        track_id=analysis.track_id,
        audio=audio,
        features=features,
        analysis=analysis,
        grid=grid,
        sections=sections,
        cue_plan=cue_plan,
        profile=profile,
    )


def _spectral_summary(features: AnalysisFeatures) -> dict:
    import numpy as np

    def mean(x):
        return float(np.mean(x)) if x.size else 0.0

    return {
        "mean_onset": round(mean(features.onset_env), 4),
        "mean_rms": round(mean(features.rms), 4),
        "mean_centroid": round(mean(features.spectral_centroid), 2),
        "mean_flux": round(mean(features.spectral_flux), 4),
        "mean_harmonic": round(mean(features.harmonic), 4),
        "mean_percussive": round(mean(features.percussive), 4),
    }


def _normalized_stem_energy(stem_feats: StemFeatures) -> list[float]:
    """Normalized fused energy aligned to analyzer frame times."""
    import numpy as np

    n = int(stem_feats.frame_times.size)
    fused = compute_fused_energy(stem_feats)
    if fused.size == 0:
        return []
    fused = fused[:n]
    m = float(np.max(fused)) if fused.size else 0.0
    if m <= 1e-9:
        return [0.0] * int(fused.size)
    return [float(v / m) for v in fused]


def _analyze_stems(
    path: str,
    audio,
    grid: BeatGrid,
    features: AnalysisFeatures,
    profile: GenreProfile | None,
    progress: ProgressCallback | None,
    base_pct: int,
    preferred_backend: str | None = None,
    enabled_stems: list[str] | None = None,
) -> tuple[list[Section], StemFeatures | None]:
    """Stem Intelligence path (SIE.md): separate -> feature -> energy -> sections.

    Never aborts analysis: if stem separation fails, sections fall back to the
    classic structure detector (SIE section 50). Stem features are cached by
    audio file hash (SIE section 46) so re-analysis skips separation.
    """
    import numpy as np

    def emit(stage: str, message: str, pct: float) -> None:
        if progress:
            progress(stage, message, pct)

    # SIE section 46: stem separation is expensive; reuse cached envelopes when
    # the file has not changed.
    from .cache import get_cached_arrays, store_cached_arrays, file_sha256
    from .stems.features import StemFeatures as _SF

    file_hash = ""
    cached: tuple[dict[str, np.ndarray], dict] | None = None
    try:
        if os.path.isfile(path):
            file_hash = file_sha256(path)
            cached = get_cached_arrays("stem_features", file_hash)
    except Exception:  # noqa: BLE001 - cache must never break analysis
        cached = None

    if cached is not None:
        arrays, meta = cached
        emit("stems", "Loading cached stem features", base_pct + 5)
        stem_feats = _SF.from_arrays(
            arrays,
            hop_length=int(meta.get("hop_length", 512)),
            sample_rate=int(meta.get("sample_rate", 22050)),
            backend=str(meta.get("backend", "cache")),
            status=str(meta.get("status", "ok")),
        )
    else:
        emit("stems", "Separating stems (vocals / drums / bass / other)", base_pct + 5)
        try:
            from .stems.backends import DspSeparator

            separator = get_separator(preferred_backend)
            try:
                stems = separator.separate(audio)
            except Exception:  # noqa: BLE001 - e.g. ONNX model present but fails to run
                stems = DspSeparator().separate(audio)
        except Exception:  # noqa: BLE001
            emit("stems", "Stem separation failed - using DSP fallback", base_pct + 5)
            stems = None

        if stems is None:
            from .structure import detect_structure

            return detect_structure(features, grid, profile), None

        emit("stem_features", "Extracting stem sensor features", base_pct + 10)
        stem_feats = extract_stem_features(stems, features.frame_times, features.hop_length)
        if file_hash:
            try:
                store_cached_arrays(
                    "stem_features",
                    file_hash,
                    stem_feats.to_arrays(),
                    meta={
                        "hop_length": stem_feats.hop_length,
                        "sample_rate": stem_feats.sample_rate,
                        "backend": stem_feats.backend,
                        "status": stem_feats.status,
                    },
                )
            except Exception:  # noqa: BLE001 - cache write must never abort
                pass
    stem_feats = _apply_stem_selection(stem_feats, enabled_stems)
    emit("stem_energy", "Computing fused energy curve", base_pct + 13)
    fused = compute_fused_energy(stem_feats, spectral=features.spectral_flux)
    emit("stem_sections", "Detecting sections from stem sensors", base_pct + 16)
    stem_analysis = detect_stem_sections(stem_feats, grid, profile, fused_energy=fused)

    sections = stem_analysis.sections
    if not sections:
        from .structure import detect_structure

        sections = detect_structure(features, grid, profile)
    return sections, stem_feats


def _stem(path: str) -> str:
    import os

    return os.path.splitext(os.path.basename(path))[0]


# Sensor fields each stem drives (SIE sections 12-15). Used to mute a stem's
# influence when the user disables it for analysis.
_STEM_SENSOR_FIELDS: dict[str, tuple[str, ...]] = {
    "vocal": ("vocal_presence", "vocal_energy", "vocal_density", "vocal_onset"),
    "drums": (
        "drum_energy",
        "kick_activity",
        "snare_activity",
        "hat_activity",
        "percussion_density",
    ),
    "bass": ("bass_energy", "bass_presence", "bass_onset"),
    "other": ("melodic_energy", "harmonic_density"),
}


def _apply_stem_selection(
    stem_feats: StemFeatures,
    enabled_stems: list[str] | None,
) -> StemFeatures:
    """Zero out sensors for stems the user disabled, so fusion ignores them."""
    import numpy as np

    if not enabled_stems:
        return stem_feats
    disabled = [s for s in _STEM_SENSOR_FIELDS if s not in enabled_stems]
    if not disabled:
        return stem_feats
    for stem in disabled:
        for field in _STEM_SENSOR_FIELDS.get(stem, ()):
            setattr(stem_feats, field, np.zeros_like(getattr(stem_feats, field)))
    return stem_feats


def _stem_curves(stem_feats: StemFeatures, bins: int = 1500) -> dict:
    """Downsampled per-stem energy curves for the UI waveform overlay.

    Returns ``{vocal/drums/bass/other: [{t, v}, ...]}`` aligned to the
    stem frame grid, plus the shared ``times`` array.
    """
    import numpy as np

    from .stems.sections import _resample_to

    times = np.asarray(stem_feats.frame_times, dtype=float)
    if times.size == 0:
        return {"times": [], "vocal": [], "drums": [], "bass": [], "other": []}
    n = min(int(times.size), bins)
    times_n = _resample_to(times, n)

    def _curve(arr) -> list[dict]:
        a = _resample_to(np.asarray(arr, dtype=float), n)
        return [{"t": float(t), "v": float(v)} for t, v in zip(times_n, a)]

    return {
        "times": [float(t) for t in times_n],
        "vocal": _curve(stem_feats.vocal_energy),
        "drums": _curve(stem_feats.drum_energy),
        "bass": _curve(stem_feats.bass_energy),
        "other": _curve(stem_feats.melodic_energy),
    }
