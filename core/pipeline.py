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
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

from .analyzer import AnalysisFeatures, AnalyzerEngine, AudioData, detect_key, get_engine
from .analyzer.base import AudioDecodeError
from .beatgrid import BeatGrid, detect_beatgrid
from .cues import generate_cueplan
from .edits import apply_edits_to_cues, load_edits
from .learn import apply_learned_patterns, adaptive_weights, get_preference_store, apply_preference
from .models import Analysis, Bar, Beat, CuePlan, EnergyPoint, GenreProfile, Phrase, Section
from .profiles import genre as _genre_module
from .profiles.genre import get_profile, list_profiles
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


def _normalize_profile_key(name: str) -> str:
    """Normalize a requested profile name exactly like genre.get_profile."""
    return (
        name.strip().lower().replace("-", "_").replace(" ", "_").replace("&", "and")
    )


def _resolve_genre_profile(profile_name: str) -> tuple[GenreProfile, bool]:
    """Resolve a genre profile, reporting a silent-fallback honestly.

    Returns ``(profile, fell_back)`` where ``fell_back`` is True when the
    requested name is not a known profile (or alias) and ``get_profile``
    silently substituted ``open_format``. A loud ``UserWarning`` is raised
    in that case: falling back without telling the user would let a track
    be analyzed under the wrong genre prior while the UI still shows the
    requested genre name.

    NOTE (T-3): there is no genre *detector* in this repo — the profile is
    a user-selected prior, not a detection. Callers must therefore not
    present ``profile.name`` as a detected genre (see ``analyze_track``).
    """
    key = _normalize_profile_key(profile_name)
    aliases = getattr(_genre_module, "_ALIASES", {})
    known = key in _genre_module.PROFILES or key in aliases
    profile = get_profile(profile_name)
    fell_back = not known
    if fell_back:
        warnings.warn(
            f"Unknown genre profile {profile_name!r}: falling back to "
            f"'open_format'. Known profiles: {', '.join(list_profiles())}. "
            "The track was NOT analyzed as the requested genre.",
            UserWarning,
            stacklevel=3,
        )
    return profile, fell_back


def _clamp_cues_to_duration(cues: list, duration: float) -> list:
    """Drop cues outside ``[0, duration]`` (never silently move them).

    Short tracks are the main trigger: a cue snapped to a downbeat past the
    end of the audio is unusable, so it is removed rather than delivered.
    A cue exactly at ``duration`` is kept (it is not outside the track).
    """
    return [c for c in cues if 0.0 <= float(c.time) <= float(duration)]


def _clip_sections_to_duration(sections: list[Section], duration: float) -> list[Section]:
    """Clip section ends to ``duration`` and drop sections past the end."""
    clipped: list[Section] = []
    for sec in sections:
        if float(sec.start) >= float(duration):
            continue
        end = min(float(sec.end), float(duration))
        if end <= float(sec.start):
            continue
        clipped.append(
            Section(
                type=sec.type,
                start=float(sec.start),
                end=end,
                confidence=sec.confidence,
                source=sec.source,
            )
        )
    return clipped


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
    try:
        audio = engine.load(path)
    except AudioDecodeError:
        raise
    except Exception as exc:  # noqa: BLE001 - engine errors get decode context
        raise AudioDecodeError(
            path,
            f"decoder failed ({type(exc).__name__}: {exc})",
            "The file may be corrupt, truncated, or in an unsupported "
            "format. Try re-exporting it as WAV (PCM 16-bit) or MP3.",
        ) from exc
    if audio is None or getattr(audio, "samples", None) is None:
        raise AudioDecodeError(
            path,
            "decoder returned no audio data",
            "The file may be corrupt or empty. Try re-exporting it as "
            "WAV (PCM 16-bit) and try again.",
        )
    try:
        sample_count = int(audio.samples.size)
    except Exception:  # noqa: BLE001 - defensive: unusual engine return value
        sample_count = 0
    if sample_count == 0 or float(audio.duration) <= 0:
        raise AudioDecodeError(
            path,
            "decoded audio is empty or has zero duration",
            "The file may be corrupt, truncated, or contain no audio. "
            "Try re-exporting it as WAV (PCM 16-bit) and try again.",
        )
    emit("extract", "Extracting spectral features & waveform", 20)
    features = engine.extract(audio, progress=progress)
    profile, profile_fell_back = _resolve_genre_profile(profile_name)
    if profile_fell_back:
        emit(
            "profile",
            f"WARNING: unknown genre profile {profile_name!r} - "
            "falling back to open_format",
            35,
        )
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

    # T-7: report the stem backend that ACTUALLY ran, never the requested
    # mode. Previously "engine" was set to `mode`, so an ONNX request that
    # silently fell back to the DSP separator was still reported as
    # engine="onnx". `backend`/`status` below come from the StemSet that
    # produced the features; `fallback`/`backendMismatch` make the
    # substitution explicit instead of silent.
    requested_stem_backend = "onnx" if mode == "onnx" else "dsp-fallback"
    stem_info: dict = {}
    if stem_feats is not None:
        actual_stem_backend = str(stem_feats.backend)
        stem_info = {
            "engine": actual_stem_backend,
            "requestedEngine": mode,
            "requestedBackend": requested_stem_backend,
            "backend": actual_stem_backend,
            "status": stem_feats.status,
            "fallback": bool(
                stem_feats.status == "fallback"
                or actual_stem_backend != requested_stem_backend
            ),
            "backendMismatch": bool(actual_stem_backend != requested_stem_backend),
            "fallbackFrom": (
                requested_stem_backend
                if actual_stem_backend != requested_stem_backend
                else None
            ),
            "fallbackReason": getattr(stem_feats, "fallback_reason", None),
            "enabledStems": list(enabled_stems or []),
            "stems": _stem_curves(stem_feats),
        }
    elif mode in ("intellistem", "onnx"):
        # Stem separation produced nothing: sections already fell back to
        # the DSP structure detector, so say so instead of leaving the
        # stem fields absent (which readers could mistake for "no stems
        # requested").
        stem_info = {
            "engine": "dsp-fallback",
            "requestedEngine": mode,
            "requestedBackend": requested_stem_backend,
            "backend": None,
            "status": "failed",
            "fallback": True,
            "backendMismatch": True,
            "fallbackFrom": requested_stem_backend,
            "fallbackReason": (
                "stem separation failed; sections fell back to the "
                "DSP structure detector"
            ),
            "enabledStems": list(enabled_stems or []),
            "stems": {},
        }

    # Audio QC report and the
    # Gig Readiness Score, computed from the same analysis pass.
    qc_report = compute_qc_report(audio, features)
    extras = dict(stem_info)
    # T-3 (honest genre provenance): CuePilot has NO genre detector, so
    # nothing here is a detection. The profile is the user-selected prior
    # that biased structure/cue generation. This block is the serialized
    # record of that fact, alongside the canonical CuePlan field
    # `genre_profile_used` (exported as `genreProfileUsed`).
    extras["genre"] = {
        "profileUsed": profile.name,
        "profileRequested": profile_name,
        "profileFallback": bool(profile_fell_back),
        "detected": None,
        "confidence": None,
        "detector": None,
        "note": (
            "No genre detector exists in this repo; 'profileUsed' is the "
            "user-selected genre profile applied as a prior, not a "
            "detected genre. CuePlan.genre_detected / genre_confidence "
            "are deprecated and intentionally left empty / 0."
        ),
    }
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
        # T-7: in "onnx" mode the engine label must be the backend that
        # actually ran. "onnx" is only written when the ONNX separator
        # really produced the stems; a DSP fallback (or total stem
        # failure, whose sections came from the DSP structure detector)
        # is reported as "dsp-fallback", never "onnx".
        engine=(
            (str(stem_feats.backend) if stem_feats is not None else "dsp-fallback")
            if mode == "onnx"
            else (mode if mode in ("intellistem", "onnx") else engine.name)
        ),
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
    # detection strategy was used and which genre profile was APPLIED.
    #
    # T-3: the old code wrote `genre_detected = profile.name` and
    # `genre_confidence = grid.confidence`, i.e. it presented the
    # user-selected profile as a detection and the BEATGRID confidence
    # as a genre confidence. There is no genre detector in this repo, so
    # both claims were false. Now:
    #   * `genre_detected` is intentionally left EMPTY and
    #     `genre_confidence` is 0.0 — they are deprecated aliases kept
    #     only because CuePlan (core/models.py) still defines them and
    #     other code may read them; an empty/0 value must be read as
    #     "no genre detection was performed", never as a detection.
    #   * the honest record is `genre_profile_used` (a real CuePlan
    #     field since consolidation, exported as `genreProfileUsed`)
    #     plus the serialized `extras["genre"]` block on the Analysis.
    cue_plan.strategy_used = strategy_for_profile(profile)
    cue_plan.genre_detected = ""
    cue_plan.genre_confidence = 0.0
    cue_plan.genre_profile_used = profile.name

    # Consolidation: cues derived from DSP-fallback stem sections carry
    # the same 0.15 penalty as the sections themselves (T-7). The
    # classic structure path is unaffected (stem_info fallback False).
    if stem_info.get("fallback") and stem_info.get("backend") == "dsp-fallback":
        for _cue in cue_plan.cues:
            _cue.confidence = max(0.0, min(1.0, float(_cue.confidence) - 0.15))
            _cue.reason.append(
                "confidence reduced by 0.15: section derived from "
                "DSP-fallback stem backend, not ML separation"
            )

    # Personalization: nudge auto cues toward the DJ's learned
    # per-genre patterns before re-applying per-track manual edits.
    # Consolidation wiring: pass the beatgrid so learned shifts are
    # re-snapped to the nearest beat (previously unwired, so the snap
    # was a no-op in production).
    _beat_times = [float(b.time) for b in grid.beats] if grid.beats else []
    cue_plan.cues = apply_learned_patterns(
        cue_plan.cues,
        analysis.duration,
        profile.name,
        beat_times=_beat_times,
        downbeat_times=list(grid.downbeat_times or []),
    )
    emit("learn", "Applying learned patterns from your edits", 85)

    if apply_edits:
        edits = load_edits(path)
        if edits is not None and getattr(edits, "corrupt", False):
            # Corrupt sidecar: never apply it, and never swallow it.
            emit(
                "edits",
                f"WARNING: edit sidecar for this track is corrupt "
                f"({getattr(edits, 'load_error', '')}); edits were NOT applied",
                85,
            )
        elif edits is not None:
            # Consolidation wiring: validate with duration + report
            # rejections/clamps instead of applying blindly.
            _edit_notifications: list = []
            cue_plan.cues = apply_edits_to_cues(
                cue_plan.cues,
                edits,
                duration=audio.duration,
                notifications=_edit_notifications,
            )
            for _note in _edit_notifications:
                emit("edits", f"Edit note: {_note}", 85)
    # S-6: never deliver a cue outside the track (short tracks are the
    # main case: a cue snapped past the end of the audio is unusable) and
    # never report a section running past the end of the audio. Clamping
    # happens after edits, so a manual edit cannot re-introduce an
    # out-of-range cue, and before readiness, so the score reflects the
    # cues actually delivered.
    cue_plan.cues = _clamp_cues_to_duration(cue_plan.cues, audio.duration)
    sections = _clip_sections_to_duration(sections, audio.duration)
    cue_plan.sections = sections
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
            # Consolidation (R-2): cache identity includes the analysis
            # grid, so a change in sample rate / hop invalidates old
            # entries instead of silently reusing stale arrays.
            cached = get_cached_arrays(
                "stem_features",
                file_hash,
                params={
                    "sample_rate": int(features.sample_rate),
                    "hop_length": int(features.hop_length),
                },
            )
    except Exception:  # noqa: BLE001 - cache must never break analysis
        cached = None

    fallback_reason: str | None = None
    if cached is not None:
        arrays, meta = cached
        emit("stems", "Loading cached stem features", base_pct + 5)
        cached_backend = str(meta.get("backend") or "")
        if cached_backend in ("", "cache"):
            # Old caches stored the literal placeholder "cache" as the
            # backend; reporting that as a backend would be a lie, and
            # guessing the original backend would be another one.
            cached_backend = "unknown-cached"
        stem_feats = _SF.from_arrays(
            arrays,
            hop_length=int(meta.get("hop_length", 512)),
            sample_rate=int(meta.get("sample_rate", 22050)),
            backend=cached_backend,
            status=str(meta.get("status", "ok")),
        )
    else:
        emit("stems", "Separating stems (vocals / drums / bass / other)", base_pct + 5)
        try:
            from .stems.backends import DspSeparator

            separator = get_separator(preferred_backend)
            try:
                stems = separator.separate(audio)
            except Exception as exc:  # noqa: BLE001 - ONNX model present but fails
                # T-7: the ONNX failure used to be swallowed silently and
                # the DSP result was then reported under the ONNX name.
                # Record it so analyze_track can report the substitution.
                fallback_reason = (
                    f"{getattr(separator, 'name', preferred_backend)} "
                    f"separator failed ({type(exc).__name__}: {exc}); "
                    "fell back to dsp-fallback"
                )
                emit("stems", f"Stem backend fell back: {fallback_reason}", base_pct + 5)
                stems = DspSeparator().separate(audio)
            else:
                # get_separator() itself falls back to DSP when the
                # preferred backend is unavailable (e.g. no ONNX model):
                # that substitution must also be visible, not silent.
                if (
                    preferred_backend
                    and getattr(separator, "name", None) != preferred_backend
                ):
                    fallback_reason = (
                        f"requested stem backend {preferred_backend!r} "
                        f"unavailable; used {separator.name!r} instead"
                    )
        except Exception:  # noqa: BLE001
            # NOTE: no DSP separation actually ran on this path — the
            # previous message ("using DSP fallback") was wrong. Sections
            # fall back to the DSP *structure detector* below instead.
            emit(
                "stems",
                "Stem separation unavailable - using DSP structure detector (no stems)",
                base_pct + 5,
            )
            stems = None

        if stems is None:
            from .structure import detect_structure

            return detect_structure(features, grid, profile), None

        emit("stem_features", "Extracting stem sensor features", base_pct + 10)
        stem_feats = extract_stem_features(stems, features.frame_times, features.hop_length)
        if fallback_reason is not None:
            # StemFeatures (core/stems/features.py) has no fallback-reason
            # field, so it is carried as an attribute for analyze_track's
            # extras block; the backend/status fields themselves already
            # name the backend that actually produced the features.
            stem_feats.fallback_reason = fallback_reason  # type: ignore[attr-defined]
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
                    params={
                        "sample_rate": int(stem_feats.sample_rate),
                        "hop_length": int(stem_feats.hop_length),
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
