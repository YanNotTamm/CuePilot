"""Global energy curve & stem-aware section detection (SIE sections 16-29).

Energy is a fused curve (SIE section 17), NOT raw RMS: RMS + spectral +
drums + bass + onset + stem activity, with configurable weights. Drop / build
/ breakdown / intro / outro candidates are scored per SIE section 24-29 from
the sensor curves, phrase boundaries, and downbeats.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..beatgrid import BeatGrid
from ..models import Section, SectionType
from . import ENERGY_WEIGHTS
from .features import StemFeatures

# Drop evidence weights (SIE section 24).
DROP_WEIGHTS = {
    "energy_spike": 0.25,
    "kick_reentry": 0.20,
    "bass_reentry": 0.20,
    "drum_density_change": 0.10,
    "phrase_boundary": 0.15,
    "previous_build_or_break": 0.10,
}


@dataclass
class StemAnalysis:
    """Fused energy curve + section candidates + stem sensor summary."""

    energy: np.ndarray
    sections: list[Section]
    backend: str
    status: str
    evidence: list[dict] = None  # type: ignore[assignment]


def _resample_to(a, n: int) -> np.ndarray:
    """Simple block-average resample to `n` samples (hop-aligned frames)."""
    a = np.asarray(a, dtype=float)
    if a.ndim != 1 or a.size == 0:
        a = np.zeros(n, dtype=float)
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


def compute_fused_energy(
    stem_feats: StemFeatures,
    spectral: np.ndarray | None = None,
    rms: np.ndarray | None = None,
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """Fused global energy curve (SIE section 16-17)."""
    w = {**(weights or ENERGY_WEIGHTS)}
    n = int(stem_feats.frame_times.size) if stem_feats.frame_times.size else 0
    if n == 0:
        return np.array([], dtype=float)

    terms: dict[str, np.ndarray] = {}

    def _norm(x: np.ndarray) -> np.ndarray:
        x = _resample_to(np.asarray(x, dtype=float), n)
        m = float(np.max(x)) if x.size else 0.0
        if m <= 1e-9:
            return np.zeros(n, dtype=float)
        return np.clip(x / m, 0.0, 1.0)

    terms["rms"] = _norm(rms if rms is not None else stem_feats.vocal_energy + stem_feats.drum_energy)
    terms["spectral"] = _norm(spectral)
    terms["drums"] = _norm(stem_feats.drum_energy)
    terms["bass"] = _norm(stem_feats.bass_energy)
    terms["onset"] = _norm(stem_feats.onset_density)
    terms["stem_activity"] = _norm(
        0.5 * stem_feats.vocal_presence
        + 0.25 * stem_feats.percussion_density
        + 0.25 * stem_feats.harmonic_density
    )

    energy = np.zeros(n, dtype=float)
    for key, val in terms.items():
        energy += w.get(key, 0.0) * val
    return _norm(energy)


def detect_stem_sections(
    stem_feats: StemFeatures,
    grid: BeatGrid,
    profile=None,
    fused_energy: np.ndarray | None = None,
) -> StemAnalysis:
    """Detect + classify sections using stem sensors (SIE 22-29)."""
    if stem_feats.frame_times.size == 0 or not grid.beats:
        return StemAnalysis(
            energy=np.array([], dtype=float),
            sections=[],
            backend=stem_feats.backend,
            status="empty",
            evidence=[],
        )

    energy = fused_energy if fused_energy is not None else compute_fused_energy(stem_feats)
    times = np.asarray(stem_feats.frame_times, dtype=float)
    beats = np.asarray(grid.beat_times, dtype=float)
    downbeats = np.asarray(grid.downbeat_times, dtype=float)

    def _at(curve: np.ndarray, t: float) -> float:
        idx = int(np.searchsorted(times, t))
        idx = min(max(idx, 0), max(0, int(np.asarray(curve).size) - 1))
        if np.asarray(curve).size == 0:
            return 0.0
        return float(np.asarray(curve)[idx])

    energy_at_beat = np.array([_at(energy, b) for b in beats], dtype=float)
    drum_at_beat = np.array([_at(stem_feats.drum_energy, b) for b in beats], dtype=float)
    bass_at_beat = np.array([_at(stem_feats.bass_energy, b) for b in beats], dtype=float)
    kick_at_beat = np.array([_at(stem_feats.kick_activity, b) for b in beats], dtype=float)
    vocal_at_beat = np.array([_at(stem_feats.vocal_presence, b) for b in beats], dtype=float)
    melodic_at_beat = np.array([_at(stem_feats.melodic_energy, b) for b in beats], dtype=float)

    energy_smooth = _moving_average(energy_at_beat, 4)
    bars_per_phrase = 4
    n_beats = len(beats)

    # Phrase boundaries = beat indices that are downbeats of phrase starts.
    phrase_idx = set()
    for db_i, db_time in enumerate(downbeats):
        bi = int(np.searchsorted(beats, db_time))
        if bi < n_beats and (db_i % bars_per_phrase) == 0:
            phrase_idx.add(bi)

    # Downbeat set for alignment scoring.
    db_set = {int(np.searchsorted(beats, db_time)) for db_time in downbeats}

    drop_scores: dict[int, dict[str, float]] = {}
    prev_label = SectionType.INTRO

    for i in range(1, n_beats - 1):
        e_prev, e_cur = energy_smooth[i - 1], energy_smooth[i]
        energy_spike = max(0.0, e_cur - e_prev)

        kick_prev = kick_at_beat[i - 1]
        kick_cur = kick_at_beat[i]
        kick_reentry = max(0.0, kick_cur - kick_prev)

        bass_prev = bass_at_beat[i - 1]
        bass_cur = bass_at_beat[i]
        bass_reentry = max(0.0, bass_cur - bass_prev)

        drum_prev = _moving_average(drum_at_beat, 8)[i - 1]
        drum_cur = _moving_average(drum_at_beat, 8)[i]
        drum_density_change = max(0.0, drum_cur - drum_prev)

        is_phrase = i in phrase_idx
        is_downbeat = i in db_set

        score = (
            DROP_WEIGHTS["energy_spike"] * min(1.0, energy_spike * 3.0)
            + DROP_WEIGHTS["kick_reentry"] * min(1.0, kick_reentry * 4.0)
            + DROP_WEIGHTS["bass_reentry"] * min(1.0, bass_reentry * 4.0)
            + DROP_WEIGHTS["drum_density_change"] * min(1.0, drum_density_change * 3.0)
            + DROP_WEIGHTS["phrase_boundary"] * (1.0 if is_phrase else 0.0)
            + DROP_WEIGHTS["previous_build_or_break"] * (0.7 if prev_label in (SectionType.BUILD, SectionType.BREAKDOWN) else 0.0)
        )
        # Alignment bonus: real drops land on downbeats (SIE 38).
        if is_downbeat:
            score = min(1.0, score + 0.05)

        drop_scores[i] = {
            "score": score,
            "energy_spike": energy_spike,
            "kick_reentry": kick_reentry,
            "bass_reentry": bass_reentry,
            "is_phrase": is_phrase,
            "is_downbeat": is_downbeat,
        }
        # Track running label so "previous section" is available.
        if energy_smooth[i] > energy_smooth[i - 1] and e_cur > 0.55:
            prev_label = SectionType.BUILD

    return _build_sections_from_scores(
        stem_feats=stem_feats,
        grid=grid,
        beats=beats,
        times=times,
        energy=energy,
        energy_at_beat=energy_at_beat,
        vocal_at_beat=vocal_at_beat,
        melodic_at_beat=melodic_at_beat,
        bass_at_beat=bass_at_beat,
        drum_at_beat=drum_at_beat,
        drop_scores=drop_scores,
        phrase_idx=phrase_idx,
        profile=profile,
    )


def _build_sections_from_scores(
    stem_feats: StemFeatures,
    grid: BeatGrid,
    beats: np.ndarray,
    times: np.ndarray,
    energy: np.ndarray,
    energy_at_beat: np.ndarray,
    vocal_at_beat: np.ndarray,
    melodic_at_beat: np.ndarray,
    bass_at_beat: np.ndarray,
    drum_at_beat: np.ndarray,
    drop_scores: dict[int, dict[str, float]],
    phrase_idx: set[int],
    profile=None,
) -> StemAnalysis:
    """Convert per-beat drop scores + sensor curves into labeled Sections."""
    n = len(beats)
    sections: list[Section] = []
    evidence: list[dict] = []
    duration = float(beats[-1]) if n else 0.0

    if n == 0:
        return StemAnalysis(energy=energy, sections=[], backend=stem_feats.backend, status="empty", evidence=evidence)

    min_gap_beats = 32  # SIE section 38: 8 bars between distinct drops.

    # Pick the top drop candidates, sorted by score, spaced >= min_gap_beats.
    ranked = sorted(drop_scores.items(), key=lambda kv: kv[1]["score"], reverse=True)
    picked: list[int] = []
    for beat_i, info in ranked:
        if len(picked) >= 2:
            break
        if all(abs(beat_i - p) >= min_gap_beats for p in picked):
            if info["score"] >= 0.45:
                picked.append(beat_i)

    # Sort picked by chronological order so picked[0] is always the
    # earliest in time → DROP, picked[1] → SECOND_DROP.
    picked.sort()

    def _label_for(i: int) -> SectionType:
        e = float(energy_at_beat[i])
        v = float(vocal_at_beat[i])
        m = float(melodic_at_beat[i])
        b = float(bass_at_beat[i])
        d = float(drum_at_beat[i])
        pos = beats[i] / max(duration, 1e-6)

        if i in picked:
            return SectionType.SECOND_DROP if picked.index(i) > 0 else SectionType.DROP
        if e < 0.25 and v < 0.25 and pos < 0.10:
            return SectionType.INTRO
        if e < 0.25 and v < 0.25 and pos > 0.85:
            return SectionType.OUTRO
        if e < 0.35 and b < 0.3 and d < 0.3:
            return SectionType.BREAKDOWN
        if v >= 0.35 and m >= 0.25:
            return SectionType.VOCAL
        if e >= 0.6 and d >= 0.5:
            return SectionType.DROP
        return SectionType.CHORUS if v >= 0.2 else SectionType.BUILD

    # Build coarse segments (phrase granularity) and classify each.
    boundary_indices = sorted(set([0, n - 1] + sorted(phrase_idx) + picked))
    labels: list[SectionType] = []

    for k in range(len(boundary_indices) - 1):
        start_i = boundary_indices[k]
        end_i = boundary_indices[k + 1]
        if end_i - start_i < 4:
            continue
        contained = [b for b in picked if start_i <= b < end_i]
        mid = (start_i + end_i) // 2
        label = _label_for(mid)
        if contained:
            # A segment that contains a picked drop beat is a DROP; the first
            # picked drop in time wins DROP, later ones become SECOND_DROP.
            rank = picked.index(contained[0])
            label = SectionType.DROP if rank == 0 else SectionType.SECOND_DROP
        # Merge tiny end sections into neighbors implicitly below.
        labels.append(label)
        start_t = float(beats[start_i])
        end_t = float(beats[end_i]) if end_i < n else start_t + 4.0
        if end_t <= start_t:
            end_t = start_t + 1.0
        sec = Section(
            type=label,
            start=start_t,
            end=end_t,
            confidence=float(np.clip(0.55 + 0.05 * drop_scores.get(mid, {}).get("score", 0.5), 0.4, 0.98)),
            source="stem",
        )
        sections.append(sec)
        if label in (SectionType.DROP, SectionType.SECOND_DROP):
            info = drop_scores.get(mid, {})
            evidence.append(
                {
                    "time": round(float(beats[mid]), 3),
                    "dropScore": round(float(info.get("score", 0.0)), 3),
                    "energySpike": round(float(info.get("energy_spike", 0.0)), 3),
                    "kickReentry": round(float(info.get("kick_reentry", 0.0)), 3),
                    "bassReentry": round(float(info.get("bass_reentry", 0.0)), 3),
                    "phraseBoundary": bool(info.get("is_phrase", False)),
                    "downbeatAligned": bool(info.get("is_downbeat", False)),
                }
            )

    # Finalize: merge consecutive same-type sections, guarantee first/last.
    if sections:
        first = sections[0]
        if first.type not in (SectionType.INTRO,) and first.start < duration * 0.08:
            first.type = SectionType.INTRO
        last = sections[-1]
        if last.type not in (SectionType.OUTRO,) and last.end > duration * 0.9 and last.type not in (SectionType.DROP, SectionType.SECOND_DROP):
            last.type = SectionType.OUTRO

    sections = _apply_genre_prior(sections, profile)

    merged: list[Section] = []
    for sec in sections:
        if merged and merged[-1].type == sec.type and abs(sec.start - merged[-1].end) < 1e-3:
            merged[-1].end = max(merged[-1].end, sec.end)
            merged[-1].confidence = max(merged[-1].confidence, sec.confidence)
        else:
            merged.append(sec)

    return StemAnalysis(
        energy=energy,
        sections=merged,
        backend=stem_feats.backend,
        status="ok",
        evidence=evidence,
    )


def _apply_genre_prior(
    sections: list[Section],
    profile=None,
) -> list[Section]:
    """Re-label sections outside the profile's expected order (SIE section 31).

    Mirrors the classic structure detector's genre prior: a genre profile
    describes the sections a track *should* have (e.g. baile: INTRO -> VOCAL
    -> DROP -> VOCAL -> SECOND_DROP -> OUTRO). Types not used by the profile
    are mapped to the profile's expected type at the same proportional
    position. DROP / SECOND_DROP are always preserved.
    """
    if not profile or not profile.section_order:
        return sections
    allowed = set(profile.section_order)
    if all(s.type in allowed for s in sections):
        return sections
    n = len(sections)
    order = profile.section_order
    out = list(sections)
    for i, sec in enumerate(out):
        if sec.type in allowed or sec.type in (SectionType.DROP, SectionType.SECOND_DROP):
            continue
        pos = i / max(1, n - 1)
        target = order[min(len(order) - 1, int(round(pos * (len(order) - 1))))]
        out[i] = Section(
            type=target,
            start=sec.start,
            end=sec.end,
            confidence=sec.confidence * 0.9,
            source=sec.source,
        )
    return out


def _moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    if arr.size == 0:
        return arr
    window = max(1, min(window, arr.size))
    kernel = np.ones(window) / window
    return np.convolve(np.asarray(arr, dtype=float), kernel, mode="same")
