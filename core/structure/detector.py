"""Musical structure detection.

Segments the track at phrase boundaries where energy and spectral change are
significant, then classifies each segment into INTRO / VOCAL / BUILD / DROP /
BREAKDOWN / CHORUS / SECOND_DROP / OUTRO using structural heuristics plus the genre
profile's expected section order as a prior.
"""

from __future__ import annotations

import numpy as np

from ..analyzer import AnalysisFeatures
from ..bands import reverse_bass_score, subbass_novelty
from ..beatgrid import BeatGrid
from ..models import GenreProfile, Section, SectionType
from .router import STRATEGY_ENERGY, STRATEGY_GRID, STRATEGY_LOCAL, STRATEGY_VOCAL, confidence_cap, strategy_for_profile


class StructureDetector:
    def __init__(self) -> None:
        self._beat_times: np.ndarray = np.array([], dtype=float)
        self._energy: np.ndarray = np.array([], dtype=float)
        self._flux: np.ndarray = np.array([], dtype=float)
        self._harmonic_ratio: np.ndarray = np.array([], dtype=float)
        self._onset: np.ndarray = np.array([], dtype=float)
        self._min_section_beats = 8
        self._strategy = STRATEGY_GRID
        self._grid: BeatGrid | None = None
        self._features: AnalysisFeatures | None = None

    def detect(
        self,
        features: AnalysisFeatures,
        grid: BeatGrid,
        profile: GenreProfile | None = None,
        strategy: str | None = None,
    ) -> list[Section]:
        """Detect and classify sections for a track.

        `strategy` overrides the automatic genre routing (logic_new.md §6.3).
        When omitted, the strategy is derived from the genre profile.

        Returns an ordered list of Sections covering the whole track with no
        overlaps and no gaps.
        """
        self._strategy = strategy or strategy_for_profile(profile)
        self._grid = grid
        self._features = features
        self._prepare(features, grid)
        boundaries = self._segment(grid, features, profile)
        sections = self._build_sections(boundaries, grid, features)
        classified = self._classify(sections, profile)
        return self._finalize(classified)

    def _prepare(self, features: AnalysisFeatures, grid: BeatGrid) -> None:
        beat_times = np.asarray(grid.beat_times, dtype=float)
        self._beat_times = beat_times

        rms = features.rms
        flux = features.spectral_flux
        harm = features.harmonic
        perc = features.percussive
        frame_times = features.frame_times

        def sample_at(arr: np.ndarray, times: np.ndarray) -> np.ndarray:
            if arr.size == 0:
                return np.zeros(len(times), dtype=float)
            idx = np.searchsorted(frame_times, times)
            idx = np.clip(idx, 0, arr.size - 1)
            return np.asarray(arr[idx], dtype=float)

        self._energy = sample_at(rms, beat_times)
        self._flux = sample_at(flux, beat_times)
        self._onset = sample_at(features.onset_env, beat_times)
        h = sample_at(harm, beat_times)
        p = sample_at(perc, beat_times)
        denom = h + p
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(denom > 1e-9, h / np.maximum(denom, 1e-9), 0.5)
        self._harmonic_ratio = np.clip(ratio, 0.0, 1.0)

    def _segment(
        self,
        grid: BeatGrid,
        features: AnalysisFeatures | None = None,
        profile: GenreProfile | None = None,
    ) -> list[int]:
        """Return beat indices marking section boundaries (inclusive of ends).

        Strategy-aware (logic_new.md §6.4-6.7):
        - Grid: novelty peaks snapped to phrase-length beats (genre's
          preferred phrase bars when available).
        - Energy / Local: rhythm-driven novelty (onset + flux) with a shorter
          phrase distance, no hard grid constraint.
        - Vocal: harmonic-change novelty (chorus/hook contrast).
        """
        n = len(self._beat_times)
        if n == 0:
            return [0]
        if n == 1:
            return [0, 0]

        energy = np.asarray(self._energy, dtype=float)
        smooth = self._moving_average(energy, 4)

        flux = np.asarray(self._flux, dtype=float)
        onset = np.asarray(self._onset, dtype=float)

        if self._strategy == STRATEGY_ENERGY or self._strategy == STRATEGY_LOCAL:
            rhythm = self._normalize(flux) + 0.5 * self._normalize(onset)
            novelty = np.abs(np.diff(self._moving_average(energy + 0.5 * rhythm, 2)))
            # Sub-bass band-limited novelty (20-120 Hz): drops jump in sub-bass
            # energy rather than full-band (logic_new.md §6.5).
            if features is not None:
                sub_novelty = subbass_novelty(features, grid)
                if sub_novelty.size == novelty.size:
                    novelty = novelty + 0.8 * sub_novelty
        elif self._strategy == STRATEGY_VOCAL:
            harm = np.abs(np.diff(np.asarray(self._harmonic_ratio, dtype=float)))
            harm = np.concatenate([[0.0], harm])
            novelty = self._normalize(flux) + 1.5 * self._normalize(harm)
        else:
            novelty = np.abs(np.diff(smooth))

        novelty = np.concatenate([[0.0], np.asarray(novelty, dtype=float)])

        # Peak-pick on a phrase-length scale so we don't over-segment on
        # intra-beat/off-beat variation. Energy/Local strategies allow denser
        # boundaries (irregular phrasing) than Grid.
        if self._strategy in (STRATEGY_ENERGY, STRATEGY_LOCAL):
            phrase_beats = 8
        elif self._strategy == STRATEGY_VOCAL:
            phrase_beats = 12
        else:
            phrase_beats = self._preferred_phrase_beats(profile)

        try:
            from scipy.signal import find_peaks

            peaks, _ = find_peaks(novelty, distance=phrase_beats)
            boundaries = sorted(set([0, int(n) - 1] + [int(p) for p in peaks]))
        except Exception:
            boundaries = [0, int(n) - 1]

        boundaries = self._merge_short_sections(boundaries, min_beats=self._min_section_beats)
        if boundaries[-1] != n - 1:
            boundaries.append(n - 1)
        boundaries = sorted(set(boundaries))
        return boundaries

    def _preferred_phrase_beats(self, profile: GenreProfile | None) -> int:
        if profile and profile.preferred_phrase_bars:
            return max(4, min(32, int(profile.preferred_phrase_bars[0]) * 4))
        return 16

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, dtype=float)
        if a.size == 0:
            return a
        m = float(np.max(a))
        if m <= 1e-9:
            return np.zeros_like(a)
        return a / m

    @staticmethod
    def _moving_average(arr: np.ndarray, window: int) -> np.ndarray:
        if arr.size == 0:
            return arr
        window = max(1, min(window, arr.size))
        kernel = np.ones(window) / window
        return np.convolve(arr, kernel, mode="same")

    def _merge_short_sections(self, boundaries: list[int], min_beats: int) -> list[int]:
        """Remove boundaries that leave a segment shorter than `min_beats`.

        Iteratively drop the weaker boundary of a too-short segment: a short
        middle segment is absorbed into its longer neighbour by dropping the
        boundary on the side with the smaller energy change.
        """
        if len(boundaries) <= 2:
            return boundaries
        result = sorted(set(boundaries))

        while len(result) > 2:
            # Find the shortest segment.
            best_i = 1
            best_w = result[1] - result[0]
            for i in range(1, len(result)):
                w = result[i] - result[i - 1]
                if w < best_w:
                    best_w = w
                    best_i = i
            if best_w >= min_beats:
                break

            # Decide which boundary of the shortest segment to drop: compare
            # the energy change across its left and right boundary.
            left = result[best_i - 1]
            right = result[best_i]
            if best_i + 1 < len(result):
                right_next = result[best_i + 1]
                change_right = abs(
                    float(np.mean(self._energy[left:right]))
                    - float(np.mean(self._energy[right:right_next]))
                )
            else:
                change_right = 0.0
            if best_i - 1 > 0:
                left_prev = result[best_i - 2]
                change_left = abs(
                    float(np.mean(self._energy[left_prev:left]))
                    - float(np.mean(self._energy[left:right]))
                )
            else:
                change_left = 0.0

            if change_left >= change_right and best_i > 1:
                result.pop(best_i - 1)
            else:
                result.pop(best_i)

        return sorted(set(result))

    def _build_sections(
        self,
        boundaries: list[int],
        grid: BeatGrid,
        features: AnalysisFeatures,
    ) -> list[Section]:
        sections: list[Section] = []
        for i in range(len(boundaries) - 1):
            start_i = boundaries[i]
            end_i = boundaries[i + 1]
            if end_i - start_i <= 0:
                continue
            start_time = float(self._beat_times[start_i])
            end_beat = min(end_i, len(self._beat_times) - 1)
            end_time = (
                float(self._beat_times[end_beat])
                if end_beat >= start_i
                else start_time + 1.0
            )
            sections.append(
                Section(
                    type=SectionType.INTRO,  # placeholder, replaced in classify
                    start=start_time,
                    end=end_time,
                    confidence=0.0,
                    source="dsp",
                )
            )
        return sections

    def _section_features(self, sections: list[Section]) -> list[dict]:
        total = float(self._beat_times[-1]) if len(self._beat_times) else 1.0
        features_out: list[dict] = []
        energy_max = float(np.max(self._energy)) if self._energy.size else 1.0
        energy_min = float(np.min(self._energy)) if self._energy.size else 0.0
        energy_span = max(1e-9, energy_max - energy_min)

        for sec in sections:
            mask = (self._beat_times >= sec.start) & (self._beat_times < sec.end)
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                features_out.append(
                    {
                        "mean_energy": 0.0,
                        "rel_energy": 0.0,
                        "slope": 0.0,
                        "mean_flux": 0.0,
                        "mean_harmonic": 0.5,
                        "position": 0.0,
                    }
                )
                continue
            e = self._energy[idx]
            rel = (float(np.mean(e)) - energy_min) / energy_span
            slope = float(np.polyfit(np.arange(len(e)), e, 1)[0]) if len(e) > 1 else 0.0
            features_out.append(
                {
                    "mean_energy": float(np.mean(e)),
                    "rel_energy": float(np.clip(rel, 0.0, 1.0)),
                    "slope": slope,
                    "mean_flux": float(np.mean(self._flux[idx])),
                    "mean_harmonic": float(np.mean(self._harmonic_ratio[idx])),
                    "position": float(sec.start / total) if total > 0 else 0.0,
                }
            )
        return features_out

    def _classify(
        self,
        sections: list[Section],
        profile: GenreProfile | None,
    ) -> list[Section]:
        if not sections:
            return []
        feats = self._section_features(sections)
        n = len(sections)

        # Global stats for thresholds.
        rel_energies = np.array([f["rel_energy"] for f in feats])
        energy_high = float(np.percentile(rel_energies, 75)) if n > 1 else 0.5
        energy_low = float(np.percentile(rel_energies, 25)) if n > 1 else 0.3
        max_energy = float(np.max(rel_energies)) if n > 1 else 1.0

        labels: list[SectionType] = []
        for i, f in enumerate(feats):
            labels.append(
                self._classify_single(
                    i,
                    f,
                    n,
                    energy_high,
                    energy_low,
                    max_energy,
                    profile,
                )
            )

        # Post-process: guarantee INTRO on first and OUTRO on last low-energy ends.
        if n >= 1 and feats[0]["rel_energy"] < max(0.4, energy_high * 0.6):
            labels[0] = SectionType.INTRO
        if n >= 2 and feats[-1]["rel_energy"] < max(0.4, energy_high * 0.6):
            labels[-1] = SectionType.OUTRO

        # Collapse a second DROP into SECOND_DROP when a DROP already appeared.
        first_drop_idx = next((k for k, t in enumerate(labels) if t == SectionType.DROP), None)
        if first_drop_idx is not None:
            for k in range(first_drop_idx + 1, n):
                if labels[k] == SectionType.DROP:
                    labels[k] = SectionType.SECOND_DROP

        # Genre prior: re-label sections whose type is outside the profile's
        # expected sequence, mapping to the order type at the same position.
        labels = self._apply_genre_prior(labels, profile)

        # Confidence based on energy distinctiveness of the section.
        cap = confidence_cap(self._strategy)
        # Reverse-bass boost (Hardstyle, Genre_pattern.md §42): when the track
        # uses a reverse-bass kick the drop/climax confidence rises; pass it
        # through so DROP/climax sections can be trusted more.
        rev_bass = 0.0
        if (
            profile is not None
            and profile.energy_profile == "climax"
            and self._features is not None
            and self._grid is not None
        ):
            rev_bass = reverse_bass_score(self._features, self._grid)
        out: list[Section] = []
        for sec, lab, f in zip(sections, labels, feats):
            conf = self._confidence_for(f, lab, energy_high, energy_low, max_energy)
            if lab in (SectionType.DROP, SectionType.SECOND_DROP) and rev_bass > 0.0:
                conf = min(0.99, conf + 0.1 * rev_bass)
            out.append(
                Section(
                    type=lab,
                    start=sec.start,
                    end=sec.end,
                    confidence=min(float(conf), cap),
                    source="dsp",
                )
            )
        return out

    def _classify_single(
        self,
        index: int,
        f: dict,
        n: int,
        energy_high: float,
        energy_low: float,
        max_energy: float,
        profile: GenreProfile | None,
    ) -> SectionType:
        rel = f["rel_energy"]
        slope = f["slope"]
        flux = f["mean_flux"]
        harmonic = f["mean_harmonic"]
        pos = f["position"]

        # Strong priors for track ends.
        if pos < 0.03 and rel < energy_high:
            return SectionType.INTRO
        if pos > 0.90 and rel < energy_high:
            return SectionType.OUTRO

        # Vocal-based strategy: chorus/hook = high harmonic + high energy,
        # preferred over the EDM drop reading (logic_new.md §6.6).
        if self._strategy == STRATEGY_VOCAL:
            if rel >= max(0.45, energy_high * 0.7) and harmonic >= 0.5:
                return SectionType.CHORUS
            if harmonic >= 0.5 and 0.2 < rel < 0.8:
                return SectionType.VOCAL

        # High energy + percussive = DROP (harmonic ratio low = percussive).
        if rel >= max(0.65, energy_high) and harmonic < 0.55:
            return SectionType.DROP

        # High energy + harmonic/vocal-rich = CHORUS.
        if rel >= max(0.5, energy_high * 0.8) and harmonic >= 0.55:
            return SectionType.CHORUS

        # Low energy + low harmonic content + strong flux delta = BREAKDOWN.
        if rel <= energy_low and flux > np.median(self._flux) if self._flux.size else True:
            return SectionType.BREAKDOWN

        # Rising energy with mid energy = BUILD. The upper cap keeps a near-
        # flat, near-drop-energy section from being labeled a build; real
        # builds ramp clearly below drop level.
        if slope > 0 and 0.25 < rel < min(0.55, energy_high * 0.85):
            return SectionType.BUILD

        # Harmonic/mid energy sections = VOCAL.
        if harmonic >= 0.55 and 0.2 < rel < 0.7:
            return SectionType.VOCAL

        # Fallbacks by position.
        if pos < 0.08:
            return SectionType.INTRO
        if pos > 0.75:
            return SectionType.OUTRO
        return SectionType.DROP

    def _apply_genre_prior(
        self,
        labels: list[SectionType],
        profile: GenreProfile | None,
    ) -> list[SectionType]:
        """Re-label sections whose type is outside the profile's expected order.

        A genre profile describes the sections a track *should* have (e.g.
        baile: INTRO -> VOCAL -> DROP -> VOCAL -> SECOND_DROP -> OUTRO, with no
        BUILD/BREAKDOWN/CHORUS). When the DSP classifier emits a type that the
        profile does not use, we map it to the profile's expected type at the
        same proportional position, using the enclosing expected types as
        anchors. Types already in the profile's order are left untouched.
        """
        if not profile or not profile.section_order:
            return labels
        order = profile.section_order
        allowed = set(order)
        if set(labels).issubset(allowed):
            return labels
        n = len(labels)
        out = list(labels)
        for i, lab in enumerate(labels):
            if lab in allowed:
                continue
            pos = i / max(1, n - 1)
            target = order[min(len(order) - 1, int(round(pos * (len(order) - 1))))]
            out[i] = target
        return out

    def _confidence_for(        self,
        f: dict,
        label: SectionType,
        energy_high: float,
        energy_low: float,
        max_energy: float,
    ) -> float:
        rel = f["rel_energy"]
        if label in (SectionType.DROP, SectionType.SECOND_DROP):
            base = 0.75 + 0.25 * (rel / max(0.7, max_energy))
        elif label in (SectionType.CHORUS, SectionType.VOCAL):
            base = 0.6 + 0.4 * f["mean_harmonic"]
        elif label == SectionType.BREAKDOWN:
            base = 0.6 + 0.4 * (1.0 - min(1.0, rel / max(0.5, energy_high)))
        elif label in (SectionType.BUILD,):
            base = 0.6 + 0.4 * min(1.0, max(0.0, f["slope"]) * 4.0)
        else:  # INTRO / OUTRO
            base = 0.7 + 0.3 * (1.0 - rel)
        return float(np.clip(base, 0.4, 0.99))

    @staticmethod
    def _finalize(sections: list[Section]) -> list[Section]:
        """Merge consecutive sections of the same type; drop degenerate ones."""
        if not sections:
            return sections
        result: list[Section] = []
        for sec in sections:
            if sec.end <= sec.start:
                continue
            if result and result[-1].type == sec.type and abs(sec.start - result[-1].end) < 1e-3:
                result[-1].end = max(result[-1].end, sec.end)
                result[-1].confidence = max(result[-1].confidence, sec.confidence)
                continue
            result.append(sec)
        return result


def detect_structure(
    features: AnalysisFeatures,
    grid: BeatGrid,
    profile: GenreProfile | None = None,
    strategy: str | None = None,
) -> list[Section]:
    """Convenience wrapper: detect sections from features + beatgrid.

    `strategy` overrides automatic genre routing (logic_new.md §6.3).
    """
    return StructureDetector().detect(features, grid, profile, strategy=strategy)
