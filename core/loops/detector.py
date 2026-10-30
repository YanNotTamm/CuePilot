"""Loop-Region Intelligence (Saved Loops).

Detects 4/8/16-bar regions that are *phrase-perfect* and *seamless*
(start/end land on the same downbeat and share similar energy/harmonic
character) and offers them as Saved Loop candidates — useful for build-up
extension or vocal-loop mashups live.

Unlike hot cues, a loop is a region, not a point:
    {type: SAVED_LOOP, bars, start, end, seamlessScore, useCase}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..analyzer import AnalysisFeatures
from ..beatgrid import BeatGrid
from ..models import EnergyPoint, Phrase

# Candidate loop lengths (bars).
LOOP_BARS = (4, 8, 16)
# Minimum seamless score to offer a loop candidate.
SEAMLESS_MIN = 0.55
# Default beats per bar for the duration formula when the caller does not
# supply a meter. The formula itself must use beats_per_bar, NOT
# BARS_PER_PHRASE: the old code multiplied bars * BARS_PER_PHRASE, which
# only gave the right answer in 4/4 because both constants happen to be
# 4 (S-2). Phrase length and bar length are different concepts.


@dataclass
class SavedLoop:
    """A detected Saved Loop candidate."""

    bars: int
    start: float
    end: float
    seamless_score: float
    use_case: str
    phrase_index: int = -1
    reasons: list[str] = field(default_factory=list)
    # S-2: True when `end` is a real downbeat from the grid (bars bars
    # after the loop head); False when no usable grid downbeat existed
    # and `end` fell back to start + nominal duration.
    end_from_grid: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "SAVED_LOOP",
            "bars": self.bars,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "seamlessScore": round(self.seamless_score, 3),
            "useCase": self.use_case,
            "phraseIndex": self.phrase_index,
            "reasons": self.reasons,
            "endFromGrid": self.end_from_grid,
        }


def _energy_at(energy_curve: list[EnergyPoint], time: float) -> float:
    if not energy_curve:
        return 0.5
    times = np.asarray([p.time for p in energy_curve], dtype=float)
    vals = np.asarray([p.energy for p in energy_curve], dtype=float)
    idx = int(np.searchsorted(times, time))
    idx = min(max(idx, 0), vals.size - 1)
    return float(vals[idx])


def _loop_duration(bpm: float, bars: int, beats_per_bar: int = 4) -> float:
    """Duration in seconds of `bars` bars at `bpm`.

    Explicitly bars * beats_per_bar * beat_seconds. The previous formula
    (bars * BARS_PER_PHRASE * beat) was numerically identical in 4/4
    only because BARS_PER_PHRASE also equals 4; it conflated phrase
    length with bar length and would be wrong for any other meter.
    """
    if bpm <= 0 or bars <= 0 or beats_per_bar <= 0:
        return 0.0
    beat = 60.0 / bpm
    return bars * beats_per_bar * beat


def _loop_end_from_grid(
    grid: BeatGrid,
    start: float,
    bars: int,
    fallback_end: float,
) -> tuple[float, bool]:
    """Loop end from the grid's next downbeats, or the documented fallback.

    Finds the downbeat nearest `start` (within half a beat period, so a
    phrase head that is a downbeat matches exactly) and returns the
    downbeat `bars` bars later -- the loop then ends exactly on a
    downbeat even when the BPM estimate is slightly off or the tempo
    drifts. Returns (fallback_end, False) when the grid has no usable
    downbeat for this loop (start not on a downbeat, or too few
    downbeats remain); callers must surface that flag, not hide it.
    """
    downbeats = list(grid.downbeat_times or [])
    if not downbeats or bars <= 0:
        return fallback_end, False
    beat_period = 60.0 / grid.bpm if grid.bpm > 0 else 0.0
    tolerance = 0.5 * beat_period if beat_period > 0 else 0.1
    arr = np.asarray(downbeats, dtype=float)
    idx = int(np.searchsorted(arr, start))
    candidates = [i for i in (idx - 1, idx) if 0 <= i < len(arr)]
    if not candidates:
        return fallback_end, False
    nearest = min(candidates, key=lambda i: abs(float(arr[i]) - start))
    if abs(float(arr[nearest]) - start) > tolerance:
        return fallback_end, False
    target = nearest + bars
    if target >= len(arr):
        return fallback_end, False
    return float(arr[target]), True


def _seamless_score(
    energy_curve: list[EnergyPoint],
    start: float,
    end: float,
    loop_sec: float,
    lookback: float = 2.0,
) -> float:
    """How seamlessly a loop at [start, end) repeats.

    Compares the energy just before `start` with the energy just before `end`
    (both are the "tail" feeding back into the loop head). If they match, the
    loop can repeat without an energy jump. Also rewards start/end both being
    low-energy (a clean seam) or both high-energy (a pumping seam).
    """
    if not energy_curve:
        return 0.6
    start_tail = _energy_at(energy_curve, max(0.0, start - lookback))
    end_tail = _energy_at(energy_curve, max(0.0, end - lookback))
    seam_match = 1.0 - abs(start_tail - end_tail)

    # Head/tail interior energy should also be comparable for the repeat.
    head = _energy_at(energy_curve, start + loop_sec * 0.25)
    tail = _energy_at(energy_curve, end - loop_sec * 0.25)
    interior = 1.0 - abs(head - tail)

    return float(np.clip(0.6 * seam_match + 0.4 * interior, 0.0, 1.0))


def _use_case(start: float, end: float, duration: float, energy_head: float) -> str:
    pos = start / max(duration, 1e-6)
    if pos < 0.18:
        return "intro-extension"
    if pos > 0.8:
        return "outro-extension"
    if energy_head >= 0.6:
        return "drop-loop"
    return "build-extension"


def detect_loops(
    grid: BeatGrid,
    energy_curve: list[EnergyPoint],
    features: AnalysisFeatures | None = None,
    duration: float = 0.0,
    max_results: int = 6,
) -> list[SavedLoop]:
    """Detect seamless Saved Loop candidates from the beatgrid + energy.

    Each phrase start is a candidate loop head. For each loop length in
    {4, 8, 16} bars that fits inside the track, we compute a seamless score
    and keep the best per (phrase, bars) — deduped by span.

    Loop ends come from the grid's downbeats when available (S-2), so a
    loop ends exactly on a downbeat even if the BPM estimate is slightly
    off; otherwise the end falls back to start + nominal duration and
    the candidate is flagged end_from_grid=False (see SavedLoop.to_dict).
    """
    if not grid.phrases:
        return []

    duration = duration or (float(grid.beats[-1].time) if grid.beats else 0.0)
    if duration <= 0:
        return []

    candidates: list[SavedLoop] = []
    seen_spans: set[tuple[float, float]] = set()

    for phrase in grid.phrases:
        start = phrase.start_time
        for bars in LOOP_BARS:
            nominal_sec = _loop_duration(grid.bpm, bars)
            end, end_from_grid = _loop_end_from_grid(
                grid, start, bars, start + nominal_sec
            )
            if end > duration + 1e-6:
                continue
            loop_sec = end - start
            if loop_sec <= 0:
                continue
            if bars > 4 and any(
                abs(s - start) < 0.5 and abs(e - end) < 0.5
                for (s, e) in seen_spans
            ):
                continue

            score = _seamless_score(energy_curve, start, end, loop_sec)
            if score < SEAMLESS_MIN:
                continue

            head_energy = _energy_at(energy_curve, start + loop_sec * 0.5)
            use_case = _use_case(start, end, duration, head_energy)
            reasons = []
            if use_case in ("intro-extension", "outro-extension"):
                reasons.append("edge extension")
            elif use_case == "drop-loop":
                reasons.append("high-energy seam")
            else:
                reasons.append("build energy")
            reasons.append(
                "grid downbeat end"
                if end_from_grid
                else "fallback end: start + nominal duration (no grid downbeat)"
            )

            seen_spans.add((start, end))
            candidates.append(
                SavedLoop(
                    bars=bars,
                    start=start,
                    end=end,
                    seamless_score=score,
                    use_case=use_case,
                    phrase_index=phrase.index,
                    reasons=reasons,
                    end_from_grid=end_from_grid,
                )
            )

    # Prefer longer seamless loops, then higher seamless score.
    candidates.sort(key=lambda c: (c.bars, c.seamless_score), reverse=True)
    return candidates[:max_results]