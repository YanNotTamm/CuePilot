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
BARS_PER_PHRASE = 4
# Minimum seamless score to offer a loop candidate.
SEAMLESS_MIN = 0.55


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
        }


def _energy_at(energy_curve: list[EnergyPoint], time: float) -> float:
    if not energy_curve:
        return 0.5
    times = np.asarray([p.time for p in energy_curve], dtype=float)
    vals = np.asarray([p.energy for p in energy_curve], dtype=float)
    idx = int(np.searchsorted(times, time))
    idx = min(max(idx, 0), vals.size - 1)
    return float(vals[idx])


def _loop_duration(bpm: float, bars: int) -> float:
    """Duration in seconds of `bars` bars at `bpm` (4/4)."""
    if bpm <= 0:
        return 0.0
    beat = 60.0 / bpm
    return bars * BARS_PER_PHRASE * beat


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
            loop_sec = _loop_duration(grid.bpm, bars)
            end = start + loop_sec
            if end > duration + 1e-6:
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
                )
            )

    # Prefer longer seamless loops, then higher seamless score.
    candidates.sort(key=lambda c: (c.bars, c.seamless_score), reverse=True)
    return candidates[:max_results]