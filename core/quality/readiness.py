"""Gig Readiness Score and track preparation status.

One 0..100 score per track that combines:

    0.35 * average cue confidence
    0.25 * audio quality        (from the QC report, Section 45.7)
    0.20 * structure completeness (all important sections detected?)
    0.20 * correction stability   (fewer manual corrections = more stable)

Rendered as a badge in the library view (`Ready` / `Needs Review` /
`Not Analyzed`) so preparing a 100-track gig becomes a one-screen triage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import Cue, GenreProfile, Section
from .qc import QcReport

# Readiness metric weights.
WEIGHTS = {
    "cue_confidence": 0.35,
    "audio_quality": 0.25,
    "structure": 0.20,
    "correction_stability": 0.20,
}

READY_THRESHOLD = 80.0
REVIEW_THRESHOLD = 50.0

# Sections required for a complete structure map.
_CORE_SECTIONS = {"INTRO", "DROP", "OUTRO"}
_OPTIONAL_SECTIONS = {"VOCAL", "BUILD", "BREAKDOWN", "CHORUS", "SECOND_DROP"}


@dataclass
class GigReadiness:
    """Gig Readiness Score and status badge."""

    score: float
    bucket: str  # "ready" | "needs_review" | "not_analyzed"
    components: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    @property
    def display_bucket(self) -> str:
        return {
            "ready": "Ready",
            "needs_review": "Needs Review",
            "not_analyzed": "Not Analyzed",
        }[self.bucket]

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 1),
            "bucket": self.bucket,
            "displayBucket": self.display_bucket,
            "components": {k: round(v, 3) for k, v in self.components.items()},
            "reasons": self.reasons,
        }


def readiness_bucket(score: float) -> str:
    if score >= READY_THRESHOLD:
        return "ready"
    if score >= REVIEW_THRESHOLD:
        return "needs_review"
    return "not_analyzed"


def _average_cue_confidence(cues: list[Cue] | None) -> float:
    if not cues:
        return 0.0
    return float(sum(c.confidence for c in cues) / len(cues))


def _structure_completeness(sections: list[Section] | None) -> tuple[float, list[str]]:
    """Coverage of core + as many optional sections as possible (0..1)."""
    if not sections:
        return 0.0, ["no sections detected"]
    present = {s.type.value for s in sections if hasattr(s.type, "value")}
    present |= {str(s.type) for s in sections}

    core_hit = len(_CORE_SECTIONS & present)
    core_total = len(_CORE_SECTIONS)
    optional_hit = len(_OPTIONAL_SECTIONS & present)
    optional_total = len(_OPTIONAL_SECTIONS)

    core_score = core_hit / core_total
    optional_score = optional_hit / optional_total
    score = 0.7 * core_score + 0.3 * optional_score

    reasons = []
    missing_core = _CORE_SECTIONS - present
    if missing_core:
        reasons.append("missing: " + ", ".join(sorted(missing_core)))
    return float(score), reasons


def _correction_stability(correction_count: int) -> float:
    """Stability scoring based on manual corrections."""
    if correction_count <= 0:
        return 1.0
    return float(max(0.0, 1.0 - 0.25 * correction_count))


def compute_readiness(
    cues: list[Cue] | None = None,
    sections: list[Section] | None = None,
    qc: QcReport | None = None,
    correction_count: int = 0,
    analyzed: bool = True,
) -> GigReadiness:
    """Compute the Gig Readiness Score from already-available analysis data.

    Args:
        cues: generated cue list (may be empty when not analyzed).
        sections: detected structure sections.
        qc: audio QC report (Section 45.7); None → quality component 0.
        correction_count: how many manual corrections exist for this track
            (learn store feedback count).
        analyzed: whether the track has been analyzed at all.
    """
    reasons: list[str] = []
    if not analyzed or (not cues and not sections):
        return GigReadiness(
            score=0.0,
            bucket="not_analyzed",
            components={
                "cue_confidence": 0.0,
                "audio_quality": qc.score if qc else 0.0,
                "structure": 0.0,
                "correction_stability": 1.0,
            },
            reasons=["not analyzed"],
        )

    components = {
        "cue_confidence": _average_cue_confidence(cues),
        "audio_quality": qc.score if qc else 0.0,
        "structure": 0.0,
        "correction_stability": _correction_stability(correction_count),
    }
    components["structure"], struct_reasons = _structure_completeness(sections)
    reasons.extend(struct_reasons)

    if correction_count > 0:
        reasons.append(f"{correction_count} manual correction(s)")

    total = 0.0
    for key, weight in WEIGHTS.items():
        total += weight * components[key]
    score = float(np_clip(total * 100.0, 0.0, 100.0))

    return GigReadiness(
        score=score,
        bucket=readiness_bucket(score),
        components=components,
        reasons=reasons,
    )


def np_clip(value: float, lo: float, hi: float) -> float:
    try:
        import numpy as np

        return float(np.clip(value, lo, hi))
    except Exception:
        return max(lo, min(hi, value))