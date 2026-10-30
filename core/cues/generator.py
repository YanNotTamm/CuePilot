"""Hot cue generation.

Builds a CuePlan from the beatgrid and structure sections, assigning each
detected section to a cue slot per the default Cue Map and scoring each
cue with the configurable scoring formula.

Slot rules (T-2): the 1-8 schema cannot give all 8 SectionTypes a unique
slot, because slot 1 is START (generated from INTRO) and slots 2-8 are
only 7 slots. CHORUS therefore shares slot 3 with VOCAL by design (both
are CueType.VOCAL); the first vocal-type section wins that slot and any
later vocal-type section is skipped, which is reported on the surviving
cue's reasons. Any remaining slot collision (e.g. two DROP sections) is
resolved by _dedupe_slots keeping the highest-confidence cue and
annotating the surviving cue's reasons with what was dropped -- never
silently.
"""

from __future__ import annotations

import numpy as np

from ..analyzer import AnalysisFeatures
from ..beatgrid import BeatGrid, snap_to_beat, snap_to_downbeat
from ..models import Cue, CuePlan, CueType, GenreProfile, Section, SectionType
from ..profiles.genre import DEFAULT_SLOT_MAP
from ..scoring.scorer import compute_cue_score

_SECTION_TO_CUE_TYPE: dict[SectionType, CueType] = {
    SectionType.INTRO: CueType.INTRO,
    SectionType.VOCAL: CueType.VOCAL,
    SectionType.BUILD: CueType.BUILD,
    SectionType.DROP: CueType.DROP,
    SectionType.BREAKDOWN: CueType.BREAK,
    SectionType.CHORUS: CueType.VOCAL,
    SectionType.SECOND_DROP: CueType.SECOND_DROP,
    SectionType.OUTRO: CueType.OUTRO,
}

_TYPE_COLORS: dict[CueType, str] = {
    CueType.INTRO: "#2ECC71",
    CueType.VOCAL: "#F1C40F",
    CueType.BUILD: "#E67E22",
    CueType.DROP: "#E74C3C",
    CueType.BREAK: "#3498DB",
    CueType.SECOND_DROP: "#9B59B6",
    CueType.OUTRO: "#1ABC9C",
    CueType.CUSTOM: "#95A5A6",
}

# Sections that collapse to the same cue type: only the first occurrence wins
# that slot; later duplicates are skipped and reported on the surviving cue
# (e.g. a second VOCAL, or a CHORUS after a VOCAL, both CueType.VOCAL, share
# slot 3 by design -- see DEFAULT_SLOT_MAP -- so only the first is emitted).
_SKIP_AFTER_FIRST: set[CueType] = {CueType.INTRO, CueType.VOCAL, CueType.BUILD}


def _slot_for(type_: SectionType, slot_map: dict[SectionType, int]) -> int:
    """Return the slot for a section type, never a silent OUTRO collision.

    A missing entry used to fall back to slot 8, which silently collided
    with OUTRO (T-2: CHORUS had no map entry and stole OUTRO's slot).
    Now a partial custom map falls back to DEFAULT_SLOT_MAP, and a type
    missing from both raises KeyError instead of colliding.
    """
    if type_ in slot_map:
        return slot_map[type_]
    if type_ in DEFAULT_SLOT_MAP:
        return DEFAULT_SLOT_MAP[type_]
    raise KeyError(f"no cue slot mapped for section type {type_!r}")


def _snap_type(time: float, type_: SectionType, grid: BeatGrid) -> float:
    """Snap cue time: INTRO/OUTRO to nearest beat, section cues to downbeat."""
    if type_ in (SectionType.INTRO, SectionType.OUTRO):
        return snap_to_beat(time, grid)
    return snap_to_downbeat(time, grid)


def _first_reliable_beat(features: AnalysisFeatures, grid: BeatGrid) -> float:
    """Find the first beat that is actually audible (Genre_pattern.md section 8).

    Cue A (START) must not be timestamp 0: audio can begin with silence,
    noise, a count-in, a producer tag, or an ambient opening. Scan beats in
    time order and pick the first whose onset strength clears a relative
    threshold. Falls back to the first downbeat, then to 0.
    """
    if not grid.beats:
        return 0.0
    onset = features.onset_env
    if onset.size == 0:
        return float(grid.downbeat_times[0]) if grid.downbeat_times else 0.0

    threshold = max(
        float(np.percentile(onset, 85)) * 0.35,
        float(np.max(onset)) * 0.05,
    )
    sr = float(features.sample_rate)
    hop = int(features.hop_length)

    for b in grid.beats:
        frame = int(round(b.time * sr / hop))
        if 0 <= frame < onset.size and onset[frame] >= threshold:
            return b.time

    return float(grid.downbeat_times[0]) if grid.downbeat_times else 0.0


def generate_cueplan(
    analysis,
    features: AnalysisFeatures,
    grid: BeatGrid,
    sections: list[Section],
    profile: GenreProfile,
    cue_weights: dict[str, float] | None = None,
) -> CuePlan:
    """Generate a complete cue plan for one analyzed track.

    `cue_weights` optionally overrides the profile's default weights (used by
    the personalization layer for adaptive scoring).
    """
    slot_map = profile.slot_map or DEFAULT_SLOT_MAP
    cue_strategy = profile.cue_strategy or {}
    weights = cue_weights or profile.cue_weights

    cues: list[Cue] = []
    used_types: set[CueType] = set()
    # (cue_type, section_type) pairs skipped by the first-wins rule, so the
    # surviving cue can report them instead of dropping them silently.
    skipped_vocal_types: list[tuple[CueType, SectionType]] = []

    start_beat = _first_reliable_beat(features, grid)

    for sec in sorted(sections, key=lambda s: s.start):
        # UNKNOWN has no cue type by design: an explicitly unclassified
        # section must never invent a hot cue.
        cue_type = _SECTION_TO_CUE_TYPE.get(sec.type)
        if cue_type is None:
            continue
        if cue_type in _SKIP_AFTER_FIRST and cue_type in used_types:
            skipped_vocal_types.append((cue_type, sec.type))
            continue
        used_types.add(cue_type)

        slot = _slot_for(sec.type, slot_map)

        if sec.type == SectionType.INTRO:
            # Cue A = START: the first usable beat, not raw section start.
            # Cue B = INTRO: the mix-in point of the intro section.
            start_time = snap_to_beat(max(sec.start, start_beat), grid)
            cues.append(_build_cue(features, grid, profile, sec, slot=1, time=start_time, weights=weights))
            intro_slot = cue_strategy and next(
                (k for k, v in cue_strategy.items() if v.value == "INTRO" and k != 1),
                2,
            )
            intro_time = _snap_type(sec.start, sec.type, grid)
            cues.append(_build_cue(features, grid, profile, sec, slot=intro_slot, time=intro_time, weights=weights))
            continue

        time = _snap_type(sec.start, sec.type, grid)
        cues.append(_build_cue(features, grid, profile, sec, slot=slot, time=time, weights=weights))

    cues.sort(key=lambda c: (c.time, c.slot))
    cues = _dedupe_slots(cues)
    for cue_type, sec_type in skipped_vocal_types:
        for cue in cues:
            if cue.type == cue_type:
                cue.reason.append(
                    f"shared slot {cue.slot}: first {cue_type.value} wins, "
                    f"later {sec_type.value} section not given a separate cue"
                )
                break
    return CuePlan(schema_version="1.0", cues=cues, sections=sections)


def _build_cue(
    features: AnalysisFeatures,
    grid: BeatGrid,
    profile: GenreProfile,
    sec: Section,
    slot: int,
    time: float,
    weights: dict[str, float] | None = None,
) -> Cue:
    cue_type = _SECTION_TO_CUE_TYPE[sec.type]
    cue_strategy = profile.cue_strategy or {}
    score = compute_cue_score(
        features,
        grid,
        profile,
        sec.type,
        time,
        weights=weights or profile.cue_weights,
    )

    # Blend section confidence with the local cue score.
    confidence = 0.6 * sec.confidence + 0.4 * score.total
    confidence = max(0.0, min(1.0, confidence))

    role = cue_strategy.get(slot)
    label = role.value if role is not None else cue_type.value
    return Cue(
        slot=slot,
        type=cue_type,
        time=time,
        label=label,
        confidence=confidence,
        color=_TYPE_COLORS[cue_type],
        locked=False,
        source="dsp",
        reason=_build_reasons(sec, score.components),
    )


def _dedupe_slots(cues: list[Cue]) -> list[Cue]:
    """Keep exactly one cue per slot, preferring the highest confidence.

    A track may legitimately contain two drop-like sections, but the Serato
    hot-cue map (Section 8) has a single slot per semantic, so the stronger
    candidate wins. The drop is NOT silent: the surviving cue's reasons
    record each dropped competitor (type and time) so the loss is visible
    in the exported plan.
    """
    by_slot: dict[int, Cue] = {}
    dropped: dict[int, list[Cue]] = {}
    for cue in cues:
        prev = by_slot.get(cue.slot)
        if prev is None or cue.confidence > prev.confidence:
            if prev is not None:
                dropped.setdefault(cue.slot, []).append(prev)
            by_slot[cue.slot] = cue
        else:
            dropped.setdefault(cue.slot, []).append(cue)
    result = sorted(by_slot.values(), key=lambda c: (c.time, c.slot))
    for cue in result:
        for loser in dropped.get(cue.slot, []):
            cue.reason.append(
                f"slot collision on slot {cue.slot}: kept over "
                f"{loser.type.value} at {loser.time:.2f}s (lower confidence)"
            )
    return result


def _build_reasons(sec: Section, components: dict[str, float]) -> list[str]:
    """Explainable reasons for each cue placement decision."""
    reasons: list[str] = []
    strongest = sorted(components.items(), key=lambda kv: kv[1], reverse=True)
    for key, val in strongest[:3]:
        if val >= 0.3:
            reasons.append(f"{key.replace('_', ' ')} ({val:.0%})")
    reasons.append(f"section {sec.type.value} confidence {sec.confidence:.0%}")
    return reasons
