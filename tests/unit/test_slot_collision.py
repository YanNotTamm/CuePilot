"""Slot-collision tests (T-2) — runnable without librosa.

The repo's core.analyzer package imports librosa at package level, so
when librosa is absent we install a bare stub module first; no audio
feature is used at runtime here (all inputs are synthetic).
"""

import sys
from types import ModuleType

try:
    import librosa  # noqa: F401
except ImportError:  # pragma: no cover - only on librosa-less machines
    sys.modules.setdefault("librosa", ModuleType("librosa"))

import numpy as np

from core.analyzer.base import AnalysisFeatures
from core.beatgrid import BeatGrid
from core.cues import generate_cueplan
from core.cues.generator import _dedupe_slots, _slot_for
from core.models import Bar, Beat, Phrase, Section, SectionType
from core.profiles.genre import DEFAULT_SLOT_MAP, get_profile


def _features() -> AnalysisFeatures:
    n = 200
    return AnalysisFeatures(
        hop_length=512,
        sample_rate=22050,
        frame_times=np.linspace(0.0, 60.0, n),
        onset_env=np.full(n, 0.8),
        rms=np.full(n, 0.5),
        spectral_flux=np.full(n, 0.5),
        harmonic=np.full(n, 0.5),
        percussive=np.full(n, 0.5),
    )


def _grid() -> BeatGrid:
    beats = [Beat(time=t, is_downbeat=(i % 4 == 0), index=i)
             for i, t in enumerate(np.arange(0.0, 60.0, 0.5))]
    downbeats = [float(t) for t in np.arange(0.0, 60.0, 2.0)]
    return BeatGrid(
        bpm=120.0,
        beats=beats,
        downbeat_times=downbeats,
        bars=[Bar(index=i, downbeat_time=t, beat_count=4)
              for i, t in enumerate(downbeats)],
        phrases=[Phrase(index=0, start_time=0.0, end_time=60.0, bar_count=30)],
        confidence=0.9,
    )


def _sec(type_: SectionType, start: float) -> Section:
    return Section(type=type_, start=start, end=start + 8.0,
                   confidence=0.9, source="dsp")


def test_default_slot_map_covers_all_section_types():
    # Every cue-bearing SectionType must have an explicit map entry:
    # no type may rely on a fallback slot (the old bug: CHORUS fell
    # back to 8). UNKNOWN is intentionally unmapped: it is an explicit
    # "not classified" marker and must never receive a hot cue.
    for type_ in SectionType:
        if type_ == SectionType.UNKNOWN:
            assert type_ not in DEFAULT_SLOT_MAP
            continue
        assert type_ in DEFAULT_SLOT_MAP, f"{type_} missing from DEFAULT_SLOT_MAP"
        assert _slot_for(type_, DEFAULT_SLOT_MAP) == DEFAULT_SLOT_MAP[type_]


def test_slot_sharing_is_only_vocal_chorus():
    # Slots must be in 1..8, and the ONLY shared slot is 3 (VOCAL/CHORUS,
    # documented sharing: both are CueType.VOCAL and first-wins applies).
    by_slot: dict[int, list[SectionType]] = {}
    for type_, slot in DEFAULT_SLOT_MAP.items():
        assert 1 <= slot <= 8, f"{type_} slot {slot} outside 1..8"
        by_slot.setdefault(slot, []).append(type_)
    shared = {s: ts for s, ts in by_slot.items() if len(ts) > 1}
    assert shared == {3: [SectionType.VOCAL, SectionType.CHORUS]}
    assert DEFAULT_SLOT_MAP[SectionType.OUTRO] == 8


def test_slot_for_never_falls_back_to_outro():
    # A partial/empty custom map must fall back to the default map, not
    # silently to OUTRO's slot 8.
    assert _slot_for(SectionType.CHORUS, {}) == 3
    assert _slot_for(SectionType.OUTRO, {}) == 8
    assert _slot_for(SectionType.VOCAL, {SectionType.VOCAL: 5}) == 5


def test_chorus_and_outro_both_survive():
    # Regression for the original bug: a track with CHORUS but no VOCAL
    # used to map CHORUS -> 8, colliding with OUTRO and losing one cue.
    plan = generate_cueplan(
        None, _features(), _grid(),
        [_sec(SectionType.INTRO, 0.0), _sec(SectionType.CHORUS, 16.0),
         _sec(SectionType.OUTRO, 56.0)],
        get_profile("open_format"),
    )
    slots = [c.slot for c in plan.cues]
    assert len(slots) == len(set(slots)), f"duplicate slots: {slots}"
    chorus = [c for c in plan.cues if c.slot == 3]
    outro = [c for c in plan.cues if c.slot == 8]
    assert len(chorus) == 1 and len(outro) == 1


def test_all_section_types_no_silent_slot_loss():
    # All 8 SectionTypes in one track: 8 cues, 8 unique slots, and the
    # skipped CHORUS (first vocal-type wins) is reported on the VOCAL cue.
    sections = [
        _sec(SectionType.INTRO, 0.0),
        _sec(SectionType.VOCAL, 8.0),
        _sec(SectionType.CHORUS, 16.0),
        _sec(SectionType.BUILD, 24.0),
        _sec(SectionType.DROP, 32.0),
        _sec(SectionType.BREAKDOWN, 40.0),
        _sec(SectionType.SECOND_DROP, 48.0),
        _sec(SectionType.OUTRO, 56.0),
    ]
    plan = generate_cueplan(None, _features(), _grid(), sections,
                            get_profile("open_format"))
    slots = [c.slot for c in plan.cues]
    assert len(slots) == len(set(slots)) == 8, f"slots: {slots}"
    assert sorted(slots) == [1, 2, 3, 4, 5, 6, 7, 8]
    vocal = [c for c in plan.cues if c.slot == 3][0]
    assert any("CHORUS" in r for r in vocal.reason), vocal.reason


def test_dedupe_collision_is_annotated_not_silent():
    # Two DROP sections share slot 5; the survivor must record the drop.
    plan = generate_cueplan(
        None, _features(), _grid(),
        [_sec(SectionType.DROP, 8.0), _sec(SectionType.DROP, 24.0)],
        get_profile("open_format"),
    )
    assert len(plan.cues) == 1
    assert any("slot collision" in r for r in plan.cues[0].reason), \
        plan.cues[0].reason


def test_dedupe_keeps_highest_confidence():
    from core.models import Cue, CueType

    low = Cue(slot=5, type=CueType.DROP, time=8.0, label="D",
              confidence=0.4, color="#fff")
    high = Cue(slot=5, type=CueType.DROP, time=24.0, label="D",
               confidence=0.9, color="#fff")
    out = _dedupe_slots([low, high])
    assert len(out) == 1 and out[0].time == 24.0
    assert any("slot collision" in r for r in out[0].reason)
