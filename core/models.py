"""Core data models for CuePilot.

Dataclasses for tracks, analysis runs, sections, cues, cue profiles, jobs, and backups.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


class TrackStatus(str, Enum):
    QUEUED = "QUEUED"
    ANALYZING = "ANALYZING"
    ANALYZED = "ANALYZED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class SectionType(str, Enum):
    INTRO = "INTRO"
    VOCAL = "VOCAL"
    BUILD = "BUILD"
    DROP = "DROP"
    BREAKDOWN = "BREAKDOWN"
    CHORUS = "CHORUS"
    SECOND_DROP = "SECOND_DROP"
    OUTRO = "OUTRO"


class CueType(str, Enum):
    INTRO = "INTRO"
    VOCAL = "VOCAL"
    BUILD = "BUILD"
    DROP = "DROP"
    BREAK = "BREAK"
    SECOND_DROP = "SECOND_DROP"
    OUTRO = "OUTRO"
    CUSTOM = "CUSTOM"


class CueRole(str, Enum):
    """Semantic role of a hot-cue slot (Genre_pattern.md section 64).

    A slot's role is genre-dependent: e.g. slot E is DROP for EDM but MAIN
    GROOVE for Baile Funk. Kept separate from SectionType so one track can
    reuse the same 8 physical slots with genre-specific meaning.
    """

    START = "START"
    INTRO = "INTRO"
    MIX_IN = "MIX_IN"
    VOCAL = "VOCAL"
    HOOK = "HOOK"
    MELODY = "MELODY"
    GROOVE = "GROOVE"
    PRE_GROOVE = "PRE_GROOVE"
    BUILD = "BUILD"
    PRE_DROP = "PRE_DROP"
    DROP = "DROP"
    CLIMAX = "CLIMAX"
    TRANSITION = "TRANSITION"
    CHORUS = "CHORUS"
    BREAKDOWN = "BREAKDOWN"
    ENERGY_PEAK = "ENERGY_PEAK"
    SECOND_DROP = "SECOND_DROP"
    OUTRO = "OUTRO"
    MIX_OUT = "MIX_OUT"
    ALTERNATIVE = "ALTERNATIVE"


class ConfidenceBucket(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


def confidence_bucket(score: float) -> ConfidenceBucket:
    """Map a 0..1 confidence score to a bucket."""
    if score >= 0.90:
        return ConfidenceBucket.HIGH
    if score >= 0.75:
        return ConfidenceBucket.MEDIUM
    return ConfidenceBucket.LOW


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS.mmm (Serato-style timestamp)."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        millis = 0
        secs += 1
    if secs >= 60:
        secs = 0
        minutes += 1
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"


@dataclass
class Track:
    """A music file discovered by the scanner."""

    id: str
    path: str
    filename: str
    artist: str = ""
    title: str = ""
    album: str = ""
    duration: float = 0.0
    file_hash: str = ""
    status: TrackStatus = TrackStatus.QUEUED
    error: str = ""
    created_at: str = ""
    updated_at: str = ""

    @property
    def display_name(self) -> str:
        if self.artist and self.title:
            return f"{self.artist} - {self.title}"
        return self.filename

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass
class Beat:
    """A single detected beat."""

    time: float
    is_downbeat: bool = False
    index: int = 0


@dataclass
class Bar:
    """A bar, starting at a downbeat."""

    index: int
    downbeat_time: float
    beat_count: int = 4


@dataclass
class Phrase:
    """A group of bars forming a musical phrase."""

    index: int
    start_time: float
    end_time: float
    bar_count: int


@dataclass
class EnergyPoint:
    """A point on the energy curve."""

    time: float
    energy: float


@dataclass
class Analysis:
    """Full analysis output.

    `beats`, `downbeats`, `bars`, `phrases` and the energy curve carry the
    musical-timing information. `spectral_features` is a
    compact summary of audio features.
    """

    track_id: str
    duration: float
    sample_rate: int
    channels: int
    bpm: float
    key: str
    beats: list[Beat]
    downbeats: list[float]
    bars: list[Bar]
    phrases: list[Phrase]
    energy_curve: list[EnergyPoint]
    spectral_features: dict[str, Any]
    engine: str = "librosa"
    engine_version: str = "0.1.0"
    confidence: float = 0.0
    created_at: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def feature_summary(self) -> dict[str, Any]:
        """Structured feature summary safe to send to the LLM layer (Section 10).

        Never includes raw audio, waveform samples, or full file paths.
        """
        return {
            "duration": round(self.duration, 3),
            "sampleRate": self.sample_rate,
            "channels": self.channels,
            "bpm": round(self.bpm, 2),
            "key": self.key,
            "downbeatCount": len(self.downbeats),
            "phraseCount": len(self.phrases),
            "energyCurve": [
                {"time": round(p.time, 3), "energy": round(p.energy, 4)}
                for p in self.energy_curve
            ],
            "spectralFeatures": self.spectral_features,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "trackId": self.track_id,
            "duration": self.duration,
            "sampleRate": self.sample_rate,
            "channels": self.channels,
            "bpm": round(self.bpm, 3),
            "key": self.key,
            "beats": [b.time for b in self.beats],
            "downbeats": self.downbeats,
            "bars": [asdict(b) for b in self.bars],
            "phrases": [asdict(p) for p in self.phrases],
            "energyCurve": [asdict(p) for p in self.energy_curve],
            "spectralFeatures": self.spectral_features,
            "engine": self.engine,
            "engineVersion": self.engine_version,
            "confidence": self.confidence,
            "extras": self.extras,
        }


@dataclass
class Section:
    """A detected musical section."""

    type: SectionType
    start: float
    end: float
    confidence: float
    source: str = "dsp"

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        type_val = self.type.value if isinstance(self.type, Enum) else str(self.type)
        return {
            "section": type_val,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "confidence": round(self.confidence, 3),
            "source": self.source,
        }


@dataclass
class Cue:
    """A hot cue candidate."""

    slot: int
    type: CueType
    time: float
    label: str
    confidence: float
    color: str
    locked: bool = False
    source: str = "dsp"
    reason: list[str] = field(default_factory=list)
    created_at: str = ""

    @property
    def bucket(self) -> ConfidenceBucket:
        return confidence_bucket(self.confidence)

    def to_dict(self) -> dict[str, Any]:
        type_val = self.type.value if isinstance(self.type, Enum) else str(self.type)
        return {
            "slot": self.slot,
            "type": type_val,
            "time": round(self.time, 3),
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "color": self.color,
            "locked": self.locked,
            "source": self.source,
            "reason": self.reason,
        }


@dataclass
class CuePlan:
    """A complete cue plan for one track (canonical export)."""

    schema_version: str = "1.0"
    track: Optional[dict[str, Any]] = None
    cues: list[Cue] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    strategy_used: str = ""
    genre_detected: str = ""
    genre_confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schemaVersion": self.schema_version,
            "track": self.track or {},
            "cues": [c.to_dict() for c in self.cues],
        }
        if self.strategy_used:
            out["strategyUsed"] = self.strategy_used
        if self.genre_detected:
            out["genreDetected"] = self.genre_detected
        if self.genre_confidence > 0:
            out["genreConfidence"] = round(self.genre_confidence, 3)
        return out


@dataclass
class GenreProfile:
    """A genre-specific cueing profile.

    Defines genre-aware cue mapping concepts:
    `slot_map` maps each physical slot to a semantic section type, and
    `cue_weights` bias which audio features matter most for the genre.
    """

    name: str
    bpm_range: tuple[float, float]
    section_order: list[SectionType]
    slot_map: dict[SectionType, int]
    cue_weights: dict[str, float]
    display_name: str = ""
    description: str = ""
    description_id: str = ""
    cue_strategy: Optional[dict[int, CueRole]] = None
    detection_weights: Optional[dict[str, float]] = None
    preferred_phrase_bars: list[int] = field(default_factory=lambda: [8, 16, 32])
    energy_profile: str = "build_drop"
    transition_style: str = "phrase"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "displayName": self.display_name or self.name,
            "description": self.description,
            "descriptionId": self.description_id or self.description,
            "bpmRange": list(self.bpm_range),
            "sectionOrder": [s.value for s in self.section_order],
            "slotMap": {s.value: v for s, v in self.slot_map.items()},
            "cueStrategy": (
                {str(k): v.value for k, v in self.cue_strategy.items()}
                if self.cue_strategy
                else {}
            ),
            "detectionWeights": self.detection_weights or {},
            "preferredPhraseBars": self.preferred_phrase_bars,
            "energyProfile": self.energy_profile,
            "transitionStyle": self.transition_style,
        }


@dataclass
class ScanResult:
    """Result of a folder scan."""

    folder: str
    tracks: list[Track]
    total: int
    duplicates: int
    failed: int
    elapsed_seconds: float
