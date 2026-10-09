"""Unit tests for core models."""

from core.models import (
    ConfidenceBucket,
    Cue,
    CuePlan,
    Section,
    SectionType,
    Track,
    TrackStatus,
    confidence_bucket,
    format_timestamp,
)


def test_format_timestamp():
    assert format_timestamp(0.0) == "00:00.000"
    assert format_timestamp(64.32) == "01:04.320"
    assert format_timestamp(59.0) == "00:59.000"
    assert format_timestamp(234.52) == "03:54.520"


def test_confidence_bucket_boundaries():
    assert confidence_bucket(0.99) == ConfidenceBucket.HIGH
    assert confidence_bucket(0.90) == ConfidenceBucket.HIGH
    assert confidence_bucket(0.89) == ConfidenceBucket.MEDIUM
    assert confidence_bucket(0.75) == ConfidenceBucket.MEDIUM
    assert confidence_bucket(0.74) == ConfidenceBucket.LOW


def test_cue_bucket_property():
    cue = Cue(slot=1, type="INTRO", time=0.0, label="INTRO", confidence=0.94, color="green")
    assert cue.bucket == ConfidenceBucket.HIGH


def test_track_display_name():
    t1 = Track(id="1", path="p.mp3", filename="p.mp3", artist="A", title="B")
    t2 = Track(id="2", path="p.mp3", filename="p.mp3")
    assert t1.display_name == "A - B"
    assert t2.display_name == "p.mp3"


def test_track_status_serialization():
    t = Track(id="1", path="p.mp3", filename="p.mp3", status=TrackStatus.QUEUED)
    assert t.to_dict()["status"] == "QUEUED"


def test_section_to_dict():
    s = Section(type=SectionType.DROP, start=64.12, end=96.18, confidence=0.94)
    d = s.to_dict()
    assert d == {
        "section": "DROP",
        "start": 64.12,
        "end": 96.18,
        "confidence": 0.94,
        "source": "dsp",
    }


def test_cue_plan_to_dict():
    plan = CuePlan(schema_version="1.0", cues=[])
    assert plan.to_dict()["schemaVersion"] == "1.0"
    assert plan.to_dict()["cues"] == []
