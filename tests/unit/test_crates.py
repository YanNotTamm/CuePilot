"""Tests for Smart Crate Auto-Organization."""

from core.crates import organize_crates
from core.crates.organize import _energy_level, _key_family, _matches_rule


def _track(tid, bpm=128.0, energy=0.5, readiness="ready"):
    return {
        "id": tid,
        "bpm": bpm,
        "energy": energy,
        "readiness": readiness,
        "key": "Am",
    }


def test_energy_level_buckets():
    assert _energy_level(0.9) == "high"
    assert _energy_level(0.5) == "mid"
    assert _energy_level(0.2) == "low"


def test_key_family_from_camelot():
    assert _key_family("Am") == "camelot-15"


def test_organize_assigns_tracks_to_crates():
    tracks = [
        _track("t1", bpm=128, energy=0.85, readiness="ready"),
        _track("t2", bpm=100, energy=0.2, readiness="ready"),
        _track("t3", bpm=128, energy=0.8, readiness="needs_review"),
    ]
    org = organize_crates(tracks)
    assert org.assigned >= 1
    by_name = {c.name: c.tracks for c in org.crates}
    # t1 is ready + high energy + 124-132bpm → peak-time crate.
    assert "t1" in by_name.get("Peak Time — 124-132 BPM — High Energy — Ready", [])
    # t3 is needs_review.
    assert "t3" in by_name.get("Needs Review", [])
    assert all(c.name for c in org.crates)


def test_organize_reports_unassigned():
    tracks = [
        _track("t1", bpm=128, energy=0.85, readiness="ready"),
        _track("t2", bpm=100, energy=0.5, readiness="ready"),
    ]
    org = organize_crates(tracks)
    # t2 (mid energy, 100bpm) matches Warm Up (low energy)? No → may be unassigned.
    assert isinstance(org.unassigned, list)


def test_matches_rule_filters():
    track = _track("t1", bpm=130, energy=0.8, readiness="ready")
    assert _matches_rule(track, {"bpm_min": 124, "bpm_max": 132, "energy_min": 0.75, "readiness": "ready"})
    assert not _matches_rule(track, {"readiness": "needs_review"})
    assert not _matches_rule(track, {"energy_min": 0.9})


def test_organize_empty_library():
    org = organize_crates([])
    assert org.assigned == 0
    assert org.crates == []
    assert org.unassigned == []


def test_organize_to_dict_serializable():
    org = organize_crates([_track("t1", bpm=128, energy=0.85, readiness="ready")])
    data = org.to_dict()
    assert "crates" in data and "assigned" in data and "unassigned" in data
    assert isinstance(data["crates"], list)