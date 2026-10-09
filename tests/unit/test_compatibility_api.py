"""Tests for the Harmonic Mixing Advisor API endpoint."""

import os
import sys

import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("CUEPILOT_SESSION_TOKEN", "test-token")
os.environ.setdefault("CUEPILOT_DATA_DIR", os.path.join(os.path.dirname(__file__), ".tmp"))

from fastapi.testclient import TestClient  # noqa: E402

from apps.desktop.server import create_app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    app = create_app()
    return TestClient(app)


_TRACKS = [
    {"id": "A", "key": "Am", "bpm": 128.0, "duration": 240.0,
     "energyCurve": [{"time": i / 4, "energy": 0.5} for i in range(16)]},
    {"id": "B", "key": "C", "bpm": 128.0, "duration": 240.0,
     "energyCurve": [{"time": i / 4, "energy": 0.5} for i in range(16)]},
    {"id": "C", "key": "F#m", "bpm": 175.0, "duration": 240.0,
     "energyCurve": [{"time": i / 4, "energy": 0.5} for i in range(16)]},
]


def test_compatibility_returns_ranked_matches(client):
    r = client.post(
        "/api/compatibility",
        json={"tracks": _TRACKS, "track_id": "A"},
        headers={"X-Session-Token": "test-token"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["trackId"] == "A"
    matches = data["compatibleWith"]
    assert len(matches) >= 1
    assert matches[0]["trackId"] == "B"
    assert "keyCompatibility" in matches[0]
    assert "score" in matches[0]
    assert matches[0]["score"] > 0.8


def test_compatibility_excludes_incompatible(client):
    r = client.post(
        "/api/compatibility",
        json={"tracks": _TRACKS, "track_id": "A", "min_score": 0.6},
        headers={"X-Session-Token": "test-token"},
    )
    data = r.json()
    ids = [m["trackId"] for m in data["compatibleWith"]]
    assert "C" not in ids


def test_compatibility_empty_library(client):
    r = client.post(
        "/api/compatibility",
        json={"tracks": [], "track_id": "A"},
        headers={"X-Session-Token": "test-token"},
    )
    assert r.status_code == 200
    assert r.json()["compatibleWith"] == []


def test_compatibility_requires_auth(client):
    r = client.post("/api/compatibility", json={"tracks": _TRACKS, "track_id": "A"})
    assert r.status_code == 401