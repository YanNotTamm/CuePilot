"""Tests for mashup and crate organization API endpoints."""

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
    return TestClient(create_app())


_TRACKS = [
    {"id": "A", "key": "Am", "bpm": 128.0, "duration": 240.0, "energy": 0.8, "readiness": "ready",
     "energyCurve": [{"time": i / 4, "energy": 0.5} for i in range(16)]},
    {"id": "B", "key": "C", "bpm": 128.0, "duration": 240.0, "energy": 0.3, "readiness": "ready",
     "energyCurve": [{"time": i / 4, "energy": 0.5} for i in range(16)]},
    {"id": "C", "key": "F#m", "bpm": 175.0, "duration": 240.0, "energy": 0.2, "readiness": "needs_review",
     "energyCurve": [{"time": i / 4, "energy": 0.5} for i in range(16)]},
]


def test_mashup_endpoint(client):
    r = client.post(
        "/api/mashup/find",
        json={"tracks": _TRACKS, "top_n": 5, "min_score": 0.4},
        headers={"X-Session-Token": "test-token"},
    )
    assert r.status_code == 200
    candidates = r.json()["candidates"]
    assert isinstance(candidates, list)


def test_mashup_endpoint_requires_auth(client):
    r = client.post("/api/mashup/find", json={"tracks": _TRACKS})
    assert r.status_code == 401


def test_crates_endpoint(client):
    r = client.post(
        "/api/crates/organize",
        json={"tracks": _TRACKS},
        headers={"X-Session-Token": "test-token"},
    )
    assert r.status_code == 200
    data = r.json()
    assert "crates" in data and "assigned" in data
    assert data["assigned"] >= 1


def test_crates_endpoint_requires_auth(client):
    r = client.post("/api/crates/organize", json={"tracks": _TRACKS})
    assert r.status_code == 401


def test_planner_endpoint(client):
    r = client.post(
        "/api/planner/plan",
        json={"tracks": _TRACKS, "target_minutes": 10.0, "must_include": ["A"]},
        headers={"X-Session-Token": "test-token"},
    )
    assert r.status_code == 200
    data = r.json()
    assert "tracks" in data and "energyArc" in data
    assert data["totalDuration"] > 0


def test_planner_endpoint_requires_auth(client):
    r = client.post("/api/planner/plan", json={"tracks": _TRACKS})
    assert r.status_code == 401


def test_privacy_endpoint(client):
    r = client.post(
        "/api/privacy/resolve",
        json={"mode": "cloud"},
        headers={"X-Session-Token": "test-token"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["mode"] == "cloud"
    assert data["private"] is False
    assert "localServerAvailable" in data


def test_privacy_endpoint_requires_auth(client):
    r = client.post("/api/privacy/resolve", json={"mode": "cloud"})
    assert r.status_code == 401