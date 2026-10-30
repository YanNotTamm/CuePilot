"""Tests for Local-First Privacy LLM configuration."""

from core.llm import (
    MODE_CLOUD,
    MODE_LOCAL_LIGHT,
    MODE_SELF_HOSTED,
    resolve_privacy,
)
from core.llm.privacy import _normalize_mode


def test_default_is_local_first_not_cloud():
    # Local-first: no mode given -> local-light, never cloud.
    status = resolve_privacy()
    assert status.mode == MODE_LOCAL_LIGHT
    assert status.private is True
    assert status.source == "default"


def test_cloud_mode_requires_explicit_consent():
    # Requesting cloud without consent must NOT activate cloud.
    status = resolve_privacy(mode=MODE_CLOUD)
    assert status.mode == MODE_LOCAL_LIGHT
    assert status.private is True
    assert status.requires_explicit_consent is True
    assert status.requested_mode == MODE_CLOUD
    assert status.warning


def test_cloud_mode_with_explicit_consent():
    status = resolve_privacy(mode=MODE_CLOUD, explicit_consent=True)
    assert status.mode == MODE_CLOUD
    assert status.private is False
    assert status.requires_explicit_consent is False


def test_self_hosted_is_private():
    status = resolve_privacy(mode=MODE_SELF_HOSTED)
    assert status.private is True
    assert status.base_url


def test_local_light_uses_defaults():
    status = resolve_privacy(mode=MODE_LOCAL_LIGHT)
    assert status.private is True
    assert "11434" in status.base_url


def test_explicit_override():
    status = resolve_privacy(mode=MODE_SELF_HOSTED, base_url="http://10.0.0.5:8000/v1", model="deepseek-v4")
    assert status.base_url == "http://10.0.0.5:8000/v1"
    assert status.model == "deepseek-v4"


def test_normalize_mode_rejects_unknown():
    # Unknown / missing mode falls back to local-first, not cloud.
    assert _normalize_mode("bogus") == MODE_LOCAL_LIGHT
    assert _normalize_mode(None) == MODE_LOCAL_LIGHT


def test_status_to_dict():
    data = resolve_privacy(mode=MODE_SELF_HOSTED).to_dict()
    assert data["mode"] == MODE_SELF_HOSTED
    assert data["private"] is True
    assert "baseUrl" in data and "model" in data and "note" in data