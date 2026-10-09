"""Tests for Local-First Privacy LLM configuration."""

from core.llm import (
    MODE_CLOUD,
    MODE_LOCAL_LIGHT,
    MODE_SELF_HOSTED,
    resolve_privacy,
)
from core.llm.privacy import _normalize_mode


def test_cloud_mode_default_not_private():
    status = resolve_privacy()
    assert status.mode == MODE_CLOUD
    assert status.private is False


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
    assert _normalize_mode("bogus") == MODE_CLOUD
    assert _normalize_mode(None) == MODE_CLOUD


def test_status_to_dict():
    data = resolve_privacy(mode=MODE_SELF_HOSTED).to_dict()
    assert data["mode"] == MODE_SELF_HOSTED
    assert data["private"] is True
    assert "baseUrl" in data and "model" in data and "note" in data