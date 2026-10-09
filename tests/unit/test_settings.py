"""Tests for user settings persistence (engine mode + enabled stems)."""

import os

import pytest

from core.settings import (
    DEFAULT_STEMS,
    STEM_NAMES,
    get_enabled_stems,
    get_engine_mode,
    set_enabled_stems,
    set_engine_mode,
)


@pytest.fixture(autouse=True)
def _isolate_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CUEPILOT_DATA_DIR", str(tmp_path))


def test_engine_default_and_roundtrip():
    assert get_engine_mode() == "basic"
    assert set_engine_mode("intellistem") == "intellistem"
    assert get_engine_mode() == "intellistem"


def test_engine_rejects_invalid():
    assert set_engine_mode("bogus") == "basic"
    assert get_engine_mode() == "basic"


def test_enabled_stems_default_all():
    assert get_enabled_stems() == list(DEFAULT_STEMS)
    assert set(STEM_NAMES) == {"vocal", "drums", "bass", "other"}


def test_enabled_stems_roundtrip():
    assert set_enabled_stems(["vocal", "bass"]) == ["vocal", "bass"]
    assert get_enabled_stems() == ["vocal", "bass"]


def test_enabled_stems_drops_invalid():
    assert set_enabled_stems(["vocal", "nope"]) == ["vocal"]
    assert get_enabled_stems() == ["vocal"]


def test_enabled_stems_empty_resets_to_all():
    assert set_enabled_stems([]) == list(DEFAULT_STEMS)
    assert set_enabled_stems(None) == list(DEFAULT_STEMS)


def test_enabled_stems_all_invalid_resets_to_all():
    assert set_enabled_stems(["bad", "worse"]) == list(DEFAULT_STEMS)
    assert get_enabled_stems() == list(DEFAULT_STEMS)