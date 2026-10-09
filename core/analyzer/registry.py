"""Engine registry for modular audio analyzers."""

from __future__ import annotations

from typing import Optional

from .base import AnalyzerEngine
from .librosa_engine import LibrosaEngine

_ENGINES: dict[str, type[AnalyzerEngine]] = {}


def register_engine(engine_cls: type[AnalyzerEngine]) -> None:
    _ENGINES[engine_cls.name] = engine_cls


def get_engine(name: Optional[str] = None) -> AnalyzerEngine:
    """Return an engine instance by name; default to librosa."""
    register_engine(LibrosaEngine)
    if not name:
        name = LibrosaEngine.name
    if name not in _ENGINES:
        raise ValueError(f"Unknown analyzer engine: {name}. Available: {sorted(_ENGINES)}")
    return _ENGINES[name]()
