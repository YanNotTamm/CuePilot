"""LLM privacy layer.

- `core.llm.privacy` — Local-First Privacy LLM Option: resolves cloud vs
  self-hosted local endpoints vs lightweight local fallback (Ollama).
"""

from __future__ import annotations

from .privacy import (
    DEFAULT_LOCAL_BASE_URL,
    DEFAULT_LOCAL_MODEL,
    MODE_CLOUD,
    MODE_LOCAL_LIGHT,
    MODE_SELF_HOSTED,
    MODES,
    PrivacyStatus,
    detect_local_server,
    resolve_privacy,
)

__all__ = [
    "DEFAULT_LOCAL_BASE_URL",
    "DEFAULT_LOCAL_MODEL",
    "MODE_CLOUD",
    "MODE_LOCAL_LIGHT",
    "MODE_SELF_HOSTED",
    "MODES",
    "PrivacyStatus",
    "detect_local_server",
    "resolve_privacy",
]