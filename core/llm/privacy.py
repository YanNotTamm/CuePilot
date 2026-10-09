"""Local-First Privacy LLM Option.

Provides configuration and resolution for semantic analysis backends:
- Cloud endpoint (OpenAI-compatible).
- Self-hosted inference server for zero data egress.
- Lightweight local model fallback (e.g. Ollama).

Resolves endpoint settings and reports privacy status.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Privacy modes.
MODE_CLOUD = "cloud"
MODE_SELF_HOSTED = "self-hosted"
MODE_LOCAL_LIGHT = "local-light"
MODES = (MODE_CLOUD, MODE_SELF_HOSTED, MODE_LOCAL_LIGHT)

# Environment keys used to point at a local inference endpoint.
ENV_LOCAL_BASE_URL = "CUEPILOT_LLM_BASE_URL"
ENV_LOCAL_MODEL = "CUEPILOT_LLM_MODEL"
ENV_PRIVACY_MODE = "CUEPILOT_LLM_MODE"

# Defaults for the light local fallback (Ollama-style OpenAI-compatible API).
DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_LOCAL_MODEL = "qwen2.5:7b"


@dataclass
class PrivacyStatus:
    """Resolution of the privacy mode."""

    mode: str
    private: bool
    base_url: str
    model: str
    source: str  # "env" | "settings" | "default"
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "private": self.private,
            "baseUrl": self.base_url,
            "model": self.model,
            "source": self.source,
            "note": self.note,
        }


def _normalize_mode(mode: str | None) -> str:
    if mode in MODES:
        return mode
    return MODE_CLOUD


def resolve_privacy(
    mode: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> PrivacyStatus:
    """Resolve the effective privacy configuration.

    Order of precedence:
    1. Environment variables (never baked into the repo).
    2. Explicit arguments (from app settings).
    3. Defaults.

    Returns a `PrivacyStatus` describing the mode, whether it is fully
    private, and which endpoint/model the semantic layer should use.
    """
    import os

    env_mode = os.environ.get(ENV_PRIVACY_MODE)
    resolved_mode = _normalize_mode(mode or env_mode)

    if resolved_mode == MODE_CLOUD:
        return PrivacyStatus(
            mode=MODE_CLOUD,
            private=False,
            base_url="",
            model="deepseek-v4-flash",
            source="env" if env_mode else "default",
            note="Cloud inference endpoint. Audio feature summaries are sent for analysis.",
        )

    if resolved_mode == MODE_SELF_HOSTED:
        url = base_url or os.environ.get(ENV_LOCAL_BASE_URL) or DEFAULT_LOCAL_BASE_URL
        mdl = model or os.environ.get(ENV_LOCAL_MODEL) or "deepseek-v4-flash"
        return PrivacyStatus(
            mode=MODE_SELF_HOSTED,
            private=True,
            base_url=url,
            model=mdl,
            source="env" if os.environ.get(ENV_LOCAL_BASE_URL) else "settings",
            note="Self-hosted local endpoint — zero data egress.",
        )

    # MODE_LOCAL_LIGHT
    url = base_url or os.environ.get(ENV_LOCAL_BASE_URL) or DEFAULT_LOCAL_BASE_URL
    mdl = model or os.environ.get(ENV_LOCAL_MODEL) or DEFAULT_LOCAL_MODEL
    return PrivacyStatus(
        mode=MODE_LOCAL_LIGHT,
        private=True,
        base_url=url,
        model=mdl,
        source="env" if os.environ.get(ENV_LOCAL_BASE_URL) else "settings",
        note="Lightweight local model (e.g. Ollama) — private but lower semantic quality.",
    )


def detect_local_server(base_url: str | None = None, timeout: float = 1.5) -> bool:
    """Probe whether a local OpenAI-compatible server is reachable.

    Used by the settings UI to tell the user whether local inference is
    actually available right now. Read-only; never sends audio.
    """
    import urllib.request

    url = base_url or DEFAULT_LOCAL_BASE_URL
    endpoint = url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(endpoint, timeout=timeout) as res:  # noqa: S310
            return res.status == 200
    except Exception:
        return False