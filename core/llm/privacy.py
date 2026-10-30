"""Local-First Privacy LLM Option.

Provides configuration and resolution for semantic analysis backends:
- Cloud endpoint (OpenAI-compatible) — OPT-IN ONLY, see below.
- Self-hosted inference server for zero data egress.
- Lightweight local model fallback (e.g. Ollama).

LOCAL-FIRST DEFAULT: when no mode is given, or an unknown mode is given,
resolution falls back to ``local-light`` — never to cloud. This matches
the product's local-first claim: no audio feature summary leaves the
machine unless the user explicitly opts into cloud.

CLOUD OPT-IN GATE: requesting ``cloud`` does NOT by itself activate
cloud. Cloud activates only when the caller passes
``explicit_consent=True`` to :func:`resolve_privacy` (e.g. after the user
confirmed a warning dialog) or sets the environment variable
``CUEPILOT_LLM_CLOUD_CONSENT`` to ``1``/``true``/``yes``. A cloud request
without that consent resolves to ``local-light`` instead, with
``requires_explicit_consent=True``, a ``warning`` explaining why, and
``requested_mode="cloud"`` so the UI can surface the choice instead of
silently sending data. Cloud mode is NOT removed — it stays available
as an explicit opt-in.

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
# Explicit cloud opt-in flag (see module docstring). Setting
# CUEPILOT_LLM_MODE=cloud alone is NOT enough to activate cloud.
ENV_CLOUD_CONSENT = "CUEPILOT_LLM_CLOUD_CONSENT"

# Local-first default mode (never cloud).
DEFAULT_MODE = MODE_LOCAL_LIGHT

CLOUD_CONSENT_WARNING = (
    "Cloud mode sends audio feature summaries to an external inference "
    "endpoint (data egress). It requires explicit user consent before it "
    "may be used: pass explicit_consent=True to resolve_privacy() after "
    "the user confirms, or set CUEPILOT_LLM_CLOUD_CONSENT=1."
)

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
    # True when the requested mode was cloud but explicit consent was
    # missing, so cloud was NOT activated (see module docstring). Callers
    # must surface `warning` to the user and re-resolve with consent.
    requires_explicit_consent: bool = False
    warning: str = ""
    # Mode that was requested before the consent gate ("" when unchanged).
    requested_mode: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "private": self.private,
            "baseUrl": self.base_url,
            "model": self.model,
            "source": self.source,
            "note": self.note,
            "requiresExplicitConsent": self.requires_explicit_consent,
            "warning": self.warning,
            "requestedMode": self.requested_mode,
        }


def _normalize_mode(mode: str | None) -> str:
    """Normalise a mode string; unknown/empty input -> local-first default."""
    if mode in MODES:
        return mode
    return DEFAULT_MODE


def _cloud_consent_given(explicit_consent: bool) -> bool:
    import os

    if explicit_consent:
        return True
    return os.environ.get(ENV_CLOUD_CONSENT, "").strip().lower() in ("1", "true", "yes")


def _local_light_status(source: str, note_suffix: str = "") -> PrivacyStatus:
    import os

    url = os.environ.get(ENV_LOCAL_BASE_URL) or DEFAULT_LOCAL_BASE_URL
    mdl = os.environ.get(ENV_LOCAL_MODEL) or DEFAULT_LOCAL_MODEL
    return PrivacyStatus(
        mode=MODE_LOCAL_LIGHT,
        private=True,
        base_url=url,
        model=mdl,
        source=source,
        note="Lightweight local model (e.g. Ollama) — private but lower semantic quality." + note_suffix,
    )


def resolve_privacy(
    mode: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    explicit_consent: bool = False,
) -> PrivacyStatus:
    """Resolve the effective privacy configuration.

    Order of precedence:
    1. Environment variables (never baked into the repo).
    2. Explicit arguments (from app settings).
    3. Defaults — local-first: no mode / unknown mode -> ``local-light``.

    Cloud is an explicit opt-in (see module docstring): a cloud request
    without ``explicit_consent`` (or the ``CUEPILOT_LLM_CLOUD_CONSENT``
    env flag) falls back to ``local-light`` with
    ``requires_explicit_consent=True`` instead of activating cloud.

    Returns a `PrivacyStatus` describing the mode, whether it is fully
    private, and which endpoint/model the semantic layer should use.
    """
    import os

    env_mode = os.environ.get(ENV_PRIVACY_MODE)
    resolved_mode = _normalize_mode(mode or env_mode)

    if resolved_mode == MODE_CLOUD and not _cloud_consent_given(explicit_consent):
        status = _local_light_status(
            source="env" if env_mode else "settings",
            note_suffix=" Cloud was requested but explicit consent was not given, "
            "so cloud was NOT activated (local-first fallback).",
        )
        status.requires_explicit_consent = True
        status.warning = CLOUD_CONSENT_WARNING
        status.requested_mode = MODE_CLOUD
        return status

    if resolved_mode == MODE_CLOUD:
        return PrivacyStatus(
            mode=MODE_CLOUD,
            private=False,
            base_url="",
            model="deepseek-v4-flash",
            source="env" if env_mode else "settings",
            note="Cloud inference endpoint. Audio feature summaries are sent for analysis. "
            "This mode was explicitly opted into (consent given).",
            requires_explicit_consent=False,
            warning=CLOUD_CONSENT_WARNING,
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

    # MODE_LOCAL_LIGHT (also the default when nothing was requested).
    url = base_url or os.environ.get(ENV_LOCAL_BASE_URL) or DEFAULT_LOCAL_BASE_URL
    mdl = model or os.environ.get(ENV_LOCAL_MODEL) or DEFAULT_LOCAL_MODEL
    return PrivacyStatus(
        mode=MODE_LOCAL_LIGHT,
        private=True,
        base_url=url,
        model=mdl,
        source="env"
        if os.environ.get(ENV_LOCAL_BASE_URL)
        else ("settings" if (mode or env_mode) else "default"),
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