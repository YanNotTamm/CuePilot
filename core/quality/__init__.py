"""Quality intelligence: audio QC and gig readiness scoring.

Non-destructive library-hygiene layer:
- `core.quality.qc` — Audio Quality / QC Report (45.7): clipping, loudness
  (integrated LUFS approximation), mono-compatibility, transcode/low-bitrate
  artifact, and edge-silence detection, all computed from the same analysis
  pass so no extra I/O is required.
- `core.quality.readiness` — Gig Readiness Score (45.6): a 0..100 score per
  track that mixes cue confidence, audio quality, structure completeness and
  correction history, exposed as a `Ready` / `Needs Review` / `Not Analyzed`
  badge for one-screen gig triage.
"""

from __future__ import annotations

from .qc import QcIssue, QcReport, compute_qc_report
from .readiness import GigReadiness, compute_readiness, readiness_bucket

__all__ = [
    "QcIssue",
    "QcReport",
    "compute_qc_report",
    "GigReadiness",
    "compute_readiness",
    "readiness_bucket",
]