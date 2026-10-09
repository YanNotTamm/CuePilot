"""Structure detection package."""

from .detector import StructureDetector, detect_structure
from .router import (
    STRATEGIES,
    DetectionStrategy,
    confidence_cap,
    strategy_for_genre,
    strategy_for_profile,
)

__all__ = [
    "StructureDetector",
    "detect_structure",
    "STRATEGIES",
    "DetectionStrategy",
    "confidence_cap",
    "strategy_for_genre",
    "strategy_for_profile",
]
