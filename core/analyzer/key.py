"""Musical key detection from chromagram analysis.

Uses chroma features correlated against Krumhansl-Schmuckler key profiles,
then maps the estimated root/mode to a Camelot wheel code (e.g. "8A") for
harmonic mixing compatibility.
"""

from __future__ import annotations

import numpy as np

# Krumhansl-Schmuckler major profiles, in semitone order starting at C.
MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

# Camelot numbering: index = pitch class of the root (C=0 ... B=11).
# A-suffix = minor, B-suffix = major.
CAMELOT_CODE = {
    0: {"major": "8B", "minor": "8A"},
    1: {"major": "3B", "minor": "3A"},
    2: {"major": "10B", "minor": "10A"},
    3: {"major": "5B", "minor": "5A"},
    4: {"major": "12B", "minor": "12A"},
    5: {"major": "7B", "minor": "7A"},
    6: {"major": "2B", "minor": "2A"},
    7: {"major": "9B", "minor": "9A"},
    8: {"major": "4B", "minor": "4A"},
    9: {"major": "11B", "minor": "11A"},
    10: {"major": "6B", "minor": "6A"},
    11: {"major": "1B", "minor": "1A"},
}


def detect_key(chroma: np.ndarray) -> str:
    """Estimate musical key from a (12, n) chromagram.

    Returns a Camelot code string like "8A". Returns "8A" (a safe default) if
    the chromagram is empty or degenerate.
    """
    if chroma.size == 0 or chroma.shape[0] != 12:
        return "8A"
    chroma_mean = chroma.mean(axis=1)
    if np.allclose(chroma_mean, 0.0):
        return "8A"

    best_corr = -1.0
    best_root = 0
    best_mode = "minor"
    for shift in range(12):
        rolled = np.roll(chroma_mean, shift)
        for mode, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            corr = np.corrcoef(rolled, profile)[0, 1]
            if corr > best_corr:
                best_corr = corr
                best_root = (12 - shift) % 12
                best_mode = mode
    return CAMELOT_CODE[best_root][best_mode]
