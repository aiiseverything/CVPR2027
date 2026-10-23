"""Evaluation row labels, extended here so common.py (scorer source) stays frozen."""
from common import MODEL_NAMES as _FROZEN_MODELS

# Insert the 9B scale reference directly after the 4B baseline.
MODEL_NAMES = ["baseline", "baseline-9b"] + [m for m in _FROZEN_MODELS if m != "baseline"]
