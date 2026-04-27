"""
Post-training inference utilities (not used by the default train loop).

- `inference.uncertainty`: MC Dropout, boundary maps, optional stats.
- CLI: `tools/generate_uncertainty_boundaries.py` loads a checkpoint, runs these helpers,
  writes NIfTI uncertainty maps for an optional extra model input (`uncertainty_boundary`).
"""

from .uncertainty import (
    compute_uncertainty_boundary,
    compute_uncertainty_statistics,
    disable_dropout,
    enable_dropout,
    predict_with_uncertainty,
    predict_with_uncertainty_batch,
)

__all__ = [
    "predict_with_uncertainty",
    "predict_with_uncertainty_batch",
    "compute_uncertainty_boundary",
    "compute_uncertainty_statistics",
    "enable_dropout",
    "disable_dropout",
]
