"""
Models module for spinal cord lesion segmentation

This module provides:
- `MultimodalLesionUNet` — multimodal encoders, dual CBAM, lesion head (primary name)
- `MultiModalUNetWithDualCBAM` — same class, backward-compatible alias
- Factory: `create_model` / `get_model_info`
"""

from .multimodal_lesion_unet import MultimodalLesionUNet, MultiModalUNetWithDualCBAM
from .factory import create_model, get_model_info

__all__ = [
    'MultimodalLesionUNet',
    'MultiModalUNetWithDualCBAM',
    'create_model',
    'get_model_info',
]
