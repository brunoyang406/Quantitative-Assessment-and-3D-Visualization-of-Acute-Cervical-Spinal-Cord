"""
Spinal cord multimodal lesion segmentation.

Main pieces: `models`, `data`, `trainers`, `losses`, `utils`, `inference` (optional UQ).
"""

__version__ = "1.0.0"
__author__ = "Spinal Cord Lesion Segmentation Team"

from .models import create_model, get_model_info
from .trainers import MultimodalLesionUnetIgniteTrainer
from .data import get_multimodal_lesion_unet_transforms, create_data_loaders
from .losses import DiceBCELoss, DiceBCEDeepSupervisionLoss
from .inference import (
    compute_uncertainty_boundary,
    predict_with_uncertainty,
)
from .utils import (
    load_config,
    parse_arguments,
    setup_experiment_directory,
    set_random_seed,
)

__all__ = [
    "create_model",
    "get_model_info",
    "MultimodalLesionUnetIgniteTrainer",
    "get_multimodal_lesion_unet_transforms",
    "create_data_loaders",
    "DiceBCELoss",
    "DiceBCEDeepSupervisionLoss",
    "predict_with_uncertainty",
    "compute_uncertainty_boundary",
    "load_config",
    "parse_arguments",
    "setup_experiment_directory",
    "set_random_seed",
]
