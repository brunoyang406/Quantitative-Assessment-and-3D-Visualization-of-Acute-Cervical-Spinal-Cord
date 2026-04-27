"""Losses for lesion segmentation."""

from .dice_bce_loss import DiceBCELoss, DiceBCEDeepSupervisionLoss

__all__ = [
    "DiceBCELoss",
    "DiceBCEDeepSupervisionLoss",
]
