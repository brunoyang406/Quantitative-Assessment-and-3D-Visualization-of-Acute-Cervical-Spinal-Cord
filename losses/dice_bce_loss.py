"""Binary segmentation: MONAI DiceLoss + BCE-with-logits, optional deep supervision."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss


def _target_b1dhw(target: torch.Tensor) -> torch.Tensor:
    if target.dim() == 4:
        target = target.unsqueeze(1)
    return target.float()


class DiceBCELoss(nn.Module):
    """Weighted sum of Dice and BCE-with-logits for single-channel logits vs binary mask."""

    def __init__(
        self,
        dice_weight: float = 0.7,
        ce_weight: float = 0.3,
        sigmoid: bool = True,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice = DiceLoss(sigmoid=sigmoid, reduction="mean")
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = _target_b1dhw(target)
        return self.dice_weight * self.dice(pred, target) + self.ce_weight * self.bce(pred, target)


class DiceBCEDeepSupervisionLoss(nn.Module):
    """Same as DiceBCELoss but `pred` is a list of logits at different resolutions (main first)."""

    def __init__(
        self,
        dice_weight: float = 0.7,
        ce_weight: float = 0.3,
        ds_weights: list | None = None,
        sigmoid: bool = True,
    ):
        super().__init__()
        self.ds_weights = ds_weights
        self.branch = DiceBCELoss(dice_weight=dice_weight, ce_weight=ce_weight, sigmoid=sigmoid)

    def forward(self, preds: list, target: torch.Tensor) -> torch.Tensor:
        target = _target_b1dhw(target)
        n = len(preds)
        if self.ds_weights is None:
            weights = [1.0 / (2**i) for i in range(n)]
        else:
            weights = list(self.ds_weights)
        while len(weights) < n:
            weights.append(weights[-1] * 0.5)
        weights = weights[:n]

        out = preds[0].new_zeros(())
        denom = 0.0
        for pred, w in zip(preds, weights):
            w = float(w)
            if pred.shape[2:] != target.shape[2:]:
                tgt = F.interpolate(target, size=pred.shape[2:], mode="nearest")
            else:
                tgt = target
            out = out + w * self.branch(pred, tgt)
            denom += w
        return out / denom
