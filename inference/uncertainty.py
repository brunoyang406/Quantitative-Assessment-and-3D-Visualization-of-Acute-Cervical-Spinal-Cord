"""
Monte Carlo Dropout inference and simple epistemic boundary helpers.

For binary lesion segmentation use predict_with_uncertainty(..., apply_sigmoid=True, apply_softmax=False).
See Kendall & Gal (NeurIPS 2017) for MC Dropout UQ context.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage


def predict_with_uncertainty(
    model: nn.Module,
    image: torch.Tensor,
    num_samples: int = 10,
    device: Optional[torch.device] = None,
    apply_softmax: bool = True,
    apply_sigmoid: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MC Dropout: model stays in train mode; returns mean prob, aleatoric E[p(1-p)], epistemic Var[p]."""
    if device is None:
        device = image.device
    model.train()
    predictions = []
    with torch.no_grad():
        for _ in range(num_samples):
            logits = model(image)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            if apply_sigmoid:
                probs = torch.sigmoid(logits)
            elif apply_softmax:
                probs = F.softmax(logits, dim=1)
            else:
                probs = logits
            predictions.append(probs.cpu())
    p_hat = torch.stack(predictions, dim=0)
    p_hat_np = p_hat.numpy()
    prediction = np.mean(p_hat_np, axis=0)
    aleatoric = np.mean(p_hat_np * (1 - p_hat_np), axis=0)
    epistemic = np.mean(p_hat_np**2, axis=0) - np.mean(p_hat_np, axis=0) ** 2
    return (
        torch.from_numpy(prediction).to(device),
        torch.from_numpy(aleatoric).to(device),
        torch.from_numpy(epistemic).to(device),
    )


def predict_with_uncertainty_batch(
    model: nn.Module,
    dataloader,
    num_samples: int = 10,
    device: Optional[torch.device] = None,
    apply_softmax: bool = True,
    apply_sigmoid: bool = False,
    lesion_class_idx: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run predict_with_uncertainty per batch; returns stacked numpy arrays (lesion_class_idx reserved)."""
    _ = lesion_class_idx
    model.train()
    all_p, all_a, all_e = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            prediction, aleatoric, epistemic = predict_with_uncertainty(
                model=model,
                image=images,
                num_samples=num_samples,
                device=device,
                apply_softmax=apply_softmax,
                apply_sigmoid=apply_sigmoid,
            )
            all_p.append(prediction.cpu().numpy())
            all_a.append(aleatoric.cpu().numpy())
            all_e.append(epistemic.cpu().numpy())
    return np.concatenate(all_p, axis=0), np.concatenate(all_a, axis=0), np.concatenate(all_e, axis=0)


def compute_uncertainty_statistics(
    aleatoric: np.ndarray,
    epistemic: np.ndarray,
    lesion_class_idx: int = 1,
) -> dict:
    """Scalar stats for lesion channel (index 1) or flat maps."""
    la = aleatoric[:, lesion_class_idx, ...] if len(aleatoric.shape) > 2 else aleatoric
    le = epistemic[:, lesion_class_idx, ...] if len(epistemic.shape) > 2 else epistemic
    total = la + le
    return {
        "aleatoric_mean": float(np.mean(la)),
        "aleatoric_std": float(np.std(la)),
        "aleatoric_min": float(np.min(la)),
        "aleatoric_max": float(np.max(la)),
        "epistemic_mean": float(np.mean(le)),
        "epistemic_std": float(np.std(le)),
        "epistemic_min": float(np.min(le)),
        "epistemic_max": float(np.max(le)),
        "total_uncertainty_mean": float(np.mean(total)),
        "total_uncertainty_std": float(np.std(total)),
    }


def enable_dropout(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()


def disable_dropout(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.eval()


def compute_uncertainty_boundary(
    prediction: np.ndarray,
    epistemic: np.ndarray,
    threshold: float = 0.5,
    uncertainty_percentile: float = 75,
) -> dict:
    """Epistemic percentile mask intersected with prediction boundary (morphological)."""
    pred_binary = (prediction > threshold).astype(np.uint8)
    uncertainty_threshold = np.percentile(epistemic, uncertainty_percentile)
    high_uncertainty = (epistemic > uncertainty_threshold).astype(np.uint8)
    pred_boundary = pred_binary - ndimage.binary_erosion(pred_binary).astype(np.uint8)
    uncertainty_boundary = high_uncertainty * pred_boundary
    return {
        "pred_binary": pred_binary,
        "high_uncertainty": high_uncertainty,
        "pred_boundary": pred_boundary,
        "uncertainty_boundary": uncertainty_boundary,
        "uncertainty_threshold": uncertainty_threshold,
        "boundary_pixels": np.sum(uncertainty_boundary),
        "boundary_percentage": np.sum(uncertainty_boundary) / np.sum(pred_boundary)
        if np.sum(pred_boundary) > 0
        else 0,
    }
