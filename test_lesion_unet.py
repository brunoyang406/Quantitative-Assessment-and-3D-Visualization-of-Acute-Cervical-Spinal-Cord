"""
Evaluate a trained multimodal lesion U-Net on the test or val split.

Same preprocessing as validation, optional NIfTI export with inverse ROI/spacing/orientation.
"""

import os
import sys
import argparse
import logging
from typing import Dict, Any, Optional
from pathlib import Path

import yaml
import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader
from monai.data import Dataset, decollate_batch
from monai.transforms import AsDiscrete
from tqdm import tqdm
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_dataset(config: Dict[str, Any], dataset_type: str = 'test'):
    """Build DataLoader for ``test`` or ``val`` using val transforms (no augmentation)."""
    from data.multimodal_transforms import get_multimodal_lesion_unet_transforms
    from data.dataloader import load_data_list, prepare_data_dicts
    
    _, val_transforms = get_multimodal_lesion_unet_transforms(config)
    
    if dataset_type == 'val':
        data_json = config.get('val_json', config.get('data', {}).get('val_json', './spinal_cord_dataset/val.json'))
        dataset_name = 'validation'
    elif dataset_type == 'test':
        data_json = config.get('test_json', config.get('data', {}).get('test_json', './spinal_cord_dataset/test.json'))
        dataset_name = 'test'
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}. Must be 'test' or 'val'.")
    
    data_dir = config.get('data_dir', config.get('data', {}).get('data_dir', './spinal_cord_dataset'))
    
    logging.info(f"Loading {dataset_name} dataset from: {data_json}")
    
    data_list = load_data_list(data_json)
    data_dicts = prepare_data_dicts(data_list, data_dir=data_dir)
    
    logging.info(f"Found {len(data_dicts)} {dataset_name} samples")
    
    dataset = Dataset(data=data_dicts, transform=val_transforms)
    
    val_batch_size = config.get('training', {}).get('val_batch_size', 1)
    val_num_workers = config.get('dataloader', {}).get('val_num_workers', 1)
    
    data_loader = DataLoader(
        dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=val_num_workers,
        pin_memory=False,
        persistent_workers=False,
    )
    
    logging.info(f"{dataset_name.capitalize()} DataLoader created: batch_size={val_batch_size}, num_workers={val_num_workers}")
    
    return data_loader, data_dicts


def load_test_dataset(config: Dict[str, Any]):
    """Backward-compatible alias for ``load_dataset(..., 'test')``."""
    return load_dataset(config, dataset_type='test')


def load_model(config: Dict[str, Any], checkpoint_path: str, device: torch.device) -> nn.Module:
    """Instantiate model via factory and load checkpoint weights."""
    from models import create_model
    
    model_config = config.get('model', {})
    spatial_size = config.get('spatial_size', [16, 512, 256])
    
    model = create_model(
        model_config=model_config,
        img_size=tuple(spatial_size)
    )
    
    logging.info(f"Loading model weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    new_state_dict = {}
    has_module_prefix = False
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
            has_module_prefix = True
        else:
            new_state_dict[k] = v
    
    if has_module_prefix:
        logging.info("  Stripped 'module.' prefix from checkpoint keys")
    
    model.load_state_dict(new_state_dict)
    logging.info("  Model weights loaded")
    
    model.to(device)
    model.eval()
    
    logging.info("Model loaded successfully")
    
    return model


def get_original_image_metadata(data_dict: Dict[str, Any]) -> tuple:
    """Read affine/header/shape/spacing from subject T1 NIfTI."""
    t1_path = data_dict['T1']
    if not os.path.exists(t1_path):
        raise FileNotFoundError(f"T1 image not found: {t1_path}")
    
    t1_nii = nib.load(t1_path)
    affine = t1_nii.affine
    header = t1_nii.header.copy()
    original_shape = t1_nii.shape
    original_spacing = header.get_zooms()[:3]
    
    return affine, header, original_shape, original_spacing, t1_nii


def load_and_preprocess_cord_mask(
    data_dict: Dict[str, Any],
    config: Dict[str, Any]
) -> tuple:
    """Load cord_mask with same RAS + spacing as training."""
    from monai.transforms import LoadImage, EnsureChannelFirst, Orientation, Spacing
    
    cord_path = data_dict.get('cord_mask')
    if not cord_path or not os.path.exists(cord_path):
        raise FileNotFoundError(f"Cord mask not found: {cord_path}")
    
    loader = LoadImage()
    ensure_channel = EnsureChannelFirst()
    orientation = Orientation(axcodes="RAS")
    
    target_spacing_config = config.get('target_spacing', [3.3, 0.54, 0.54])  # [X, Y, Z]
    from data.multimodal_transforms import _reorder_spacing_for_ras
    pixdim = _reorder_spacing_for_ras(target_spacing_config)  # [X, Z, Y] = [D, H, W]
    
    spacing = Spacing(pixdim=pixdim, mode="nearest")
    
    cord_data = loader(cord_path)
    cord_data = ensure_channel(cord_data)
    cord_data = orientation(cord_data)
    cord_data = spacing(cord_data)
    
    if isinstance(cord_data, torch.Tensor):
        cord_data = cord_data.numpy()
    
    if cord_data.ndim == 4 and cord_data.shape[0] == 1:
        cord_mask = cord_data[0]  # (D, H, W)
    elif cord_data.ndim == 3:
        cord_mask = cord_data
    else:
        raise ValueError(f"Unexpected cord_mask shape: {cord_data.shape}")
    
    preprocessed_shape = cord_mask.shape
    
    return cord_mask, preprocessed_shape


def compute_roi_crop_params(
    cord_mask: np.ndarray,
    target_size: tuple,
    margin: tuple
) -> tuple:
    """ROI crop/pad indices aligned with cord-based cropping in training."""
    target_size = np.array(target_size)
    margin = np.array(margin)
    
    coords = np.where(cord_mask > 0)
    
    if len(coords[0]) == 0:
        img_shape = np.array(cord_mask.shape)
        return np.array([0, 0, 0]), img_shape, np.array([0, 0, 0]), np.array([0, 0, 0])
    
    min_coords = np.array([coords[0].min(), coords[1].min(), coords[2].min()])
    max_coords = np.array([coords[0].max(), coords[1].max(), coords[2].max()])
    img_shape = np.array(cord_mask.shape)
    
    crop_start = np.zeros(3, dtype=int)
    crop_end = np.zeros(3, dtype=int)
    pad_before = np.zeros(3, dtype=int)
    pad_after = np.zeros(3, dtype=int)
    
    for i in range(3):
        roi_start = max(0, min_coords[i] - margin[i])
        roi_end = min(img_shape[i], max_coords[i] + margin[i] + 1)
        roi_actual_size = roi_end - roi_start
        
        if roi_actual_size >= target_size[i]:
            center = (roi_start + roi_end) // 2
            crop_start[i] = max(0, center - target_size[i] // 2)
            crop_end[i] = min(img_shape[i], crop_start[i] + target_size[i])
            
            if crop_end[i] - crop_start[i] < target_size[i]:
                crop_start[i] = max(0, crop_end[i] - target_size[i])
        else:
            crop_start[i] = roi_start
            crop_end[i] = roi_end
            
            total_pad = target_size[i] - roi_actual_size
            pad_before[i] = total_pad // 2
            pad_after[i] = total_pad - pad_before[i]
    
    return crop_start, crop_end, pad_before, pad_after


def inverse_roi_crop(
    data: np.ndarray,
    original_shape: tuple,
    crop_start: np.ndarray,
    crop_end: np.ndarray,
    pad_before: np.ndarray,
    pad_after: np.ndarray
) -> np.ndarray:
    """Undo ROI crop: strip pad, paste into full preprocessed shape."""
    unpadded = data[
        pad_before[0]:data.shape[0] - pad_after[0] if pad_after[0] > 0 else data.shape[0],
        pad_before[1]:data.shape[1] - pad_after[1] if pad_after[1] > 0 else data.shape[1],
        pad_before[2]:data.shape[2] - pad_after[2] if pad_after[2] > 0 else data.shape[2]
    ]
    
    full_image = np.zeros(original_shape, dtype=data.dtype)
    full_image[
        crop_start[0]:crop_end[0],
        crop_start[1]:crop_end[1],
        crop_start[2]:crop_end[2]
    ] = unpadded
    
    return full_image


def resize_to_target_shape(
    data: np.ndarray,
    target_shape: tuple,
    is_label: bool = False
) -> np.ndarray:
    """Center-crop or center-pad to ``target_shape``."""
    current_shape = np.array(data.shape)
    target_shape = np.array(target_shape)
    
    if is_label:
        output = np.zeros(target_shape, dtype=data.dtype)
    else:
        output = np.zeros(target_shape, dtype=np.float32)
    
    slices_in = []
    slices_out = []
    
    for i in range(3):
        if current_shape[i] > target_shape[i]:
            start = (current_shape[i] - target_shape[i]) // 2
            slices_in.append(slice(start, start + target_shape[i]))
            slices_out.append(slice(0, target_shape[i]))
        else:
            start = (target_shape[i] - current_shape[i]) // 2
            slices_in.append(slice(0, current_shape[i]))
            slices_out.append(slice(start, start + current_shape[i]))
    
    output[slices_out[0], slices_out[1], slices_out[2]] = \
        data[slices_in[0], slices_in[1], slices_in[2]]
    
    return output


def resample_to_original_space(
    data: np.ndarray,
    original_nii: nib.Nifti1Image,
    current_spacing: tuple,
    is_label: bool = False
) -> np.ndarray:
    """Reorient RAS grid to original NIfTI axes, then resample to native voxel grid."""
    try:
        original_ornt = nib.io_orientation(original_nii.affine)
        
        ras_ornt = np.array([[0, 1], [1, 1], [2, 1]])  # R=x+, A=y+, S=z+
        
        ornt_transform = nib.orientations.ornt_transform(ras_ornt, original_ornt)
        
        data_reoriented = nib.orientations.apply_orientation(data, ornt_transform)
        
        logging.debug(
            "    Reorient: RAS shape=%s -> original-axes shape=%s",
            data.shape,
            data_reoriented.shape,
        )
    except Exception as e:
        logging.warning("    Reorientation failed, skipping: %s", e)
        data_reoriented = data
    
    target_shape = original_nii.shape
    current_shape = np.array(data_reoriented.shape)
    target_shape_arr = np.array(target_shape)
    
    zoom_factors = target_shape_arr / current_shape
    
    logging.debug(
        "    Resample: %s -> %s, zoom=%s",
        current_shape,
        target_shape,
        zoom_factors,
    )
    
    if is_label:
        resampled = ndimage.zoom(data_reoriented, zoom_factors, order=0, mode='nearest')
    else:
        resampled = ndimage.zoom(data_reoriented, zoom_factors, order=1, mode='nearest')
    
    if resampled.shape != tuple(target_shape):
        resampled = resize_to_target_shape(resampled, target_shape, is_label=is_label)
    
    return resampled


def compute_surface_distances(pred: np.ndarray, gt: np.ndarray, spacing: tuple) -> tuple:
    """Symmetric surface-to-surface distances (mm) using 6-neighborhood boundaries."""
    structure = ndimage.generate_binary_structure(3, 1)
    
    pred_border = pred.astype(bool) ^ ndimage.binary_erosion(pred.astype(bool), structure=structure)
    gt_border = gt.astype(bool) ^ ndimage.binary_erosion(gt.astype(bool), structure=structure)
    
    pred_coords = np.argwhere(pred_border)
    gt_coords = np.argwhere(gt_border)
    
    if len(pred_coords) == 0 or len(gt_coords) == 0:
        return np.array([]), np.array([])
    
    pred_coords_scaled = pred_coords * np.array(spacing)
    gt_coords_scaled = gt_coords * np.array(spacing)
    
    distances_pred_to_gt = []
    for pred_point in pred_coords_scaled:
        distances = np.sqrt(np.sum((gt_coords_scaled - pred_point) ** 2, axis=1))
        distances_pred_to_gt.append(np.min(distances))
    
    distances_gt_to_pred = []
    for gt_point in gt_coords_scaled:
        distances = np.sqrt(np.sum((pred_coords_scaled - gt_point) ** 2, axis=1))
        distances_gt_to_pred.append(np.min(distances))
    
    return np.array(distances_pred_to_gt), np.array(distances_gt_to_pred)


def compute_metrics_3d(pred: np.ndarray, gt: np.ndarray, spacing: tuple = (1.0, 1.0, 1.0)) -> Dict[str, float]:
    """Overlap, classification, boundary (HD95, ASSD), and volume metrics."""
    metrics = {}
    
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()
    
    pred = (pred > 0).astype(np.float32)
    gt = (gt > 0).astype(np.float32)
    
    tp = np.sum(pred * gt)  # True Positive
    fp = np.sum(pred * (1 - gt))  # False Positive
    fn = np.sum((1 - pred) * gt)  # False Negative
    tn = np.sum((1 - pred) * (1 - gt))  # True Negative
    
    # Dice Score
    intersection = tp
    union = tp + tp + fp + fn
    dice = 2.0 * intersection / (union + 1e-8) if union > 0 else 0.0
    metrics['dice'] = dice
    
    # IoU / Jaccard
    iou = intersection / (tp + fp + fn + 1e-8) if (tp + fp + fn) > 0 else 0.0
    metrics['iou'] = iou
    
    # Precision (PPV)
    precision = tp / (tp + fp + 1e-8) if (tp + fp) > 0 else 0.0
    metrics['precision'] = precision
    
    # Recall / Sensitivity
    recall = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else 0.0
    metrics['recall'] = recall
    metrics['sensitivity'] = recall
    
    # Specificity
    specificity = tn / (tn + fp + 1e-8) if (tn + fp) > 0 else 0.0
    metrics['specificity'] = specificity
    
    # Relative Volume Difference (RVD)
    pred_volume = np.sum(pred)
    gt_volume = np.sum(gt)
    rvd = (pred_volume - gt_volume) / (gt_volume + 1e-8) if gt_volume > 0 else 0.0
    metrics['rvd'] = rvd
    metrics['volume_error_percent'] = abs(rvd) * 100
    
    if np.sum(pred) > 0 and np.sum(gt) > 0:
        try:
            distances_pred_to_gt, distances_gt_to_pred = compute_surface_distances(pred, gt, spacing)
            
            if len(distances_pred_to_gt) > 0 and len(distances_gt_to_pred) > 0:
                # 95% Hausdorff Distance (95HD)
                hd95_pred_to_gt = np.percentile(distances_pred_to_gt, 95)
                hd95_gt_to_pred = np.percentile(distances_gt_to_pred, 95)
                hd95 = max(hd95_pred_to_gt, hd95_gt_to_pred)
                metrics['hd95'] = hd95
                
                # Average Symmetric Surface Distance (ASSD / ASD)
                avg_surf_dist_pred_to_gt = np.mean(distances_pred_to_gt)
                avg_surf_dist_gt_to_pred = np.mean(distances_gt_to_pred)
                assd = (avg_surf_dist_pred_to_gt + avg_surf_dist_gt_to_pred) / 2.0
                metrics['assd'] = assd
                metrics['asd'] = assd
                
                tolerance_1mm = 1.0
                tolerance_2mm = 2.0
                
                surf_overlap_1mm = np.sum(distances_pred_to_gt <= tolerance_1mm) + np.sum(distances_gt_to_pred <= tolerance_1mm)
                surf_total = len(distances_pred_to_gt) + len(distances_gt_to_pred)
                surface_dice_1mm = surf_overlap_1mm / (surf_total + 1e-8) if surf_total > 0 else 0.0
                metrics['surface_dice_1mm'] = surface_dice_1mm
                
                surf_overlap_2mm = np.sum(distances_pred_to_gt <= tolerance_2mm) + np.sum(distances_gt_to_pred <= tolerance_2mm)
                surface_dice_2mm = surf_overlap_2mm / (surf_total + 1e-8) if surf_total > 0 else 0.0
                metrics['surface_dice_2mm'] = surface_dice_2mm
            else:
                metrics['hd95'] = 0.0
                metrics['assd'] = 0.0
                metrics['asd'] = 0.0
                metrics['surface_dice_1mm'] = 0.0
                metrics['surface_dice_2mm'] = 0.0
        except Exception as e:
            logging.warning("Surface-distance metrics failed: %s", e)
            metrics['hd95'] = 0.0
            metrics['assd'] = 0.0
            metrics['asd'] = 0.0
            metrics['surface_dice_1mm'] = 0.0
            metrics['surface_dice_2mm'] = 0.0
    else:
        metrics['hd95'] = 0.0 if np.sum(pred) == 0 and np.sum(gt) == 0 else float('inf')
        metrics['assd'] = 0.0 if np.sum(pred) == 0 and np.sum(gt) == 0 else float('inf')
        metrics['asd'] = metrics['assd']
        metrics['surface_dice_1mm'] = 0.0
        metrics['surface_dice_2mm'] = 0.0
    
    return metrics


def evaluate_on_test(
    config: Dict[str, Any],
    model: nn.Module,
    test_loader: DataLoader,
    test_data_dicts: list,
    device: torch.device,
    use_deep_supervision: bool = True,
    save_predictions: bool = False,
    output_dir: Optional[str] = None,
) -> Dict[str, float]:
    """Run eval loop, aggregate per-sample metrics and global Dice."""
    model.eval()
    
    post_pred_lesion = AsDiscrete(threshold=0.5)
    
    target_spacing_config = config.get('target_spacing', [3.3, 0.54, 0.54])  # [X, Y, Z]
    spacing = (target_spacing_config[0], target_spacing_config[2], target_spacing_config[1])
    logging.info(f"Using spacing for surface distance metrics: {spacing} mm (D, H, W)")
    
    dice_intersection = 0.0
    dice_union = 0.0
    
    metric_names = ['dice', 'iou', 'precision', 'recall', 'sensitivity', 'specificity', 
                    'hd95', 'assd', 'asd', 'surface_dice_1mm', 'surface_dice_2mm', 
                    'rvd', 'volume_error_percent']
    accumulated_metrics = {name: [] for name in metric_names}
    
    per_sample_results = []
    
    logging.info("\nEvaluating on test dataset...")
    logging.info("="*70)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            inputs = batch["image"].to(device)
            lesion_labels = batch["lesion_mask"].to(device)
            
            lesion_output = model(inputs)
            
            if use_deep_supervision:
                lesion_logits = lesion_output[0]
            else:
                lesion_logits = lesion_output
            
            lesion_probs = torch.sigmoid(lesion_logits)
            lesion_preds = post_pred_lesion(lesion_probs)
            
            if lesion_labels.dim() == 4:
                lesion_labels = lesion_labels.unsqueeze(1)

            batch_subject_ids = batch.get("subject_id", None)
            if batch_subject_ids is None:
                batch_subject_ids = [f"sample_{batch_idx}_{i}" for i in range(inputs.shape[0])]

            for b in range(inputs.shape[0]):
                subject_id = batch_subject_ids[b]
                pred_b = lesion_preds[b]
                label_b = lesion_labels[b]

                pred_np = pred_b[0].cpu().numpy() if pred_b.dim() == 4 else pred_b.cpu().numpy()
                label_np = label_b[0].cpu().numpy() if label_b.dim() == 4 else label_b.cpu().numpy()
                
                sample_metrics = compute_metrics_3d(pred_np, label_np, spacing=spacing)
                
                intersection = (pred_b * label_b).sum().item()
                union = pred_b.sum().item() + label_b.sum().item()
                dice_intersection += intersection
                dice_union += union
                
                for metric_name in metric_names:
                    if metric_name in sample_metrics:
                        accumulated_metrics[metric_name].append(sample_metrics[metric_name])
                
                sample_result = {
                    'subject_id': subject_id,
                    'intersection': intersection,
                    'union': union,
                }
                sample_result.update(sample_metrics)
                per_sample_results.append(sample_result)

                if save_predictions and output_dir:
                    data_dict_idx = batch_idx * inputs.shape[0] + b
                    if data_dict_idx < len(test_data_dicts):
                        data_dict = test_data_dicts[data_dict_idx]
                    else:
                        logging.warning("data_dict index out of range, skip inverse transform: %s", data_dict_idx)
                        data_dict = None
                    
                    save_prediction(
                        pred_b,
                        subject_id,
                        os.path.join(output_dir, "predictions_baseline"),
                        config=config,
                        data_dict=data_dict
                    )
    
    if dice_union > 0:
        overall_dice = 2.0 * dice_intersection / (dice_union + 1e-8)
    else:
        overall_dice = 0.0
    
    metric_stats = {}
    for metric_name in metric_names:
        values = accumulated_metrics[metric_name]
        if len(values) > 0:
            valid_values = [v for v in values if not np.isinf(v)]
            if len(valid_values) > 0:
                metric_stats[metric_name] = {
                    'mean': float(np.mean(valid_values)),
                    'std': float(np.std(valid_values)),
                    'median': float(np.median(valid_values)),
                    'min': float(np.min(valid_values)),
                    'max': float(np.max(valid_values)),
                }
            else:
                metric_stats[metric_name] = {
                    'mean': 0.0, 'std': 0.0, 'median': 0.0, 'min': 0.0, 'max': 0.0
                }
        else:
            metric_stats[metric_name] = {
                'mean': 0.0, 'std': 0.0, 'median': 0.0, 'min': 0.0, 'max': 0.0
            }
    
    logging.info("")
    logging.info("="*70)
    logging.info("Test Results")
    logging.info("="*70)
    logging.info("")
    logging.info(f"Total test samples: {len(per_sample_results)}")
    logging.info("")
    logging.info("Overall Metrics (mean±std | median):")
    logging.info("-"*70)
    
    logging.info("Region Overlap Metrics:")
    logging.info(f"  Dice:        {metric_stats['dice']['mean']:.4f}±{metric_stats['dice']['std']:.4f} | {metric_stats['dice']['median']:.4f}")
    logging.info(f"  IoU:         {metric_stats['iou']['mean']:.4f}±{metric_stats['iou']['std']:.4f} | {metric_stats['iou']['median']:.4f}")
    logging.info("")
    
    logging.info("Pixel-wise Classification Metrics:")
    logging.info(f"  Precision:   {metric_stats['precision']['mean']:.4f}±{metric_stats['precision']['std']:.4f} | {metric_stats['precision']['median']:.4f}")
    logging.info(f"  Recall:      {metric_stats['recall']['mean']:.4f}±{metric_stats['recall']['std']:.4f} | {metric_stats['recall']['median']:.4f}")
    logging.info(f"  Specificity: {metric_stats['specificity']['mean']:.4f}±{metric_stats['specificity']['std']:.4f} | {metric_stats['specificity']['median']:.4f}")
    logging.info("")
    
    logging.info("Surface Distance Metrics (mm):")
    logging.info(f"  95HD:        {metric_stats['hd95']['mean']:.2f}±{metric_stats['hd95']['std']:.2f} | {metric_stats['hd95']['median']:.2f}")
    logging.info(f"  ASSD:        {metric_stats['assd']['mean']:.2f}±{metric_stats['assd']['std']:.2f} | {metric_stats['assd']['median']:.2f}")
    logging.info("")
    
    # Surface Dice
    logging.info("Surface Dice (tolerance):")
    logging.info(f"  @1mm:        {metric_stats['surface_dice_1mm']['mean']:.4f}±{metric_stats['surface_dice_1mm']['std']:.4f} | {metric_stats['surface_dice_1mm']['median']:.4f}")
    logging.info(f"  @2mm:        {metric_stats['surface_dice_2mm']['mean']:.4f}±{metric_stats['surface_dice_2mm']['std']:.4f} | {metric_stats['surface_dice_2mm']['median']:.4f}")
    logging.info("")
    
    logging.info("Volume Metrics:")
    logging.info(f"  RVD:         {metric_stats['rvd']['mean']:.4f}±{metric_stats['rvd']['std']:.4f} | {metric_stats['rvd']['median']:.4f}")
    logging.info(f"  Vol Error %: {metric_stats['volume_error_percent']['mean']:.2f}±{metric_stats['volume_error_percent']['std']:.2f} | {metric_stats['volume_error_percent']['median']:.2f}")
    
    logging.info("")
    logging.info(f"Overall Dice (global): {overall_dice:.4f}")
    logging.info("="*70)
    logging.info("")
    
    logging.info("Per-sample results (Dice | IoU | 95HD | ASSD):")
    logging.info("-"*70)
    for result in per_sample_results:
        logging.info(f"  {result['subject_id']}: Dice={result['dice']:.4f} | IoU={result['iou']:.4f} | 95HD={result['hd95']:.2f}mm | ASSD={result['assd']:.2f}mm")
    logging.info("="*70)
    logging.info("")
    
    return {
        'overall_dice': overall_dice,
        'metric_stats': metric_stats,
        'per_sample_results': per_sample_results,
        'dice_intersection': dice_intersection,
        'dice_union': dice_union,
    }


def save_prediction(
    pred: torch.Tensor, 
    subject_id: str, 
    output_dir: str,
    config: Dict[str, Any] = None,
    data_dict: Dict[str, Any] = None
):
    """Write prediction NIfTI; optionally map back to native space via config/data_dict."""
    os.makedirs(output_dir, exist_ok=True)
    
    pred_np = pred.cpu().numpy()
    
    if pred_np.shape[0] == 1:
        pred_np = pred_np[0]
    
    if config is not None and data_dict is not None:
        try:
            affine, header, original_shape, original_spacing, original_nii = get_original_image_metadata(data_dict)
            
            logging.info("  Subject %s:", subject_id)
            logging.info("    Model output (after ROI crop): shape=%s", pred_np.shape)
            logging.info("    Native image space: shape=%s spacing=%s", original_shape, original_spacing)
            
            roi_config = config.get('roi_crop', {})
            use_roi_crop = roi_config.get('use_roi_crop', False)
            
            if use_roi_crop:
                try:
                    cord_mask_processed, preprocessed_shape = load_and_preprocess_cord_mask(data_dict, config)
                    
                    roi_target_size = tuple(roi_config.get('roi_target_size', config.get('spatial_size', [16, 512, 256])))
                    roi_margin = tuple(roi_config.get('roi_margin', (1, 27, 27)))
                    
                    crop_start, crop_end, pad_before, pad_after = compute_roi_crop_params(
                        cord_mask=cord_mask_processed,
                        target_size=roi_target_size,
                        margin=roi_margin
                    )
                    
                    logging.info("    Inverse ROI crop...")
                    pred_full = inverse_roi_crop(
                        data=pred_np,
                        original_shape=preprocessed_shape,
                        crop_start=crop_start,
                        crop_end=crop_end,
                        pad_before=pad_before,
                        pad_after=pad_after
                    )
                    logging.info("      After inverse ROI crop: shape=%s", pred_full.shape)
                except Exception as e:
                    logging.warning("    Inverse ROI crop failed, using crop-space pred: %s", e)
                    pred_full = pred_np
            else:
                pred_full = pred_np
            
            logging.info("    Resample to native spacing + orientation...")
            
            target_spacing_config = config.get('target_spacing', [3.3, 0.54, 0.54])  # [X, Y, Z]
            current_spacing = (target_spacing_config[0], target_spacing_config[2], target_spacing_config[1])
            
            pred_original = resample_to_original_space(
                data=pred_full,
                original_nii=original_nii,
                current_spacing=current_spacing,
                is_label=True,
            )
            
            logging.info("    Final native shape: %s", pred_original.shape)
            
            pred_np = pred_original.astype(np.uint8)
            
            output_path = os.path.join(output_dir, f"{subject_id}_pred.nii.gz")
            nifti_img = nib.Nifti1Image(pred_np, affine=affine, header=header)
            nib.save(nifti_img, output_path)
            
        except Exception as e:
            logging.warning("    Inverse transform failed, saving crop-space volume: %s", e)
            import traceback
            logging.debug(traceback.format_exc())
            pred_np = (pred_np * 255).astype(np.uint8)
            output_path = os.path.join(output_dir, f"{subject_id}_pred.nii.gz")
            nifti_img = nib.Nifti1Image(pred_np, affine=np.eye(4))
            nib.save(nifti_img, output_path)
    else:
        pred_np = (pred_np * 255).astype(np.uint8)
        output_path = os.path.join(output_dir, f"{subject_id}_pred.nii.gz")
        nifti_img = nib.Nifti1Image(pred_np, affine=np.eye(4))
        nib.save(nifti_img, output_path)


def save_results(results: Dict[str, Any], output_path: str):
    """Serialize aggregate metrics and per-sample scores to JSON."""
    import json
    
    save_data = {
        'overall_dice': float(results['overall_dice']),
        'dice_intersection': float(results['dice_intersection']),
        'dice_union': float(results['dice_union']),
        'num_samples': len(results['per_sample_results']),
        'metric_stats': results.get('metric_stats', {}),
        'per_sample_results': []
    }
    
    for sample_result in results['per_sample_results']:
        sample_data = {
            'subject_id': sample_result['subject_id'],
            'intersection': float(sample_result['intersection']),
            'union': float(sample_result['union']),
        }
        
        for key, value in sample_result.items():
            if key not in ['subject_id', 'intersection', 'union']:
                if isinstance(value, (int, float, np.number)):
                    sample_data[key] = float(value) if not np.isinf(value) else None
        
        save_data['per_sample_results'].append(sample_data)
    
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    logging.info(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate multimodal lesion U-Net on test or val split")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file (e.g., configs/multimodal_lesion_unet.yaml)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (e.g., experiments/.../weights/best_model_checkpoint_lesion_dice=0.5361.pt)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for saving results and predictions (default: same as checkpoint dir)'
    )
    parser.add_argument(
        '--save_predictions',
        action='store_true',
        help='Save prediction masks as nifti files'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use (default: cuda)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='test',
        choices=['test', 'val'],
        help='Dataset to evaluate on: test or val (default: test)'
    )
    parser.add_argument(
        '--test_json',
        type=str,
        default=None,
        help='Path to test JSON file (overrides config, e.g., ./spinal_cord_dataset/test_high_dice.json)'
    )
    parser.add_argument(
        '--val_json',
        type=str,
        default=None,
        help='Path to validation JSON file (overrides config)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
    
    if args.output_dir is None:
        checkpoint_dir = os.path.dirname(os.path.dirname(args.checkpoint))
        args.output_dir = os.path.join(checkpoint_dir, f'{args.dataset}_results')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    logging.info("\n[1/4] Loading configuration...")
    from utils.config import merge_config_with_args
    raw_config = load_config(args.config)
    config = merge_config_with_args(raw_config, args)
    
    if args.test_json is not None:
        config['test_json'] = args.test_json
        logging.info(f"Using custom test JSON: {args.test_json}")
    if args.val_json is not None:
        config['val_json'] = args.val_json
        logging.info(f"Using custom val JSON: {args.val_json}")
    
    dataset_name = 'validation' if args.dataset == 'val' else 'test'
    logging.info(f"\n[2/4] Loading {dataset_name} dataset...")
    data_loader, data_dicts = load_dataset(config, dataset_type=args.dataset)
    
    logging.info("\n[3/4] Loading model...")
    model = load_model(config, args.checkpoint, device)
    
    use_deep_supervision = config.get('model', {}).get('deep_supervision', True)
    
    logging.info(f"\n[4/4] Evaluating on {dataset_name} dataset...")
    results = evaluate_on_test(
        config=config,
        model=model,
        test_loader=data_loader,
        test_data_dicts=data_dicts,
        device=device,
        use_deep_supervision=use_deep_supervision,
        save_predictions=args.save_predictions,
        output_dir=args.output_dir if args.save_predictions else None,
    )
    
    results_path = os.path.join(args.output_dir, f'{args.dataset}_results.json')
    save_results(results, results_path)
    
    logging.info("\n" + "="*70)
    logging.info(f"{dataset_name.capitalize()} evaluation completed successfully!")
    logging.info(f"Results saved to: {args.output_dir}")
    logging.info("="*70 + "\n")


if __name__ == '__main__':
    main()
