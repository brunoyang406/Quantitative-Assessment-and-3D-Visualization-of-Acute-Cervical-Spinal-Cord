#!/usr/bin/env python3
"""
CLI: build uncertainty-boundary NIfTIs for optional model input `uncertainty_boundary`.

Pipeline: load checkpoint → same transforms as training → `inference.uncertainty` (MC Dropout +
`compute_uncertainty_boundary`) → save per-subject volumes under output_dir/<split>/uncertainty_boundaries/.

Usage:
  python tools/generate_uncertainty_boundaries.py \\
    --experiment_dir experiments/multimodal_lesion_unet_YYYYMMDD_HHMMSS \\
    --output_dir uncertainty_boundaries \\
    --num_samples 10 \\
    --splits train val test

Core library code: `inference/uncertainty.py` (this file is I/O + dataset wiring only).
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from tqdm import tqdm
from scipy import ndimage

# MONAI imports
from monai.config import print_config
from monai.data import DataLoader, Dataset
from monai.transforms import Spacing

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from models import create_model
from data.multimodal_transforms import get_multimodal_lesion_unet_transforms
from data.dataloader import prepare_data_dicts, load_data_list
from inference.uncertainty import predict_with_uncertainty, compute_uncertainty_boundary

print_config()


def setup_logging(output_dir: str):
    """Configure file + console logging."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"uncertainty_generation_{timestamp}.log")
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logging.info(f"Log file: {log_file}")


def load_config_from_experiment(experiment_dir: str) -> Dict[str, Any]:
    """Load YAML config from experiment `code_snapshot` or repo `configs/`."""
    config_path = os.path.join(experiment_dir, "code_snapshot", "multimodal_lesion_unet.yaml")
    if not os.path.exists(config_path):
        legacy = os.path.join(experiment_dir, "code_snapshot", "single_task_lesion.yaml")
        if os.path.exists(legacy):
            config_path = legacy
    if not os.path.exists(config_path):
        config_path = os.path.join("configs", "multimodal_lesion_unet.yaml")
    if not os.path.exists(config_path):
        legacy2 = os.path.join("configs", "single_task_lesion.yaml")
        if os.path.exists(legacy2):
            config_path = legacy2
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found. Tried multimodal_lesion_unet.yaml and legacy single_task_lesion.yaml: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    flat_config = {}
    
    if 'data' in config:
        flat_config['data_dir'] = config['data'].get('data_dir', './spinal_cord_dataset')
        flat_config['spatial_size'] = tuple(config['data'].get('spatial_size', [16, 512, 256]))
        flat_config['target_spacing'] = tuple(config['data'].get('target_spacing', [3.3, 0.54, 0.54]))
        flat_config['use_spacing'] = config['data'].get('use_spacing', True)
    
    if 'model' in config:
        flat_config['model'] = config['model']
    
    if 'roi_crop' in config:
        flat_config['roi_crop'] = config['roi_crop']
    
    if 'normalization' in config:
        flat_config['normalization'] = config['normalization']
    
    flat_config['train_json'] = config.get('train_json', './spinal_cord_dataset/train.json')
    flat_config['val_json'] = config.get('val_json', './spinal_cord_dataset/val.json')
    flat_config['test_json'] = config.get('test_json', './spinal_cord_dataset/test.json')
    
    return flat_config


def find_best_model_checkpoint(experiment_dir: str) -> str:
    """Pick best (or newest) checkpoint under `experiment_dir/weights`."""
    import glob
    weights_dir = os.path.join(experiment_dir, "weights")
    
    if not os.path.exists(weights_dir):
        raise FileNotFoundError(f"Weights directory not found: {weights_dir}")
    
    checkpoint_files = glob.glob(os.path.join(weights_dir, "*.pt"))
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint .pt files in: {weights_dir}")
    
    # Prefer filenames containing best_model_checkpoint
    best_checkpoints = [f for f in checkpoint_files if "best_model_checkpoint" in os.path.basename(f)]
    
    if best_checkpoints:
        best_checkpoints.sort(key=os.path.getmtime, reverse=True)
        return best_checkpoints[0]
    else:
        checkpoint_files.sort(key=os.path.getmtime, reverse=True)
        return checkpoint_files[0]


def load_model(model_path: str, config: Dict[str, Any], device: torch.device) -> nn.Module:
    """
    Build model from config and load checkpoint weights.

    Args:
        model_path: Path to .pt checkpoint.
        config: Flattened config dict (spatial_size, model, ...).
        device: torch device.

    Returns:
        Model on `device`, eval mode left to caller.
    """
    model_config = config['model'].copy()
    model_config['type'] = 'multimodal_lesion_unet'
    if 'in_channels' not in model_config:
        include_cord = model_config.get('include_cord', True)
        model_config['in_channels'] = 4 if include_cord else 3
    
    if 'include_cord' not in model_config:
        model_config['include_cord'] = True
    
    if 'lesion_out_channels' not in model_config:
        model_config['lesion_out_channels'] = 1
    
    model = create_model(
        model_config=model_config,
        img_size=config['spatial_size']
    )
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    logging.info(f"Loading model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Support several checkpoint dict layouts
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Strip DataParallel "module." prefix if present
    if any(key.startswith('module.') for key in state_dict.keys()):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_state_dict[key[7:]] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict, strict=False)
    logging.info("Model lesion head: 1 channel (sigmoid)")
    model = model.to(device)
    
    logging.info("Model loaded successfully")
    
    return model


def get_original_image_metadata(data_dict: Dict[str, Any]) -> tuple:
    """
    Read original-space metadata from the subject T1 NIfTI.

    Returns:
        (affine, header, original_shape, original_spacing, original_nii)
    """
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
    """
    Load cord_mask and apply the same Orientation(RAS) + Spacing as training.

    Returns:
        (cord_mask_processed, preprocessed_shape)
    """
    from monai.transforms import LoadImage, EnsureChannelFirst, Orientation, Spacing
    
    cord_path = data_dict.get('cord_mask')
    if not cord_path or not os.path.exists(cord_path):
        raise FileNotFoundError(f"Cord mask not found: {cord_path}")
    
    # Match training preprocessing
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


def transform_from_ras_to_original(
    data: np.ndarray,
    original_nii: nib.Nifti1Image,
    is_label: bool = False
) -> np.ndarray:
    """
    Reorient array from RAS axis order to the original NIfTI orientation.
    """
    original_ornt = nib.io_orientation(original_nii.affine)
    
    ras_ornt = np.array([[0, 1], [1, 1], [2, 1]])  # R=x+, A=y+, S=z+
    
    ornt_transform = nib.orientations.ornt_transform(ras_ornt, original_ornt)
    
    if is_label:
        transformed = nib.orientations.apply_orientation(data, ornt_transform)
    else:
        transformed = nib.orientations.apply_orientation(data, ornt_transform)
    
    return transformed


def compute_roi_crop_params(
    cord_mask: np.ndarray,
    target_size: tuple,
    margin: tuple
) -> tuple:
    """
    ROI crop/pad parameters aligned with cord-based cropping in training.

    Returns:
        (crop_start, crop_end, pad_before, pad_after)
    """
    target_size = np.array(target_size)
    margin = np.array(margin)
    
    coords = np.where(cord_mask > 0)
    
    if len(coords[0]) == 0:
        # No cord voxels: use full volume
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
    """
    Undo ROI crop: strip symmetric pad, then paste crop back into full preprocessed shape.
    """
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


def resample_to_original_space(
    data: np.ndarray,
    original_nii: nib.Nifti1Image,
    current_spacing: tuple,
    is_label: bool = False
) -> np.ndarray:
    """
    Reorient from RAS to original NIfTI axes, then resample to original voxel grid.

    Args:
        data: (D, H, W) in RAS training grid.
        current_spacing: (sx, sz, sy) matching [X, Z, Y] for that grid.
        is_label: nearest vs linear interpolation.
    """
    try:
        original_ornt = nib.io_orientation(original_nii.affine)
        
        ras_ornt = np.array([[0, 1], [1, 1], [2, 1]])  # R=x+, A=y+, S=z+
        
        ornt_transform = nib.orientations.ornt_transform(ras_ornt, original_ornt)
        
        data_reoriented = nib.orientations.apply_orientation(data, ornt_transform)
        
        logging.debug(
            f"    Reorient: RAS shape={data.shape} -> original axes shape={data_reoriented.shape}"
        )
    except Exception as e:
        logging.warning(f"    Reorientation failed, skipping: {e}")
        data_reoriented = data
    
    target_shape = original_nii.shape
    current_shape = np.array(data_reoriented.shape)
    target_shape_arr = np.array(target_shape)
    
    zoom_factors = target_shape_arr / current_shape
    
    logging.debug(f"    Resample: {current_shape} -> {target_shape}, zoom_factors={zoom_factors}")
    
    if is_label:
        resampled = ndimage.zoom(data_reoriented, zoom_factors, order=0, mode='nearest')
    else:
        resampled = ndimage.zoom(data_reoriented, zoom_factors, order=1, mode='nearest')
    
    # Trim/pad to exact target shape (rounding)
    if resampled.shape != tuple(target_shape):
        resampled = resize_to_target_shape(resampled, target_shape, is_label=is_label)
    
    return resampled


def resize_to_target_shape(
    data: np.ndarray,
    target_shape: tuple,
    is_label: bool = False
) -> np.ndarray:
    """Center-crop or center-pad `data` to `target_shape`."""
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


def generate_uncertainty_boundaries(
    model: nn.Module,
    data_json: str,
    config: Dict[str, Any],
    output_dir: str,
    device: torch.device,
    num_samples: int = 10,
    uncertainty_percentile: float = 75,
    threshold: float = 0.5,
    split_name: str = "train",
    save_all_uncertainties: bool = False
):
    """
    Run MC dropout, compute uncertainty boundary maps, save NIfTIs per subject.

    Args:
        model: Trained model (eval + dropout still active for MC).
        data_json: Split JSON list path.
        config: Flattened training-like config.
        split_name: train / val / test (for output subfolders).
        save_all_uncertainties: Also write epistemic, aleatoric, prediction volumes.
    """
    boundaries_dir = os.path.join(output_dir, split_name, "uncertainty_boundaries")
    os.makedirs(boundaries_dir, exist_ok=True)
    
    if save_all_uncertainties:
        epistemic_dir = os.path.join(output_dir, split_name, "epistemic")
        aleatoric_dir = os.path.join(output_dir, split_name, "aleatoric")
        prediction_dir = os.path.join(output_dir, split_name, "prediction")
        os.makedirs(epistemic_dir, exist_ok=True)
        os.makedirs(aleatoric_dir, exist_ok=True)
        os.makedirs(prediction_dir, exist_ok=True)
    
    data_list = load_data_list(data_json)
    data_dicts = prepare_data_dicts(data_list, data_dir=config.get('data_dir'))
    
    _, val_transforms = get_multimodal_lesion_unet_transforms(config)
    dataset = Dataset(data=data_dicts, transform=val_transforms)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    
    logging.info(f"\nGenerating uncertainty-boundary maps for split: {split_name}")
    logging.info(f"Number of samples: {len(dataset)}")
    logging.info(f"MC samples: {num_samples}")
    logging.info(f"Uncertainty percentile threshold: {uncertainty_percentile}")
    logging.info(f"Binary threshold: {threshold}")
    
    stats = {
        'total_samples': len(dataset),
        'successful': 0,
        'failed': 0,
        'boundary_stats': {
            'mean_boundary_pixels': [],
            'mean_boundary_percentage': [],
        }
    }
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc=f"{split_name} split")):
            try:
                images = batch["image"].to(device)  # (1, 3, D, H, W)
                
                data_dict = data_dicts[idx]
                affine, header, original_shape, original_spacing, original_nii = get_original_image_metadata(data_dict)
                
                cord_mask_processed, preprocessed_shape = load_and_preprocess_cord_mask(data_dict, config)
                
                roi_config = config.get('roi_crop', {})
                use_roi_crop = roi_config.get('use_roi_crop', False)
                
                if use_roi_crop:
                    roi_target_size = tuple(roi_config.get('roi_target_size', config.get('spatial_size', [16, 512, 256])))
                    roi_margin = tuple(roi_config.get('roi_margin', (1, 27, 27)))
                    
                    crop_start, crop_end, pad_before, pad_after = compute_roi_crop_params(
                        cord_mask=cord_mask_processed,
                        target_size=roi_target_size,
                        margin=roi_margin
                    )
                else:
                    crop_start = np.array([0, 0, 0])
                    crop_end = np.array(preprocessed_shape)
                    pad_before = np.array([0, 0, 0])
                    pad_after = np.array([0, 0, 0])
                
                prediction, aleatoric, epistemic = predict_with_uncertainty(
                    model=model,
                    image=images,
                    num_samples=num_samples,
                    device=device,
                    apply_sigmoid=True,
                    apply_softmax=False
                )
                pred_lesion = prediction[0, 0, ...].cpu().numpy()  # (D, H, W)
                epistemic_lesion = epistemic[0, 0, ...].cpu().numpy()  # (D, H, W)
                aleatoric_lesion = aleatoric[0, 0, ...].cpu().numpy()  # (D, H, W)
                
                pred_lesion = np.clip(pred_lesion, 0.0, 1.0)
                epistemic_lesion = np.clip(epistemic_lesion, 0.0, 1.0)
                aleatoric_lesion = np.clip(aleatoric_lesion, 0.0, 1.0)
                
                boundary_info = compute_uncertainty_boundary(
                    prediction=pred_lesion,
                    epistemic=epistemic_lesion,
                    threshold=threshold,
                    uncertainty_percentile=uncertainty_percentile
                )
                
                # Weighted boundary map: mask * (aleatoric + epistemic) for continuous uncertainty on edges
                pred_boundary_mask = boundary_info['pred_boundary'].astype(np.float32)
                total_uncertainty = (aleatoric_lesion + epistemic_lesion).astype(np.float32)
                total_uncertainty = np.maximum(total_uncertainty, 0.0)
                uncertainty_boundary = pred_boundary_mask * total_uncertainty  # (D, H, W)
                
                # Inverse: ROI crop -> full preprocessed grid -> original NIfTI space
                subject_id = data_dict.get('subject_id', f'sample_{idx:04d}')
                
                logging.info(f"  Subject {subject_id}:")
                logging.info(f"    Model output (after ROI crop): shape={pred_lesion.shape}")
                logging.info(f"    Preprocessed grid (RAS + target spacing): shape={preprocessed_shape}")
                logging.info(f"    Original image space: shape={original_shape}, spacing={original_spacing}")
                
                if use_roi_crop:
                    logging.info("    Inverse ROI crop...")
                    logging.info(f"      crop: [{crop_start}] -> [{crop_end}], pad: before={pad_before}, after={pad_after}")
                    
                    uncertainty_boundary_full = inverse_roi_crop(
                        data=uncertainty_boundary,
                        original_shape=preprocessed_shape,
                        crop_start=crop_start,
                        crop_end=crop_end,
                        pad_before=pad_before,
                        pad_after=pad_after
                    )
                    
                    pred_lesion_full = inverse_roi_crop(
                        data=pred_lesion,
                        original_shape=preprocessed_shape,
                        crop_start=crop_start,
                        crop_end=crop_end,
                        pad_before=pad_before,
                        pad_after=pad_after
                    )
                    
                    epistemic_lesion_full = inverse_roi_crop(
                        data=epistemic_lesion,
                        original_shape=preprocessed_shape,
                        crop_start=crop_start,
                        crop_end=crop_end,
                        pad_before=pad_before,
                        pad_after=pad_after
                    )
                    
                    aleatoric_lesion_full = inverse_roi_crop(
                        data=aleatoric_lesion,
                        original_shape=preprocessed_shape,
                        crop_start=crop_start,
                        crop_end=crop_end,
                        pad_before=pad_before,
                        pad_after=pad_after
                    )
                    
                    logging.info(f"      After inverse ROI crop: shape={uncertainty_boundary_full.shape}")
                else:
                    uncertainty_boundary_full = uncertainty_boundary
                    pred_lesion_full = pred_lesion
                    epistemic_lesion_full = epistemic_lesion
                    aleatoric_lesion_full = aleatoric_lesion
                
                logging.info("    Resample to original spacing + reorient to original axes...")
                
                target_spacing_config = config.get('target_spacing', [3.3, 0.54, 0.54])  # [X, Y, Z]
                current_spacing = (target_spacing_config[0], target_spacing_config[2], target_spacing_config[1])
                
                uncertainty_boundary_original = resample_to_original_space(
                    data=uncertainty_boundary_full,
                    original_nii=original_nii,
                    current_spacing=current_spacing,
                    is_label=False
                )
                
                pred_lesion_original = resample_to_original_space(
                    data=pred_lesion_full,
                    original_nii=original_nii,
                    current_spacing=current_spacing,
                    is_label=False
                )
                
                epistemic_lesion_original = resample_to_original_space(
                    data=epistemic_lesion_full,
                    original_nii=original_nii,
                    current_spacing=current_spacing,
                    is_label=False
                )
                
                aleatoric_lesion_original = resample_to_original_space(
                    data=aleatoric_lesion_full,
                    original_nii=original_nii,
                    current_spacing=current_spacing,
                    is_label=False
                )
                
                logging.info(f"    Final in original space: shape={uncertainty_boundary_original.shape}")
                logging.info("    Restored to original image space OK")
                
                stats['boundary_stats']['mean_boundary_pixels'].append(boundary_info['boundary_pixels'])
                stats['boundary_stats']['mean_boundary_percentage'].append(boundary_info['boundary_percentage'])
                
                output_filename = f"{subject_id}_uncertainty_boundary.nii.gz"
                output_path = os.path.join(boundaries_dir, output_filename)
                
                uncertainty_nii = nib.Nifti1Image(
                    uncertainty_boundary_original,
                    affine=affine,
                    header=header
                )
                nib.save(uncertainty_nii, output_path)
                
                if save_all_uncertainties:
                    epistemic_path = os.path.join(epistemic_dir, f"{subject_id}_epistemic.nii.gz")
                    epistemic_nii = nib.Nifti1Image(
                        epistemic_lesion_original,
                        affine=affine,
                        header=header
                    )
                    nib.save(epistemic_nii, epistemic_path)
                    
                    aleatoric_path = os.path.join(aleatoric_dir, f"{subject_id}_aleatoric.nii.gz")
                    aleatoric_nii = nib.Nifti1Image(
                        aleatoric_lesion_original,
                        affine=affine,
                        header=header
                    )
                    nib.save(aleatoric_nii, aleatoric_path)
                    
                    pred_path = os.path.join(prediction_dir, f"{subject_id}_prediction.nii.gz")
                    pred_nii = nib.Nifti1Image(
                        pred_lesion_original,
                        affine=affine,
                        header=header
                    )
                    nib.save(pred_nii, pred_path)
                
                stats['successful'] += 1
                
            except Exception as e:
                logging.error(
                    f"Failed sample idx={idx} "
                    f"({data_dicts[idx].get('subject_id', 'unknown')}): {str(e)}"
                )
                import traceback
                logging.error(traceback.format_exc())
                stats['failed'] += 1
                continue
    
    if stats['boundary_stats']['mean_boundary_pixels']:
        stats['boundary_stats']['mean_boundary_pixels'] = float(np.mean(stats['boundary_stats']['mean_boundary_pixels']))
        stats['boundary_stats']['mean_boundary_percentage'] = float(np.mean(stats['boundary_stats']['mean_boundary_percentage']))
    else:
        stats['boundary_stats']['mean_boundary_pixels'] = 0.0
        stats['boundary_stats']['mean_boundary_percentage'] = 0.0
    
    logging.info(f"\nSplit {split_name} finished:")
    logging.info(f"  OK: {stats['successful']}/{stats['total_samples']}")
    logging.info(f"  Failed: {stats['failed']}/{stats['total_samples']}")
    if stats['successful'] > 0:
        logging.info(f"  Mean boundary voxels: {stats['boundary_stats']['mean_boundary_pixels']:.2f}")
        logging.info(f"  Mean boundary fraction: {stats['boundary_stats']['mean_boundary_percentage']:.2%}")
    logging.info(f"  Output dir: {boundaries_dir}")
    
    stats_file = os.path.join(output_dir, f"{split_name}_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate uncertainty-boundary NIfTIs for optional multimodal input.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # All splits
  python tools/generate_uncertainty_boundaries.py \\
    --experiment_dir experiments/multimodal_lesion_unet_20260101_120000 \\
    --output_dir uncertainty_boundaries \\
    --num_samples 10 \\
    --splits train val test

  # Train only
  python tools/generate_uncertainty_boundaries.py \\
    --experiment_dir experiments/multimodal_lesion_unet_20260101_120000 \\
    --output_dir uncertainty_boundaries \\
    --num_samples 20 \\
    --splits train
        """
    )
    
    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        help="Experiment folder with weights/ and code_snapshot/",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="uncertainty_boundaries",
        help="Root output directory (default: uncertainty_boundaries)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="MC Dropout forward passes (default: 10; typical 10–50)",
    )
    parser.add_argument(
        "--uncertainty_percentile",
        type=float,
        default=75,
        help="Percentile for high-uncertainty voxels (default: 75, range 0–100)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for lesion mask (default: 0.5)",
    )
    parser.add_argument(
        "--splits",
        nargs='+',
        default=['train', 'val', 'test'],
        help="Splits to process (default: train val test)",
    )
    parser.add_argument(
        "--save_all",
        action="store_true",
        help="Also save epistemic, aleatoric, and prediction volumes",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Optional path to .pt; default: newest best_model_checkpoint under weights/",
    )
    
    args = parser.parse_args()
    
    setup_logging(args.output_dir)
    
    logging.info("="*70)
    logging.info("Uncertainty-boundary NIfTI generator")
    logging.info("="*70)
    logging.info(f"Experiment dir: {args.experiment_dir}")
    logging.info(f"Output dir: {args.output_dir}")
    logging.info(f"MC samples: {args.num_samples}")
    logging.info(f"Uncertainty percentile: {args.uncertainty_percentile}")
    logging.info(f"Threshold: {args.threshold}")
    logging.info(f"Splits: {args.splits}")
    logging.info(f"Save all uncertainty maps: {args.save_all}")
    logging.info("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"\nDevice: {device}")
    
    logging.info("\n[1/3] Loading config...")
    config = load_config_from_experiment(args.experiment_dir)
    logging.info("Config loaded")
    logging.info(f"  data_dir: {config['data_dir']}")
    logging.info(f"  spatial_size: {config['spatial_size']}")
    logging.info(f"  target_spacing: {config['target_spacing']}")
    
    logging.info("\n[2/3] Loading model...")
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = find_best_model_checkpoint(args.experiment_dir)
    logging.info(f"Checkpoint: {model_path}")
    model = load_model(model_path, config, device)
    
    logging.info("\n[3/3] Generating maps...")
    
    all_stats = {}
    for split in args.splits:
        if split == 'train':
            data_json = config['train_json']
        elif split == 'val':
            data_json = config['val_json']
        elif split == 'test':
            data_json = config['test_json']
        else:
            logging.warning(f"Unknown split '{split}', skipping")
            continue
        
        if not os.path.exists(data_json):
            logging.warning(f"JSON list not found: {data_json}, skipping")
            continue
        
        stats = generate_uncertainty_boundaries(
            model=model,
            data_json=data_json,
            config=config,
            output_dir=args.output_dir,
            device=device,
            num_samples=args.num_samples,
            uncertainty_percentile=args.uncertainty_percentile,
            threshold=args.threshold,
            split_name=split,
            save_all_uncertainties=args.save_all
        )
        all_stats[split] = stats
    
    overall_stats_file = os.path.join(args.output_dir, "overall_stats.json")
    with open(overall_stats_file, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    
    logging.info("\n" + "="*70)
    logging.info("All requested splits finished")
    logging.info("="*70)
    logging.info(f"Outputs under: {args.output_dir}")
    logging.info(f"Summary JSON: {overall_stats_file}")
    
    logging.info("\nPer-split OK / total:")
    for split, stats in all_stats.items():
        logging.info(f"  {split}: {stats['successful']}/{stats['total_samples']}")


if __name__ == "__main__":
    main()
