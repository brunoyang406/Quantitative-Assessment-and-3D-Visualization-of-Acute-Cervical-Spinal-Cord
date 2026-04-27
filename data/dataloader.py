"""
Data loader creation utilities for multimodal MRI segmentation
"""

import os
import json
import logging
from typing import Tuple, List, Dict

from torch.utils.data import DataLoader
from monai.data import Dataset
from monai.transforms import Compose


def load_data_list(json_path: str) -> List[Dict]:
    """
    Load data list from JSON file
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Data list
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # All JSON files use dict format with 'data' key
    if isinstance(data, dict) and 'data' in data:
        return data['data']
    else:
        raise ValueError(f"Invalid JSON format in {json_path}. Expected dict with 'data' key.")


def prepare_data_dicts(
    data_list: List[Dict],
    data_dir: str = None,
    split: str = None,  # 'train', 'val', or 'test' - for future use
    uncertainty_boundary_dir: str = None,  # Directory containing uncertainty boundary files
) -> List[Dict]:
    """
    Prepare MONAI-format data dictionaries
    
    Args:
        data_list: Data list loaded from JSON
        data_dir: Root directory for data files (optional)
        uncertainty_boundary_dir: Directory containing uncertainty boundary files (optional)
        split: Dataset split name ('train', 'val', or 'test') - determines which uncertainty_boundary directory to search
        
    Returns:
        List of MONAI-format data dictionaries
    """
    data_dicts = []
    
    for item in data_list:
        data_dict = {
            'subject_id': item.get('subject_id', ''),
            'center': item.get('center', ''),
            'injury_grade': item.get('injury_grade', ''),
        }
        
        # Handle file paths - prepend data_dir if provided and paths are relative
        for key in ['T1', 'T2', 'T2FS', 'lesion_mask', 'cord_mask']:
            path = item.get(key, '')
            if path:
                # Normalize path separators (convert Windows backslashes to forward slashes)
                # This handles paths in JSON files that may have Windows-style separators
                path = path.replace('\\', '/')
                if data_dir and not os.path.isabs(path):
                    # Data is in raw/ subdirectory
                    path = os.path.join(data_dir, 'raw', path)
            data_dict[key] = path
        
        # Add uncertainty_boundaries path if directory is provided
        if uncertainty_boundary_dir and split:
            subject_id = item.get('subject_id', '')
            if subject_id:
                # Construct path: uncertainty_boundaries/{split}/uncertainty_boundaries/{subject_id}_uncertainty_boundary.nii.gz
                uncertainty_path = os.path.join(
                    uncertainty_boundary_dir, 
                    split, 
                    'uncertainty_boundaries',
                    f"{subject_id}_uncertainty_boundary.nii.gz"
                )
                data_dict['uncertainty_boundary'] = uncertainty_path
        
        data_dicts.append(data_dict)
    
    return data_dicts


def create_data_loaders(
    train_json: str,
    val_json: str,
    test_json: str,
    train_transforms: Compose,
    val_transforms: Compose,
    batch_size: int = 2,
    val_batch_size: int = 1,
    num_workers: int = 8,
    val_num_workers: int = 4,
    pin_memory: bool = True,
    data_dir: str = None,
    uncertainty_boundary_dir: str = None,
    ) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders for multimodal MRI
    
    Args:
        train_json: Training JSON file
        val_json: Validation JSON file
        test_json: Test JSON file
        train_transforms: Training transforms
        val_transforms: Validation transforms
        batch_size: Training batch size
        val_batch_size: Validation batch size
        num_workers: Number of workers for training loader
        val_num_workers: Number of workers for validation loader
        pin_memory: Whether to pin memory for faster GPU transfer
        data_dir: Root directory for data files (optional)
        uncertainty_boundary_dir: Directory containing uncertainty boundary files (optional)

    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
    """
    # Load data lists
    train_data_list = load_data_list(train_json)
    val_data_list = load_data_list(val_json)
    test_data_list = load_data_list(test_json)

    # Prepare data dicts
    train_data_dicts = prepare_data_dicts(train_data_list, data_dir=data_dir, split='train', uncertainty_boundary_dir=uncertainty_boundary_dir)
    val_data_dicts = prepare_data_dicts(val_data_list, data_dir=data_dir, split='val', uncertainty_boundary_dir=uncertainty_boundary_dir)
    test_data_dicts = prepare_data_dicts(test_data_list, data_dir=data_dir, split='test', uncertainty_boundary_dir=uncertainty_boundary_dir)
    
    logging.info(f"Number of training samples: {len(train_data_dicts)}")
    logging.info(f"Number of validation samples: {len(val_data_dicts)}")
    logging.info(f"Number of test samples: {len(test_data_dicts)}")
    
    train_dataset = Dataset(data=train_data_dicts, transform=train_transforms)
    val_dataset = Dataset(data=val_data_dicts, transform=val_transforms)
    test_dataset = Dataset(data=test_data_dicts, transform=val_transforms)
    
    # Create data loaders
    # Disable persistent_workers to reduce GPU memory usage
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False  # Disabled to reduce memory usage
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=val_num_workers,
        pin_memory=pin_memory,
        persistent_workers=False  # Disabled to reduce memory usage
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=val_num_workers,
        pin_memory=pin_memory,
        persistent_workers=False  # Disabled to reduce memory usage
    )

    
    logging.info(f"Train loader: {len(train_loader)} batches")
    logging.info(f"Val loader: {len(val_loader)} batches")
    logging.info(f"Test loader: {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_json = "/root/autodl-tmp/scl/spinal_cord_dataset/train.json"
    val_json = "/root/autodl-tmp/scl/spinal_cord_dataset/val.json"
    test_json = "/root/autodl-tmp/scl/spinal_cord_dataset/test.json"
    data_dir = "/root/autodl-tmp/scl/spinal_cord_dataset"
    train_loader, val_loader = create_data_loaders(
        train_json=train_json,
        val_json=val_json,
        test_json=test_json,
        train_transforms=None,
        val_transforms=None,
        batch_size=1,
        val_batch_size=1,
        num_workers=1,
        val_num_workers=1,
        data_dir=data_dir,
    )
    for data in train_loader:
        print(data)