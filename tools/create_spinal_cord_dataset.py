#!/usr/bin/env python3
"""
Spinal Cord Lesion Dataset Generation Script
Generate complete dataset JSON files with training/validation/testing splits

Supports multi-modal inputs (T1, T2, T2FS) and multi-task outputs (Cord + Lesion)
"""

import os
import json
import argparse
import random
import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import logging


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def load_injury_grades_from_csv(csv_path: str) -> Dict[Tuple[str, str], str]:
    """Parse optional CSV of injury grades into ``{(center, subject_id): grade}``."""
    injury_grades = {}
    
    if not os.path.exists(csv_path):
        logging.warning(f"Injury grade CSV file not found: {csv_path}")
        return injury_grades
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            
            for row in reader:
                if len(row) < 3 or not row[0].strip():
                    continue
                
                center = row[0].strip()
                subject_id = row[1].strip()
                injury_grade = row[2].strip()
                
                key = (center, subject_id)
                injury_grades[key] = injury_grade
        
        logging.info("Loaded %s injury grade records from CSV", len(injury_grades))
    except Exception as e:
        logging.warning(f"Failed to load injury grades from CSV: {e}")
    
    return injury_grades


def collect_valid_samples(raw_dir: str) -> List[Tuple[str, str, str]]:
    """
    Collect all valid samples
    
    Args:
        raw_dir: Raw data directory path (contains center_tongji/, center_jlu/, center_PUTH/)
    
    Returns:
        List of valid samples, each element is (center_name, sample_id, sample_dir)
    """
    centers = ['center_tongji', 'center_jlu', 'center_PUTH']
    valid_samples = []
    missing_count = 0
    
    for center in centers:
        center_dir = os.path.join(raw_dir, center)
        
        if not os.path.exists(center_dir):
            logging.warning(f"Center directory not found: {center_dir}")
            continue
        
        # Iterate through all sample folders in the center directory
        sample_dirs = sorted([d for d in os.listdir(center_dir) 
                             if os.path.isdir(os.path.join(center_dir, d))])
        
        logging.info(f"Processing {center}: {len(sample_dirs)} sample folders found")
        
        for sample_id in sample_dirs:
            sample_dir = os.path.join(center_dir, sample_id)
            
            # Check required files (support multiple naming patterns)
            files = os.listdir(sample_dir)
            
            # T1 (only look for T1_reg)
            t1_file = f"{sample_id}_T1_reg.nii.gz" if f"{sample_id}_T1_reg.nii.gz" in files else None
            
            # T2
            t2_file = f"{sample_id}_T2.nii.gz"
            
            # T2FS (only look for T2FS_reg)
            t2fs_file = f"{sample_id}_T2FS_reg.nii.gz" if f"{sample_id}_T2FS_reg.nii.gz" in files else None
            
            # Cord mask
            cord_file = f"{sample_id}_cord_mask.nii.gz"
            
            # Lesion mask
            lesion_file = f"{sample_id}_lesion_mask.nii.gz"
            
            # Check if all required files exist
            required_files = [t2_file, cord_file, lesion_file]
            
            # Check required files
            has_required = all(f in files for f in required_files)
            
            # Check if T1 file exists
            has_t1 = t1_file is not None
            
            # Check if T2FS file exists
            has_t2fs = t2fs_file is not None
            
            # Sample validity condition: all required files exist and at least one modality file (T1 or T2FS)
            if has_required and (has_t1 or has_t2fs):
                valid_samples.append((center, sample_id, sample_dir))
            else:
                # Dataset already validated, only record skip info
                missing_count += 1
                logging.debug(f"Skipping {center}/{sample_id}: missing required files")
    
    logging.info(f"\nTotal valid samples: {len(valid_samples)}")
    if missing_count > 0:
        logging.warning(f"Skipped {missing_count} samples due to missing files")
    
    return valid_samples


def split_samples(
    samples: List[Tuple[str, str, str]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    seed: int = 42
) -> Dict[str, List[Tuple[str, str, str]]]:
    """
    Split samples into training, validation, and test sets
    
    Args:
        samples: Sample list
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed
    
    Returns:
        Dictionary containing 'training', 'validation', 'testing'
    """
    # Check ratio sum
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Shuffle samples
    shuffled_samples = samples.copy()
    random.shuffle(shuffled_samples)
    
    # Calculate split points
    total = len(shuffled_samples)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    # Split
    train_samples = shuffled_samples[:train_end]
    val_samples = shuffled_samples[train_end:val_end]
    test_samples = shuffled_samples[val_end:]
    
    logging.info(f"\nDataset split:")
    logging.info(f"  Training:   {len(train_samples):4d} samples ({len(train_samples)/total*100:.1f}%)")
    logging.info(f"  Validation: {len(val_samples):4d} samples ({len(val_samples)/total*100:.1f}%)")
    logging.info(f"  Testing:    {len(test_samples):4d} samples ({len(test_samples)/total*100:.1f}%)")
    
    return {
        'training': train_samples,
        'validation': val_samples,
        'testing': test_samples
    }


def create_sample_entry(
    center: str, 
    sample_id: str, 
    sample_dir: str, 
    injury_grades: Optional[Dict[Tuple[str, str], str]] = None,
    use_relative_paths: bool = True
) -> Dict[str, str]:
    """Create data entry for a single sample"""
    files = os.listdir(sample_dir)
    
    # Find T1 file (only look for T1_reg)
    t1_file = f"{sample_id}_T1_reg.nii.gz" if f"{sample_id}_T1_reg.nii.gz" in files else None
    
    # T2
    t2_file = f"{sample_id}_T2.nii.gz"
    
    # T2FS (only look for T2FS_reg)
    t2fs_file = f"{sample_id}_T2FS_reg.nii.gz" if f"{sample_id}_T2FS_reg.nii.gz" in files else None
    
    # Masks
    cord_file = f"{sample_id}_cord_mask.nii.gz"
    lesion_file = f"{sample_id}_lesion_mask.nii.gz"
    
    injury_grade = ""
    if injury_grades is not None:
        key = (center, sample_id)
        injury_grade = injury_grades.get(key, "")
    
    if use_relative_paths:
        # Use paths relative to raw directory
        base_path = os.path.join(center, sample_id)
        entry = {
            "subject_id": sample_id,
            "center": center,
            "T1": os.path.join(base_path, t1_file) if t1_file else "",
            "T2": os.path.join(base_path, t2_file),
            "T2FS": os.path.join(base_path, t2fs_file) if t2fs_file else "",
            "injury_grade": injury_grade,
            "cord_mask": os.path.join(base_path, cord_file),
            "lesion_mask": os.path.join(base_path, lesion_file),
        }
    else:
        # Use absolute paths
        entry = {
            "subject_id": sample_id,
            "center": center,
            "T1": os.path.join(sample_dir, t1_file) if t1_file else "",
            "T2": os.path.join(sample_dir, t2_file),
            "T2FS": os.path.join(sample_dir, t2fs_file) if t2fs_file else "",
            "injury_grade": injury_grade,
            "cord_mask": os.path.join(sample_dir, cord_file),
            "lesion_mask": os.path.join(sample_dir, lesion_file),
        }
    
    return entry


def create_dataset_dict(
    split_samples: Dict[str, List[Tuple[str, str, str]]],
    injury_grades: Optional[Dict[Tuple[str, str], str]] = None,
    use_relative_paths: bool = True
) -> Dict[str, Any]:
    """
    Create MONAI-format dataset dictionary
    
    Args:
        split_samples: Split sample dictionary
        injury_grades: Optional injury-grade map from CSV
        use_relative_paths: Whether to use relative paths
    
    Returns:
        MONAI-format dataset dictionary
    """
    # Create data lists for each split
    training_list = [create_sample_entry(center, sid, sdir, injury_grades, use_relative_paths) 
                     for center, sid, sdir in split_samples['training']]
    validation_list = [create_sample_entry(center, sid, sdir, injury_grades, use_relative_paths) 
                       for center, sid, sdir in split_samples['validation']]
    testing_list = [create_sample_entry(center, sid, sdir, injury_grades, use_relative_paths) 
                    for center, sid, sdir in split_samples['testing']]
    
    if injury_grades is not None:
        train_with_grade = sum(1 for item in training_list if item['injury_grade'])
        val_with_grade = sum(1 for item in validation_list if item['injury_grade'])
        test_with_grade = sum(1 for item in testing_list if item['injury_grade'])
        
        logging.info(f"  Training samples with injury_grade:   {train_with_grade}/{len(training_list)}")
        logging.info(f"  Validation samples with injury_grade: {val_with_grade}/{len(validation_list)}")
        logging.info(f"  Testing samples with injury_grade:    {test_with_grade}/{len(testing_list)}")
    
    # Create complete dataset dictionary
    dataset_dict = {
        "name": "Spinal Cord Lesion Segmentation Dataset",
        "description": "Multi-modal (T1, T2, T2FS) MRI dataset for spinal cord and lesion segmentation",
        "reference": "Multi-center dataset (Tongji, JLU, PUTH)",
        "licence": "N/A",
        "release": "1.0",
        "tensorImageSize": "3D",
        "modality": {
            "0": "MRI-T1",
            "1": "MRI-T2",
            "2": "MRI-T2FS"
        },
        "labels": {
            "cord": {
                "0": "background",
                "1": "spinal_cord"
            },
            "lesion": {
                "0": "background",
                "1": "lesion"
            }
        },
        "numTraining": len(training_list),
        "numValidation": len(validation_list),
        "numTesting": len(testing_list),
        "training": training_list,
        "validation": validation_list,
        "testing": testing_list
    }
    
    return dataset_dict


def save_dataset_json(dataset_dict: Dict[str, Any], output_path: str):
    """
    Save dataset dictionary to JSON file
    
    Args:
        dataset_dict: Dataset dictionary
        output_path: Output file path
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_dict, f, indent=2, ensure_ascii=False)
    
    logging.info(f"✓ Dataset JSON file saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate spinal cord lesion dataset JSON file with train/val/test splits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python create_spinal_cord_dataset.py --raw_dir ./spinal_cord_dataset/raw --output_dir ./spinal_cord_dataset

This will create train.json, val.json, test.json files with:
  - Training set (70%)
  - Validation set (15%)
  - Testing set (15%)
        """
    )
    
    parser.add_argument(
        "--raw_dir",
        type=str,
        required=True,
        help="Raw data directory containing center_tongji/, center_jlu/, center_PUTH/ folders"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./spinal_cord_dataset",
        help="Output directory for JSON files (default: ./spinal_cord_dataset)"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Training set ratio (default: 0.7)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.10,
        help="Validation set ratio (default: 0.15)"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.20,
        help="Test set ratio (default: 0.15)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--injury_csv",
        type=str,
        default=None,
        help="Optional CSV path for injury grades (e.g. raw/injury_grades.csv)",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    logging.info("=" * 70)
    logging.info("Spinal Cord Lesion Dataset Generation Tool")
    logging.info("=" * 70)
    logging.info(f"Raw data directory: {args.raw_dir}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Split ratios: Train={args.train_ratio}, Val={args.val_ratio}, Test={args.test_ratio}")
    logging.info(f"Random seed: {args.seed}")
    if args.injury_csv:
        logging.info(f"Injury grade CSV: {args.injury_csv}")
    logging.info("=" * 70)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 0. Load injury grades from CSV (if provided)
    injury_grades = None
    if args.injury_csv:
        logging.info("\n[0/4] Loading injury grades from CSV...")
        injury_grades = load_injury_grades_from_csv(args.injury_csv)
    
    # 1. Collect valid samples
    logging.info("\n[1/4] Collecting valid samples...")
    valid_samples = collect_valid_samples(args.raw_dir)
    
    if len(valid_samples) == 0:
        logging.error("No valid samples found! Exiting.")
        return
    
    # 2. Split dataset
    logging.info("\n[2/4] Splitting dataset...")
    split_dict = split_samples(
        valid_samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    # 3. Create dataset dictionary
    logging.info("\n[3/4] Creating dataset dictionary...")
    dataset_dict = create_dataset_dict(split_dict, injury_grades, use_relative_paths=True)
    
    # 4. Save to files (save separately as train.json, val.json, test.json)
    logging.info("\n[4/4] Saving JSON files...")
    train_output = os.path.join(args.output_dir, "train.json")
    val_output = os.path.join(args.output_dir, "val.json")
    test_output = os.path.join(args.output_dir, "test.json")
    
    # Save training set
    train_dict = {k: v for k, v in dataset_dict.items() if k != 'validation' and k != 'testing'}
    train_dict['data'] = dataset_dict['training']
    del train_dict['training']
    save_dataset_json(train_dict, train_output)
    
    # Save validation set
    val_dict = {k: v for k, v in dataset_dict.items() if k != 'training' and k != 'testing'}
    val_dict['data'] = dataset_dict['validation']
    del val_dict['validation']
    save_dataset_json(val_dict, val_output)
    
    # Save test set
    test_dict = {k: v for k, v in dataset_dict.items() if k != 'training' and k != 'validation'}
    test_dict['data'] = dataset_dict['testing']
    del test_dict['testing']
    save_dataset_json(test_dict, test_output)
    
    # 5. Output summary
    logging.info("\n" + "=" * 70)
    logging.info("Dataset Summary")
    logging.info("=" * 70)
    logging.info(f"Training samples:   {dataset_dict['numTraining']:4d}")
    logging.info(f"Validation samples: {dataset_dict['numValidation']:4d}")
    logging.info(f"Testing samples:    {dataset_dict['numTesting']:4d}")
    logging.info(f"Total samples:      {dataset_dict['numTraining'] + dataset_dict['numValidation'] + dataset_dict['numTesting']:4d}")
    logging.info("=" * 70)
    
    logging.info(f"\n✓ Done! Created {train_output}, {val_output}, {test_output}")
    logging.info(f"\nTo use this dataset, update your config file:")
    logging.info(f"  data:")
    logging.info(f"    data_dir: \"{args.raw_dir}\"")
    logging.info(f"    train_json: \"{train_output}\"")
    logging.info(f"    val_json: \"{val_output}\"")
    logging.info(f"    test_json: \"{test_output}\"")


if __name__ == "__main__":
    main()

