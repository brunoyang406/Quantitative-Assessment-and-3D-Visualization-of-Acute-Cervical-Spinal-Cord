"""
Miscellaneous utility functions

This module provides:
- Random seed setting for reproducibility
- Code snapshot creation for experiments
- Utility functions for experiment management
"""

import os
import sys
import shutil
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across all random number generators
    
    This function sets seeds for:
    - NumPy random number generator
    - PyTorch random number generator (CPU)
    - PyTorch CUDA random number generators (all GPUs)
    - CuDNN deterministic mode
    
    Args:
        seed: Random seed value
    
    Example:
        >>> set_random_seed(42)
        >>> # All random operations will be reproducible
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Enable deterministic mode for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logging.info(f"Random seed set to: {seed} (deterministic mode enabled)")


def copy_code_to_experiment(
    root_dir: str,
    config_path: Optional[str] = None,
    copy_directories: bool = True,
    copy_requirements: bool = True
) -> None:
    """
    Copy training scripts, config files, and code to experiment directory for reproducibility
    
    This function creates a code snapshot that includes:
    - Training scripts (train_*.py)
    - Configuration files
    - Source code directories (models, losses, trainers, data, utils)
    - Requirements file (if exists)
    - Command line used to run training
    
    Args:
        root_dir: Experiment root directory
        config_path: Path to configuration file (optional)
        copy_directories: Whether to copy source code directories (default: True)
        copy_requirements: Whether to copy requirements.txt (default: True)
    
    Example:
        >>> copy_code_to_experiment(
        ...     './experiments/exp_001',
        ...     config_path='configs/multimodal_lesion_unet.yaml'
        ... )
    """
    code_dir = Path(root_dir) / "code_snapshot"
    code_dir.mkdir(parents=True, exist_ok=True)
    
    copied_count = 0
    failed_count = 0
    
    # Get project root directory (assume this file is in utils/)
    project_root = Path(__file__).parent.parent
    
    # Training scripts to copy
    training_scripts = [
        "train_lesion_unet.py",
        "train_multitask.py",
        "resume_training.py",
    ]
    
    # Copy training scripts
    for filename in training_scripts:
        src = project_root / filename
        if src.exists():
            dst = code_dir / filename
            try:
                shutil.copy2(src, dst)
                copied_count += 1
                logging.debug(f"Copied {filename}")
            except Exception as e:
                failed_count += 1
                logging.warning(f"Failed to copy {filename}: {e}")
    
    # Copy config file
    if config_path:
        config_path_obj = Path(config_path)
        if config_path_obj.exists():
            # If config_path is relative, try to resolve it
            if not config_path_obj.is_absolute():
                config_path_obj = project_root / config_path
                if not config_path_obj.exists():
                    config_path_obj = Path(config_path)  # Try original path
            
            if config_path_obj.exists():
                config_filename = config_path_obj.name
                dst = code_dir / config_filename
                try:
                    shutil.copy2(config_path_obj, dst)
                    copied_count += 1
                    logging.info(f"Copied config: {config_filename}")
                except Exception as e:
                    failed_count += 1
                    logging.warning(f"Failed to copy config: {e}")
        else:
            logging.warning(f"Config file not found: {config_path}")
    
    # Copy source code directories
    if copy_directories:
        directories_to_copy = [
            "models",
            "losses",
            "trainers",
            "data",
            "utils",
            "inference",
            "configs",
        ]
        
        for dirname in directories_to_copy:
            src_dir = project_root / dirname
            if src_dir.exists() and src_dir.is_dir():
                dst_dir = code_dir / dirname
                try:
                    # Use copytree with ignore patterns
                    shutil.copytree(
                        src_dir,
                        dst_dir,
                        ignore=shutil.ignore_patterns(
                            '__pycache__',
                            '*.pyc',
                            '*.pyo',
                            '.git',
                            '*.log',
                            '*.pth',
                            '*.pt',
                            '*.h5',
                            '.DS_Store',
                        ),
                        dirs_exist_ok=True
                    )
                    copied_count += 1
                    logging.debug(f"Copied directory: {dirname}")
                except Exception as e:
                    failed_count += 1
                    logging.warning(f"Failed to copy directory {dirname}: {e}")
    
    # Copy requirements.txt if exists
    if copy_requirements:
        requirements_file = project_root / "requirements.txt"
        if requirements_file.exists():
            dst = code_dir / "requirements.txt"
            try:
                shutil.copy2(requirements_file, dst)
                copied_count += 1
                logging.debug("Copied requirements.txt")
            except Exception as e:
                failed_count += 1
                logging.warning(f"Failed to copy requirements.txt: {e}")
    
    # Save command line and environment info
    try:
        # Save command line
        cmd_file = code_dir / "command.txt"
        with open(cmd_file, 'w', encoding='utf-8') as f:
            f.write(' '.join(sys.argv))
            f.write('\n')
        copied_count += 1
        
        # Save Python version and PyTorch version
        env_file = code_dir / "environment.txt"
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(f"Python version: {sys.version}\n")
            f.write(f"PyTorch version: {torch.__version__}\n")
            if torch.cuda.is_available():
                f.write(f"CUDA version: {torch.version.cuda}\n")
                f.write(f"CUDA devices: {torch.cuda.device_count()}\n")
        copied_count += 1
        logging.debug("Saved environment information")
    except Exception as e:
        failed_count += 1
        logging.warning(f"Failed to save command/environment info: {e}")
    
    # Summary
    if copied_count > 0:
        logging.info(f"Code snapshot created: {copied_count} items copied to {code_dir}")
    if failed_count > 0:
        logging.warning(f"Failed to copy {failed_count} items")