"""
Configuration loading and parsing utilities

This module provides:
- YAML configuration loading
- Configuration flattening (nested dict to flat dict)
- Command-line argument parsing and merging
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        Configuration dictionary
    
    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logging.info(f"✓ Loaded config file: {config_path}")
    return config


def _flatten_nested_config(config: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
    """
    Recursively flatten nested configuration dictionary
    
    Args:
        config: Nested configuration dictionary
        prefix: Prefix for flattened keys (used in recursion)
    
    Returns:
        Flattened configuration dictionary
    """
    flat = {}
    for key, value in config.items():
        new_key = f"{prefix}.{key}" if prefix else key
        
        if isinstance(value, dict):
            # Recursively flatten nested dictionaries
            flat.update(_flatten_nested_config(value, new_key))
        else:
            flat[new_key] = value
    
    return flat


def _extract_special_fields(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and process special configuration fields that need custom handling
    
    Args:
        config: Original configuration dictionary
    
    Returns:
        Dictionary with extracted special fields
    """
    flat_config = {}
    
    # Fields to preserve as-is (nested structures)
    preserve_fields = [
        'trainer_type', 'teacher_mode', 'input_mode', 'device',
        'loss', 'roi_crop', 'normalization', 'normalize_zscore',
        'model', 'student_model', 'teacher_model', 'teacher1_model', 'teacher2_model',
        'augmentation', 'distillation', 'scheduler'
    ]
    for field in preserve_fields:
        if field in config:
            flat_config[field] = config[field]
    
    # Data configuration (extract nested fields)
    if 'data' in config:
        data = config['data']
        flat_config['data_dir'] = data.get('data_dir', '')
        flat_config['json_file'] = data.get('json_file', '')
        # Support both single json_file and separate train/val/test json files
        flat_config['train_json'] = data.get('train_json', data.get('json_file', ''))
        flat_config['val_json'] = data.get('val_json', data.get('json_file', ''))
        flat_config['test_json'] = data.get('test_json', data.get('json_file', ''))
        flat_config['spatial_size'] = tuple(data.get('spatial_size', [160, 160, 160]))
        flat_config['target_spacing'] = tuple(data.get('target_spacing', [1.0, 1.0, 1.0]))
        flat_config['use_spacing'] = data.get('use_spacing', True)
        flat_config['intensity_min'] = data.get('intensity_min', -175)
        flat_config['intensity_max'] = data.get('intensity_max', 250)
        flat_config['uncertainty_boundary_dir'] = data.get('uncertainty_boundary_dir', None)
        if 'intensity_range' in data:
            flat_config['intensity_range'] = data['intensity_range']
    
    # Training configuration
    if 'training' in config:
        training = config['training']
        flat_config['max_iterations'] = training.get('max_iterations', 5000)
        flat_config['max_epochs'] = training.get('max_epochs', None)
        flat_config['eval_num'] = training.get('eval_num', 50)
        flat_config['batch_size'] = training.get('batch_size', 2)
        flat_config['val_batch_size'] = training.get('val_batch_size', 1)
        flat_config['seed'] = training.get('seed', 42)
        flat_config['checkpoint_interval'] = training.get('checkpoint_interval', 10)
    
    # Optimizer configuration (handle scientific notation)
    if 'optimizer' in config:
        opt = config['optimizer']
        lr = opt.get('lr', 1e-4)
        flat_config['lr'] = float(lr) if isinstance(lr, str) else lr
        weight_decay = opt.get('weight_decay', 1e-5)
        flat_config['weight_decay'] = float(weight_decay) if isinstance(weight_decay, str) else weight_decay
        flat_config['optimizer_type'] = opt.get('type', 'AdamW')
    
    # Data loader configuration
    if 'dataloader' in config:
        loader = config['dataloader']
        flat_config['num_workers'] = loader.get('num_workers', 8)
        flat_config['val_num_workers'] = loader.get('val_num_workers', 4)
        flat_config['pin_memory'] = loader.get('pin_memory', True)
    
    # Experiment configuration
    if 'experiment' in config:
        exp = config['experiment']
        flat_config['root_dir'] = exp.get('root_dir', 'experiments')
        flat_config['experiment_name'] = exp.get('name', 'student_kd')
        flat_config['use_tensorboard'] = exp.get('use_tensorboard', True)
    
    # Top-level training flags
    flat_config['use_amp'] = config.get('use_amp', True)
    flat_config['use_ignite'] = config.get('use_ignite', True)
    
    return flat_config


def merge_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Merge YAML configuration with command-line arguments
    Command-line arguments take priority
    
    Args:
        config: YAML configuration dictionary
        args: Command-line arguments
    
    Returns:
        Merged configuration dictionary
    """
    # Extract special fields that need custom handling
    flat_config = _extract_special_fields(config)
    
    # Override with command-line arguments (if provided)
    # Command-line arguments have higher priority
    if hasattr(args, 'data_dir') and args.data_dir is not None:
        flat_config['data_dir'] = args.data_dir
    if hasattr(args, 'root_dir') and args.root_dir is not None:
        flat_config['root_dir'] = args.root_dir
    if hasattr(args, 'batch_size') and args.batch_size is not None:
        flat_config['batch_size'] = args.batch_size
    if hasattr(args, 'lr') and args.lr is not None:
        flat_config['lr'] = args.lr
    
    return flat_config


def parse_arguments() -> Tuple[argparse.Namespace, Dict[str, Any]]:
    """
    Parse command-line arguments and load configuration
    
    Returns:
        args: Command-line arguments
        config: Final merged configuration dictionary
    
    Example:
        >>> args, config = parse_arguments()
        >>> print(config['lr'])  # Learning rate from config or command line
    """
    parser = argparse.ArgumentParser(
        description="Training script with YAML config support\n"
                    "Command-line arguments override YAML configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Required: configuration file
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    
    # Optional overrides (these take priority over YAML config)
    parser.add_argument("--data_dir", type=str, default=None, 
                       help="Dataset root directory (overrides config)")
    parser.add_argument("--root_dir", type=str, default=None, 
                       help="Experiment output directory (overrides config)")
    parser.add_argument("--batch_size", type=int, default=None, 
                       help="Training batch size (overrides config)")
    parser.add_argument("--lr", type=float, default=None, 
                       help="Learning rate (overrides config)")
    parser.add_argument("--resume", type=str, default=None, 
                       help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Load and merge configuration
    yaml_config = load_config(args.config)
    final_config = merge_config_with_args(yaml_config, args)
    
    return args, final_config

