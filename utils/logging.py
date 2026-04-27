"""
Logging and experiment setup utilities

This module provides:
- Logging setup (console + file)
- Experiment directory creation
- TensorBoard writer initialization
"""

import os
import sys
import logging
from datetime import datetime
from typing import Tuple, Optional

from torch.utils.tensorboard import SummaryWriter


def setup_logging(
    log_dir: str,
    experiment_name: str,
    level: int = logging.INFO,
    clear_existing: bool = True,
    format_string: Optional[str] = None
) -> str:
    """
    Setup logging to both console and file
    
    Args:
        log_dir: Directory to save log files
        experiment_name: Name of experiment for log filename
        level: Logging level (default: logging.INFO)
        clear_existing: Whether to clear existing handlers (default: True)
        format_string: Custom format string (default: standard format)
    
    Returns:
        Path to log file
    
    Example:
        >>> log_file = setup_logging('./logs', 'experiment')
        >>> logging.info("This will be logged to both console and file")
    """
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Clear existing handlers if requested
    if clear_existing:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
    
    # Create formatter
    if format_string is None:
        format_string = '%(asctime)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(format_string)
    
    # Add file handler (only if not already exists)
    has_file_handler = any(
        isinstance(h, logging.FileHandler) and h.baseFilename == log_filename
        for h in logger.handlers
    )
    if not has_file_handler:
        file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add console handler (only if not already exists)
    has_console_handler = any(
        isinstance(h, logging.StreamHandler) and h.stream == sys.stdout
        for h in logger.handlers
    )
    if not has_console_handler:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    logging.info(f"Log file: {log_filename}")
    return log_filename


def setup_experiment_directory(
    root_dir: str,
    experiment_name: str,
    config_path: Optional[str] = None,
    use_tensorboard: bool = True,
    create_subdirs: bool = True
) -> Tuple[Optional[SummaryWriter], str]:
    """
    Create experiment directory with timestamp and initialize TensorBoard writer
    
    This function:
    1. Creates experiment directory with timestamp
    2. Sets up logging (console + file)
    3. Copies code/config for reproducibility
    4. Optionally initializes TensorBoard writer
    5. Optionally creates standard subdirectories (weights, checkpoints, etc.)
    
    Args:
        root_dir: Root directory for experiments
        experiment_name: Name of experiment
        config_path: Path to config file (for copying, optional)
        use_tensorboard: Whether to create TensorBoard writer (default: True)
        create_subdirs: Whether to create standard subdirectories (default: True)
    
    Returns:
        writer: TensorBoard writer (None if use_tensorboard=False)
        full_root_dir: Full path to experiment directory
    
    Example:
        >>> writer, exp_dir = setup_experiment_directory(
        ...     './experiments', 'my_experiment', config_path='config.yaml'
        ... )
        >>> # Use writer for logging metrics
        >>> writer.add_scalar('loss', 0.5, 0)
    """
    try:
        from .misc import copy_code_to_experiment
    except ImportError:
        copy_code_to_experiment = None
        logging.warning("copy_code_to_experiment not available, skipping code copy")
    
    # Create experiment directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_root_dir = os.path.join(root_dir, f"{experiment_name}_{timestamp}")
    
    try:
        os.makedirs(full_root_dir, exist_ok=True)
        logging.info(f"Experiment directory: {full_root_dir}")
    except OSError as e:
        logging.error(f"Failed to create experiment directory: {e}")
        raise
    
    # Setup logging (only if not already set up)
    log_dir = os.path.join(full_root_dir, "logs")
    try:
        log_file = setup_logging(log_dir, experiment_name, clear_existing=False)
        logging.info(f"Logging initialized: {log_file}")
    except Exception as e:
        logging.warning(f"Failed to setup logging: {e}")
    
    # Copy code/config for reproducibility
    if config_path and copy_code_to_experiment:
        try:
            copy_code_to_experiment(full_root_dir, config_path)
            logging.info(f"Code/config copied to experiment directory")
        except Exception as e:
            logging.warning(f"Failed to copy code/config: {e}")
    
    # Create standard subdirectories
    if create_subdirs:
        subdirs = ['weights', 'checkpoints', 'predictions']
        for subdir in subdirs:
            subdir_path = os.path.join(full_root_dir, subdir)
            try:
                os.makedirs(subdir_path, exist_ok=True)
            except OSError as e:
                logging.warning(f"Failed to create subdirectory {subdir}: {e}")
    
    # Initialize TensorBoard (optional)
    writer = None
    if use_tensorboard:
        try:
            tensorboard_dir = os.path.join(full_root_dir, "tensorboard")
            writer = SummaryWriter(log_dir=tensorboard_dir)
            logging.info(f"TensorBoard directory: {tensorboard_dir}")
        except Exception as e:
            logging.warning(f"Failed to initialize TensorBoard: {e}")
            writer = None
    
    return writer, full_root_dir

