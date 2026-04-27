"""
Spinal cord lesion training (MultimodalLesionUNet, multimodal input, lesion head only).

Usage:
    python train_lesion_unet.py --config configs/multimodal_lesion_unet.yaml
"""

import os
import logging

# Set CUDA memory allocation configuration
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'max_split_size_mb:256,roundup_power2_divisions:4')

import torch
from monai.config import print_config

from models import create_model
from trainers.multimodal_lesion_unet_trainer_ignite import MultimodalLesionUnetIgniteTrainer
from data.multimodal_transforms import get_multimodal_lesion_unet_transforms
from data.dataloader import create_data_loaders
from utils import (
    get_scheduler,
    parse_arguments,
    setup_experiment_directory,
    set_random_seed,
)

def create_optimizer_and_scheduler(config, model):
    """Create optimizer and scheduler"""
    # Optimizer
    optimizer_config = config.get('optimizer', {})
    optimizer_type = optimizer_config.get('type', config.get('optimizer_type', 'AdamW'))
    lr = optimizer_config.get('lr', config.get('lr', 0.0002))
    weight_decay = optimizer_config.get('weight_decay', config.get('weight_decay', 1e-5))
    eps = optimizer_config.get('eps', 1e-6)  # Default to 1e-6 for AMP stability
    
    if optimizer_type == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            eps=eps
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            eps=eps
        )
    
    logging.info(f"Optimizer: {optimizer_type} (lr={lr}, weight_decay={weight_decay}, eps={eps})")
    
    # Learning rate scheduler
    scheduler_config = config.get('scheduler', {})
    scheduler_type = scheduler_config.get('type', 'cosine_warmup')
    
    max_epochs = config.get('max_epochs')
    if max_epochs is None:
        # Try num_epochs first (for linear scheduler), then max_epochs, then default
        max_epochs = scheduler_config.get('num_epochs') or scheduler_config.get('max_epochs', 300)
    
    warmup_epochs = scheduler_config.get('warmup_epochs', 10)
    min_lr = scheduler_config.get('min_lr', 1e-6)
    min_lr = float(min_lr) if isinstance(min_lr, str) else min_lr
    
    # For linear decay scheduler
    start_lr = scheduler_config.get('start_lr', 0.01)
    start_lr = float(start_lr) if isinstance(start_lr, str) else start_lr
    end_lr = scheduler_config.get('end_lr', 0.007)
    end_lr = float(end_lr) if isinstance(end_lr, str) else end_lr
    
    # For plateau scheduler
    patience = scheduler_config.get('patience', 10)
    factor = scheduler_config.get('factor', 0.5)
    
    scheduler = get_scheduler(
        optimizer=optimizer,
        scheduler_type=scheduler_type,
        num_epochs=max_epochs,
        warmup_epochs=warmup_epochs,
        min_lr=min_lr,
        start_lr=start_lr,
        end_lr=end_lr,
        patience=patience,
        factor=factor,
    )
    
    if scheduler_type == 'linear':
        logging.info(f"LR Scheduler: {scheduler_type} (start_lr={start_lr}, end_lr={end_lr}, max_epochs={max_epochs})")
    elif scheduler_type == 'plateau':
        logging.info(f"LR Scheduler: {scheduler_type} (warmup_epochs={warmup_epochs}, patience={patience}, factor={factor}, min_lr={min_lr})")
    else:
        logging.info(f"LR Scheduler: {scheduler_type} (warmup_epochs={warmup_epochs}, max_epochs={max_epochs}, min_lr={min_lr})")
    
    return optimizer, scheduler


def train_lesion_unet(config, args):
    """Train multimodal lesion U-Net (MultimodalLesionUNet)."""
    logging.info("="*70)
    logging.info(" Multimodal Lesion U-Net Training")
    logging.info("="*70)
    
    logging.info("\nConfiguration Summary:")
    logging.info(f"  Data directory: {config['data_dir']}")
    logging.info(f"  Output directory: {config['root_dir']}")
    logging.info(f"  Batch size: {config['batch_size']}")
    logging.info(f"  Learning rate: {config['lr']}")
    logging.info(f"  Max epochs: {config.get('max_epochs', 300)}")
    logging.info(f"  Random seed: {config['seed']}")
    
    # Set random seed
    set_random_seed(config['seed'])
    
    # Setup experiment directory (creates logging and TensorBoard writer)
    writer, experiment_dir = setup_experiment_directory(
        config['root_dir'],
        config.get('experiment_name', 'multimodal_lesion_unet'),
        config_path=args.config if args.config else None,
    )
    config['root_dir'] = experiment_dir
    
    # Create transforms
    logging.info("\n" + "="*70)
    logging.info(" Preparing Dataset and DataLoaders")
    logging.info("="*70)
    
    train_transforms, val_transforms = get_multimodal_lesion_unet_transforms(config)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_json=config['train_json'],
        val_json=config['val_json'],
        test_json=config['test_json'],
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        batch_size=config['batch_size'],
        val_batch_size=config['val_batch_size'],
        num_workers=config['num_workers'],
        val_num_workers=config['val_num_workers'],
        data_dir=config.get('data_dir'),
        uncertainty_boundary_dir=config.get('uncertainty_boundary_dir'),
    )
    
    max_epochs = config.get('max_epochs', 300)
    config['max_epochs'] = max_epochs
    
    # Create model
    logging.info("\n" + "="*70)
    logging.info(" Creating Multimodal Lesion U-Net")
    logging.info("="*70)
    
    model_config = config["model"].copy()
    model_config.setdefault("type", "multimodal_lesion_unet")
    
    model = create_model(
        model_config=model_config,
        img_size=config['spatial_size'],
    )
    
    # Multi-GPU training support
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            logging.info(f"Using {num_gpus} GPUs for training:")
            for i in range(num_gpus):
                logging.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            model = torch.nn.DataParallel(model)
            logging.info("Model wrapped with DataParallel for multi-GPU training")
        else:
            logging.info(f"Using single GPU: {torch.cuda.get_device_name(0)}")
    else:
        logging.info("Using CPU training")
    
    # Setup optimizer and scheduler
    logging.info("\n" + "="*70)
    logging.info("Initializing Optimizer and Scheduler")
    logging.info("="*70)
    
    optimizer, scheduler = create_optimizer_and_scheduler(config, model)
    
    # Update scheduler with correct max_epochs if it was created
    if scheduler is not None and hasattr(scheduler, 'max_epochs'):
        scheduler.max_epochs = max_epochs
        logging.info(f"Updated scheduler with max_epochs={max_epochs}")
    
    # Create trainer
    logging.info("\n" + "="*70)
    logging.info("Initializing Multimodal Lesion U-Net Trainer")
    logging.info("="*70)
    
    requested_device = str(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')).lower()
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Config requests device='cuda' but this Python environment has no CUDA-enabled PyTorch. "
            "Install a CUDA build of torch/torchvision/torchaudio from the official PyTorch CUDA index URL."
        )
    device = requested_device
    logging.info(f"Using device from config: {device}")
    logging.info("Using Ignite-based trainer")
    
    trainer = MultimodalLesionUnetIgniteTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        writer=writer,
        root_dir=experiment_dir,
        device=device,
    )
    
    # Load checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        trainer.trainer.state.epoch = trainer.start_epoch
    
    # Start training
    trainer.train(max_epochs=max_epochs)
    
    logging.info("\n" + "="*70)
    logging.info("Training completed!")
    logging.info("="*70)
    logging.info(f"Best Lesion Dice: {trainer.best_lesion_dice:.4f}")
    logging.info(f"Experiment directory: {experiment_dir}")


def main():
    """Main entry point"""
    # Keep MONAI environment print in main guard so DataLoader worker imports
    # do not spam repeated dependency/version logs.
    print_config()
    args, config = parse_arguments()
    
    # Merge command line arguments into config (already done in parse_arguments, but ensure)
    if args.lr is not None:
        config['lr'] = args.lr
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    
    train_lesion_unet(config, args)


if __name__ == "__main__":
    main()

