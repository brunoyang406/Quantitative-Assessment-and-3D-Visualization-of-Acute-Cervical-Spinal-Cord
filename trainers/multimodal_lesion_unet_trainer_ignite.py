"""
Multimodal Lesion U-Net trainer (MultimodalLesionUNet) using PyTorch Ignite.

Multi-modal input: T1, T2, T2FS, cord_mask. Lesion segmentation only (binary, 1 channel).
Uses DiceBCELoss / DiceBCEDeepSupervisionLoss when deep supervision is enabled.
"""

import os
import logging
from typing import Dict, Optional, Any

import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

try:
    import ignite
    from ignite.engine import Engine, Events
    from ignite.handlers import ModelCheckpoint
    from ignite.metrics import RunningAverage
except ImportError:
    raise ImportError(
        "PyTorch Ignite is required for Ignite-based trainer. "
        "Install it with: pip install pytorch-ignite"
    )
from monai.handlers import (
    TensorBoardStatsHandler,
)
from monai.transforms import AsDiscrete


class MultimodalLesionUnetIgniteTrainer:
    """
    Training loop for MultimodalLesionUNet (lesion head) via Ignite.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        config: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: DataLoader,
        writer: Optional[SummaryWriter] = None,
        root_dir: str = "./experiments",
        device: str = "cuda",
    ):
        """
        Initialize MultimodalLesionUnetIgniteTrainer
        
        Args:
            model: Multimodal lesion U-Net
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            config: Configuration dictionary
            train_loader: Training data loader
            val_loader: Validation data loader
            writer: TensorBoard writer (optional)
            root_dir: Root directory for saving models and logs
            device: Device to use ('cuda' or 'cpu')
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.writer = writer
        self.root_dir = root_dir
        self.device = device
        
        self.use_amp = config.get('use_amp', True)
        if self.use_amp:
            self.scaler = torch.amp.GradScaler("cuda")
            logging.info("AMP (mixed precision) enabled")
        
        self.gradient_clip_norm = config.get('gradient_clip_norm', None)
        if self.gradient_clip_norm:
            logging.info("Gradient clipping max_norm=%s", self.gradient_clip_norm)
        
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        self._accumulation_counter = 0
        if self.gradient_accumulation_steps > 1:
            bs = config.get('batch_size', 4)
            eff = bs * self.gradient_accumulation_steps
            logging.info(
                "Gradient accumulation steps=%s (effective batch size %s)",
                self.gradient_accumulation_steps,
                eff,
            )
        
        from losses.dice_bce_loss import DiceBCELoss, DiceBCEDeepSupervisionLoss

        loss_config = config.get('loss', {})
        lesion_loss_config = loss_config.get('lesion', {})
        model_config = config.get('model', {})
        self.use_deep_supervision = model_config.get('deep_supervision', False)

        if self.use_deep_supervision:
            ds_weights = lesion_loss_config.get('ds_weights', None)
            self.criterion = DiceBCEDeepSupervisionLoss(
                dice_weight=lesion_loss_config.get('dice_weight', 0.7),
                ce_weight=lesion_loss_config.get('ce_weight', 0.3),
                ds_weights=ds_weights,
                sigmoid=lesion_loss_config.get('sigmoid', True),
            ).to(device)
            logging.info("Loss: Dice+BCE with deep supervision")
            if ds_weights:
                logging.info(f"  ds_weights: {ds_weights}")
        else:
            self.criterion = DiceBCELoss(
                dice_weight=lesion_loss_config.get('dice_weight', 0.7),
                ce_weight=lesion_loss_config.get('ce_weight', 0.3),
                sigmoid=lesion_loss_config.get('sigmoid', True),
            ).to(device)
            logging.info("Loss: Dice+BCE (single scale)")
        
        # Metrics for validation (binary classification)
        # For binary segmentation: directly compute Dice without one-hot conversion
        # Target: (B, 1, D, H, W) or (B, D, H, W) with values 0 or 1
        # Prediction: (B, 1, D, H, W) logits -> sigmoid -> threshold -> binary mask
        self.post_pred_lesion = AsDiscrete(threshold=0.5)  # Simple threshold for binary prediction
        # Accumulate intersection and union for Dice calculation across all batches
        self._dice_intersection = 0.0
        self._dice_union = 0.0
        
        # Best model tracking
        self.best_lesion_dice = 0.0
        self.start_epoch = 0
        
        # Create Ignite engines
        self.trainer = Engine(self._train_step)
        self.evaluator = Engine(self._val_step)
        
        # Setup handlers
        self._setup_handlers()
        
        logging.info("MultimodalLesionUnetIgniteTrainer initialized")
        logging.info(f"Device: {device}")
        logging.info(f"Output: Binary segmentation (1 channel)")
        logging.info(f"Loss type: Dice+BCE")
        logging.info(f"Loss weights: Dice={lesion_loss_config.get('dice_weight', 0.7)}, BCE={lesion_loss_config.get('ce_weight', 0.3)}")
    
    def _train_step(self, engine: Engine, batch: Dict) -> Dict[str, float]:
        """One training step; supports AMP, gradient accumulation, and optional grad clip."""
        self.model.train()
        
        inputs = batch["image"].to(self.device)
        lesion_labels = batch["lesion_mask"].to(self.device)
        
        if self._accumulation_counter == 0:
            self.optimizer.zero_grad()
        
        if self.use_amp:
            # Mixed Precision Training
            with torch.amp.autocast('cuda'):
                lesion_output = self.model(inputs)
                
                # Compute loss
                if self.use_deep_supervision:
                    loss = self.criterion(lesion_output, lesion_labels)
                else:
                    loss = self.criterion(lesion_output, lesion_labels)
                
                loss = loss / self.gradient_accumulation_steps
            
            self.scaler.scale(loss).backward()
            
            loss_val = loss.item()
            del lesion_output, inputs, lesion_labels, loss
            
            self._accumulation_counter += 1
            
            if self._accumulation_counter >= self.gradient_accumulation_steps:
                if self.gradient_clip_norm:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=self.gradient_clip_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self._accumulation_counter = 0
        else:
            lesion_output = self.model(inputs)
            
            if self.use_deep_supervision:
                loss = self.criterion(lesion_output, lesion_labels)
            else:
                loss = self.criterion(lesion_output, lesion_labels)
            
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            
            loss_val = loss.item()
            del lesion_output, inputs, lesion_labels, loss
            
            self._accumulation_counter += 1
            
            if self._accumulation_counter >= self.gradient_accumulation_steps:
                if self.gradient_clip_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=self.gradient_clip_norm
                    )
                
                self.optimizer.step()
                self._accumulation_counter = 0
        
        return {
            'loss': loss_val * self.gradient_accumulation_steps,
        }
    
    def _val_step(self, engine: Engine, batch: Dict) -> Dict[str, float]:
        """
        Validation step function for Ignite
        
        Args:
            engine: Ignite engine
            batch: Batch data dictionary
        
        Returns:
            Dictionary with metric values
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get multimodal input (T1, T2, T2FS, cord_mask - 4 channels)
            inputs = batch["image"].to(self.device)  # (B, 4, D, H, W)
            
            # Get lesion target (binary mask: 0 or 1)
            lesion_labels = batch["lesion_mask"].to(self.device)  # (B, 1, D, H, W) or (B, D, H, W)
            
            # Forward pass
            lesion_output = self.model(inputs)
            
            # Compute loss
            if self.use_deep_supervision:
                # lesion_output is a list: [main_output, ds_output_1, ds_output_2, ...]
                loss = self.criterion(lesion_output, lesion_labels)
                # Use main output (highest resolution) for metrics
                lesion_logits = lesion_output[0]
            else:
                # lesion_output is a single tensor: (B, 1, D, H, W)
                loss = self.criterion(lesion_output, lesion_labels)
                lesion_logits = lesion_output
            
            # Compute metrics for binary segmentation
            # Apply sigmoid to logits to get probabilities, then threshold
            lesion_probs = torch.sigmoid(lesion_logits)  # (B, 1, D, H, W)
            lesion_preds = self.post_pred_lesion(lesion_probs)  # (B, 1, D, H, W) binary mask
            
            # Ensure labels and predictions have same shape
            if lesion_labels.dim() == 4:
                lesion_labels = lesion_labels.unsqueeze(1)  # (B, D, H, W) -> (B, 1, D, H, W)
            
            # Accumulate intersection and union across all batches for accurate Dice calculation
            # Dice = 2 * |pred ∩ gt| / (|pred| + |gt|)
            intersection = (lesion_preds * lesion_labels).sum().item()  # Sum over all spatial dims and batch
            union = lesion_preds.sum().item() + lesion_labels.sum().item()  # Sum over all spatial dims and batch
            
            # Accumulate for final Dice calculation at epoch end
            self._dice_intersection += intersection
            self._dice_union += union
            
            loss_val = loss.item()
            
            del inputs, lesion_labels, lesion_output, lesion_logits, lesion_probs, lesion_preds, loss
            
            return {
                'loss': loss_val,
            }
    
    def _setup_handlers(self):
        """Setup Ignite handlers for training and validation"""
        # Training handlers
        RunningAverage(output_transform=lambda x: x['loss']).attach(self.trainer, 'avg_loss')
        
        # Validation handlers
        @self.evaluator.on(Events.EPOCH_STARTED)
        def reset_dice_accumulators(engine):
            # Reset accumulators at the start of each validation epoch
            self._dice_intersection = 0.0
            self._dice_union = 0.0
        
        @self.evaluator.on(Events.EPOCH_COMPLETED)
        def compute_metrics(engine):
            # Compute Dice from accumulated intersection and union across all batches
            if self._dice_union > 0:
                lesion_dice_value = 2.0 * self._dice_intersection / (self._dice_union + 1e-8)
            else:
                lesion_dice_value = 0.0
            
            engine.state.metrics['lesion_dice'] = lesion_dice_value
            engine.state.metrics['lesion_dice_per_class'] = np.array([lesion_dice_value])  # Single value for binary
            
            # Log to TensorBoard
            if self.writer:
                epoch = getattr(self, '_current_val_epoch', engine.state.epoch)
                self.writer.add_scalar('val/lesion_dice', lesion_dice_value, epoch)
        
        # Logging handlers
        if self.writer:
            # Use log_dir instead of writer for TensorBoardStatsHandler
            # Note: global_epoch_transform receives epoch (int), not engine
            tensorboard_dir = os.path.join(self.root_dir, "tensorboard")
            # Training metrics: loss
            TensorBoardStatsHandler(
                log_dir=tensorboard_dir,
                output_transform=lambda x: {'train/loss': x['loss']},
                global_epoch_transform=lambda epoch: epoch,
            ).attach(self.trainer)
            
            # Validation metrics: loss (dice metrics logged manually in compute_metrics)
            TensorBoardStatsHandler(
                log_dir=tensorboard_dir,
                output_transform=lambda x: {'val/loss': x['loss']},
                global_epoch_transform=lambda epoch: epoch,
            ).attach(self.evaluator)
        
        # Model checkpointing
        weights_dir = os.path.join(self.root_dir, "weights")
        os.makedirs(weights_dir, exist_ok=True)
        
        # Best model checkpoint
        def score_function(engine):
            return engine.state.metrics.get('lesion_dice', 0.0)
        
        best_model_handler = ModelCheckpoint(
            weights_dir,
            "best_model",
            n_saved=1,
            score_function=score_function,
            score_name="lesion_dice",
            create_dir=True,
            require_empty=False,
        )
        to_save = {"model": self.model, "optimizer": self.optimizer}
        if self.scheduler is not None:
            to_save["scheduler"] = self.scheduler
        
        self.evaluator.add_event_handler(
            Events.EPOCH_COMPLETED,
            best_model_handler,
            to_save,
        )
        
        # Regular checkpoint (every N epochs)
        checkpoint_interval = self.config.get('checkpoint_interval', 10)
        if checkpoint_interval > 0:
            checkpoint_handler = ModelCheckpoint(
                weights_dir,
                "checkpoint",
                n_saved=3,
                create_dir=True,
                require_empty=False,
            )
            checkpoint_to_save = {"model": self.model, "optimizer": self.optimizer}
            if self.scheduler is not None:
                checkpoint_to_save["scheduler"] = self.scheduler
            
            self.evaluator.add_event_handler(
                Events.EPOCH_COMPLETED(every=checkpoint_interval),
                checkpoint_handler,
                checkpoint_to_save,
            )
        
        # Validation logging
        @self.evaluator.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            # Use stored epoch from trainer or engine state epoch
            epoch = getattr(self, '_current_val_epoch', engine.state.epoch)
            lesion_dice = engine.state.metrics.get('lesion_dice', 0.0)
            lesion_dice_per_class = engine.state.metrics.get('lesion_dice_per_class', np.array([0.0]))
            
            logging.info("")
            logging.info("="*70)
            logging.info(f"Validation Results (Epoch: {epoch})")
            logging.info("="*70)
            logging.info("")
            logging.info("Lesion Segmentation (Binary):")
            logging.info(f"  Lesion Dice Score: {lesion_dice:.4f}")
            logging.info("")
            logging.info("="*70)
            logging.info("")
            
            # Update best model
            if lesion_dice > self.best_lesion_dice:
                self.best_lesion_dice = lesion_dice
                logging.info("New best lesion Dice: %.4f", lesion_dice)
                if self.writer:
                    self.writer.add_scalar('best_lesion_dice', lesion_dice, epoch)
            else:
                logging.info("Best lesion Dice so far: %.4f", self.best_lesion_dice)
    
    def train(self, max_epochs: int):
        """
        Start training
        
        Args:
            max_epochs: Maximum number of epochs
        """
        # Update scheduler max_epochs if needed
        if self.scheduler is not None and hasattr(self.scheduler, 'max_epochs'):
            self.scheduler.max_epochs = max_epochs
        
        # Add progress bar using Ignite's ProgressBar
        from ignite.contrib.handlers import ProgressBar
        
        pbar = ProgressBar()
        pbar.attach(self.trainer, output_transform=lambda x: {'loss': x['loss']})
        
        # Training loop
        @self.trainer.on(Events.EPOCH_STARTED)
        def log_epoch_start(engine):
            logging.info(f"\nEpoch[{engine.state.epoch}/{max_epochs}]")
        
        @self.trainer.on(Events.EPOCH_COMPLETED)
        def log_epoch_complete(engine):
            avg_loss = engine.state.metrics.get('avg_loss', 0.0)
            logging.info(f"Epoch[{engine.state.epoch}] Complete - Avg Loss: {avg_loss:.4f}, Time: {engine.state.times['EPOCH_COMPLETED']:.1f}s")
        
        # Validation after each epoch
        @self.trainer.on(Events.EPOCH_COMPLETED)
        def run_validation(engine):
            current_epoch = engine.state.epoch
            # Set evaluator epoch to match trainer epoch for proper logging
            self.evaluator.state.epoch = current_epoch
            # Store current epoch for use in validation handlers
            self._current_val_epoch = current_epoch
            # Reset max_epochs to avoid conflict with previous state
            self.evaluator.state.max_epochs = None
            self.evaluator.run(self.val_loader, max_epochs=1)
        
        # Learning rate scheduler handler (run after validation)
        @self.trainer.on(Events.EPOCH_COMPLETED)
        def update_lr(engine):
            if self.scheduler is not None:
                if hasattr(self.scheduler, 'step'):
                    try:
                        self.scheduler.step()
                    except Exception as e:
                        logging.warning(f"Error updating learning rate: {e}")
                
                current_lr = self.optimizer.param_groups[0]['lr']
                if self.writer:
                    self.writer.add_scalar('learning_rate', current_lr, engine.state.epoch)
        
        logging.info("")
        logging.info("="*70)
        logging.info("Starting Multimodal Lesion U-Net Training (Ignite)")
        logging.info("="*70)
        logging.info(f"Input modalities: T1, T2, T2FS, cord_mask (4 channels)")
        logging.info(f"Note: cord_mask is used for BOTH ROI cropping and as an input channel")
        logging.info(f"Output task: Lesion segmentation (binary, 1 channel)")
        logging.info(f"Max epochs: {max_epochs}")
        logging.info(f"Validation interval: every 1 epochs")
        logging.info("")
        
        self.trainer.run(self.train_loader, max_epochs=max_epochs)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model (and optimizer/scheduler if present)."""
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        # Handle DataParallel prefix mismatch (module.)
        curr_state_dict = self.model.state_dict()
        new_state_dict = {}
        
        curr_has_module = any(k.startswith('module.') for k in curr_state_dict.keys())
        ckpt_has_module = any(k.startswith('module.') for k in state_dict.keys())
        
        if curr_has_module and not ckpt_has_module:
            # Model is DataParallel, but checkpoint is not
            new_state_dict = {f"module.{k}": v for k, v in state_dict.items()}
            logging.info("  Added 'module.' prefix to state_dict for DataParallel model")
        elif not curr_has_module and ckpt_has_module:
            # Model is not DataParallel, but checkpoint is
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            logging.info("  Stripped 'module.' prefix from state_dict for single-device model")
        else:
            new_state_dict = state_dict
            
        try:
            self.model.load_state_dict(new_state_dict)
            logging.info("  Model state loaded")
        except RuntimeError as e:
            logging.warning("  Strict load failed, retrying non-strict: %s", e)
            self.model.load_state_dict(new_state_dict, strict=False)
            logging.info("  Model state loaded (non-strict)")
        
        # Load optimizer state
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            logging.info("  Optimizer state loaded")
        
        if 'scheduler' in checkpoint and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                logging.info("  Scheduler state loaded (last_epoch=%s)", self.scheduler.last_epoch)
            except Exception as e:
                logging.warning("  Failed to load scheduler state: %s", e)
                logging.warning("  Scheduler will restart; LR schedule may be discontinuous.")
        elif self.scheduler is not None:
            logging.warning("  No scheduler state in checkpoint; warmup/cosine may restart from epoch 0.")
        
        # Load epoch
        if 'epoch' in checkpoint:
            self.start_epoch = checkpoint['epoch']
            logging.info("  Checkpoint epoch: %s", self.start_epoch)
        else:
            import re
            match = re.search(r'epoch[_\s]*(\d+)', checkpoint_path)
            if match:
                self.start_epoch = int(match.group(1))
                logging.info("  Inferred epoch from filename: %s", self.start_epoch)
            else:
                self.start_epoch = 0
                logging.warning("  Could not infer epoch; using 0")
        
        if self.scheduler is not None and self.start_epoch > 0 and 'scheduler' not in checkpoint:
            logging.warning("  Stepping scheduler %s times to align with start epoch", self.start_epoch)
            for _ in range(self.start_epoch):
                self.scheduler.step()
            logging.info(
                "  Scheduler stepped to start_epoch=%s, lr=%.2e",
                self.start_epoch,
                self.optimizer.param_groups[0]['lr'],
            )
        
        logging.info(f"Checkpoint loaded successfully")


SingleTaskLesionIgniteTrainer = MultimodalLesionUnetIgniteTrainer
