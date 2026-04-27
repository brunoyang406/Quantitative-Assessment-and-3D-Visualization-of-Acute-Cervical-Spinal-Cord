"""
Learning rate schedules: linear warmup + cosine annealing.

Used by training entry scripts with YAML `scheduler.type: cosine_warmup`.
"""

import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """
    Learning rate scheduler with linear warmup followed by cosine annealing.

    - Warmup: LR linearly increases from warmup_start_lr to each param group's base_lr.
    - Cosine: LR decays from base_lr to eta_min.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 1e-8,
        eta_min: float = 1e-6,
        last_epoch: int = -1,
    ):
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")
        if max_epochs <= warmup_epochs:
            raise ValueError(
                f"max_epochs ({max_epochs}) must be > warmup_epochs ({warmup_epochs})"
            )

        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < 0:
            return [self.warmup_start_lr for _ in self.optimizer.param_groups]
        if self.last_epoch < self.warmup_epochs:
            if self.warmup_epochs == 1:
                return [base_lr for base_lr in self.base_lrs]
            progress = self.last_epoch / (self.warmup_epochs - 1)
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * progress
                for base_lr in self.base_lrs
            ]
        progress = (self.last_epoch - self.warmup_epochs) / (
            self.max_epochs - self.warmup_epochs
        )
        progress = min(1.0, max(0.0, progress))
        return [
            self.eta_min + (base_lr - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))
            for base_lr in self.base_lrs
        ]


def get_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = "cosine_warmup",
    num_epochs: int = 300,
    warmup_epochs: int = 10,
    min_lr: float = 1e-6,
    **kwargs,
):
    """Build a learning rate scheduler by name."""
    if scheduler_type == "cosine_warmup":
        return LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=num_epochs,
            warmup_start_lr=min_lr,
            eta_min=min_lr,
        )
    raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
