"""
Learning rate schedulers for training.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LRScheduler,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
)


class WarmupCosineScheduler(LRScheduler):
    """
    Cosine annealing learning rate scheduler with linear warmup.
    
    Learning rate starts from a small value, increases linearly to base LR
    during warmup, then decreases following a cosine curve.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
        warmup_start_lr: float = 1e-7,
        last_epoch: int = -1,
    ):
        """
        Initialize scheduler.
        
        Args:
            optimizer: Wrapped optimizer.
            warmup_epochs: Number of warmup epochs.
            total_epochs: Total training epochs.
            min_lr: Minimum learning rate at end of schedule.
            warmup_start_lr: Learning rate at start of warmup.
            last_epoch: The index of the last epoch.
        """
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> list[float]:
        """Calculate learning rate for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [
                self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            return [
                self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


class WarmupLinearScheduler(LRScheduler):
    """
    Linear decay learning rate scheduler with warmup.
    
    LR increases linearly during warmup, then decreases linearly to min_lr.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
        warmup_start_lr: float = 1e-7,
        last_epoch: int = -1,
    ):
        """
        Initialize scheduler.
        
        Args:
            optimizer: Wrapped optimizer.
            warmup_epochs: Number of warmup epochs.
            total_epochs: Total training epochs.
            min_lr: Minimum learning rate at end of schedule.
            warmup_start_lr: Learning rate at start of warmup.
            last_epoch: The index of the last epoch.
        """
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> list[float]:
        """Calculate learning rate for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [
                self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                for base_lr in self.base_lrs
            ]
        else:
            # Linear decay
            progress = (self.last_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            return [
                base_lr - progress * (base_lr - self.min_lr)
                for base_lr in self.base_lrs
            ]


class WarmupExponentialScheduler(LRScheduler):
    """
    Exponential decay learning rate scheduler with warmup.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        gamma: float = 0.95,
        min_lr: float = 1e-6,
        warmup_start_lr: float = 1e-7,
        last_epoch: int = -1,
    ):
        """
        Initialize scheduler.
        
        Args:
            optimizer: Wrapped optimizer.
            warmup_epochs: Number of warmup epochs.
            gamma: Multiplicative factor of LR decay per epoch.
            min_lr: Minimum learning rate.
            warmup_start_lr: Learning rate at start of warmup.
            last_epoch: The index of the last epoch.
        """
        self.warmup_epochs = warmup_epochs
        self.gamma = gamma
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> list[float]:
        """Calculate learning rate for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [
                self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                for base_lr in self.base_lrs
            ]
        else:
            # Exponential decay
            decay_epochs = self.last_epoch - self.warmup_epochs
            return [
                max(self.min_lr, base_lr * (self.gamma ** decay_epochs))
                for base_lr in self.base_lrs
            ]


class PolynomialLRScheduler(LRScheduler):
    """
    Polynomial learning rate scheduler with optional warmup.
    
    LR = (base_lr - min_lr) * (1 - progress)^power + min_lr
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        total_epochs: int,
        power: float = 1.0,
        min_lr: float = 0.0,
        warmup_epochs: int = 0,
        warmup_start_lr: float = 1e-7,
        last_epoch: int = -1,
    ):
        """
        Initialize scheduler.
        
        Args:
            optimizer: Wrapped optimizer.
            total_epochs: Total training epochs.
            power: Power of the polynomial. Default 1.0 = linear.
            min_lr: Minimum learning rate.
            warmup_epochs: Number of warmup epochs.
            warmup_start_lr: Starting LR for warmup.
            last_epoch: The index of the last epoch.
        """
        self.total_epochs = total_epochs
        self.power = power
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> list[float]:
        """Calculate learning rate for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [
                self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                for base_lr in self.base_lrs
            ]
        else:
            # Polynomial decay
            effective_epoch = self.last_epoch - self.warmup_epochs
            effective_total = self.total_epochs - self.warmup_epochs
            progress = effective_epoch / effective_total
            
            decay_factor = (1 - progress) ** self.power
            return [
                (base_lr - self.min_lr) * decay_factor + self.min_lr
                for base_lr in self.base_lrs
            ]


def get_scheduler(
    name: str,
    optimizer: Optimizer,
    total_epochs: int,
    warmup_epochs: int = 5,
    min_lr: float = 1e-6,
    **kwargs: Any,
) -> LRScheduler | ReduceLROnPlateau:
    """
    Factory function to create learning rate scheduler.
    
    Args:
        name: Name of the scheduler.
        optimizer: Wrapped optimizer.
        total_epochs: Total training epochs.
        warmup_epochs: Number of warmup epochs.
        min_lr: Minimum learning rate.
        **kwargs: Additional arguments for specific schedulers.
    
    Returns:
        Learning rate scheduler.
    
    Supported schedulers:
        - "warmup_cosine": Cosine annealing with warmup
        - "warmup_linear": Linear decay with warmup
        - "warmup_exponential": Exponential decay with warmup
        - "polynomial": Polynomial decay with optional warmup
        - "cosine": Cosine annealing
        - "cosine_restarts": Cosine annealing with warm restarts
        - "step": Step decay
        - "reduce_on_plateau": Reduce on plateau
        - "one_cycle": One cycle policy
    """
    name = name.lower().replace("-", "_")
    
    if name == "warmup_cosine":
        return WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            min_lr=min_lr,
            warmup_start_lr=kwargs.get("warmup_start_lr", 1e-7),
        )
    
    elif name == "warmup_linear":
        return WarmupLinearScheduler(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            min_lr=min_lr,
            warmup_start_lr=kwargs.get("warmup_start_lr", 1e-7),
        )
    
    elif name == "warmup_exponential":
        return WarmupExponentialScheduler(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            gamma=kwargs.get("gamma", 0.95),
            min_lr=min_lr,
            warmup_start_lr=kwargs.get("warmup_start_lr", 1e-7),
        )
    
    elif name == "polynomial":
        return PolynomialLRScheduler(
            optimizer=optimizer,
            total_epochs=total_epochs,
            power=kwargs.get("power", 1.0),
            min_lr=min_lr,
            warmup_epochs=warmup_epochs,
            warmup_start_lr=kwargs.get("warmup_start_lr", 1e-7),
        )
    
    elif name == "cosine":
        return CosineAnnealingLR(
            optimizer=optimizer,
            T_max=total_epochs,
            eta_min=min_lr,
        )
    
    elif name == "cosine_restarts":
        return CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=kwargs.get("T_0", 10),
            T_mult=kwargs.get("T_mult", 2),
            eta_min=min_lr,
        )
    
    elif name == "step":
        return StepLR(
            optimizer=optimizer,
            step_size=kwargs.get("step_size", 10),
            gamma=kwargs.get("gamma", 0.1),
        )
    
    elif name == "reduce_on_plateau":
        return ReduceLROnPlateau(
            optimizer=optimizer,
            mode=kwargs.get("mode", "min"),
            factor=kwargs.get("factor", 0.1),
            patience=kwargs.get("patience", 10),
            min_lr=min_lr,
            verbose=kwargs.get("verbose", True),
        )
    
    elif name == "one_cycle":
        steps_per_epoch = kwargs.get("steps_per_epoch", 100)
        return OneCycleLR(
            optimizer=optimizer,
            max_lr=kwargs.get("max_lr", optimizer.defaults["lr"]),
            total_steps=total_epochs * steps_per_epoch,
            pct_start=kwargs.get("pct_start", 0.3),
            anneal_strategy=kwargs.get("anneal_strategy", "cos"),
            div_factor=kwargs.get("div_factor", 25),
            final_div_factor=kwargs.get("final_div_factor", 10000),
        )
    
    else:
        raise ValueError(f"Unknown scheduler: {name}")


class SchedulerWrapper:
    """
    Wrapper for schedulers that need step() called per batch or per epoch.
    
    Handles the distinction between epoch-based and step-based schedulers.
    """
    
    def __init__(
        self,
        scheduler: LRScheduler | ReduceLROnPlateau,
        step_on: str = "epoch",
        metric_name: str = "val_loss",
    ):
        """
        Initialize wrapper.
        
        Args:
            scheduler: Underlying scheduler.
            step_on: When to step - "epoch" or "batch".
            metric_name: Metric name for ReduceLROnPlateau.
        """
        self.scheduler = scheduler
        self.step_on = step_on
        self.metric_name = metric_name
        self._is_reduce_on_plateau = isinstance(scheduler, ReduceLROnPlateau)
    
    def step_batch(self) -> None:
        """Call after each batch if step_on="batch"."""
        if self.step_on == "batch" and not self._is_reduce_on_plateau:
            self.scheduler.step()
    
    def step_epoch(self, metrics: dict[str, float] | None = None) -> None:
        """Call after each epoch if step_on="epoch"."""
        if self.step_on == "epoch":
            if self._is_reduce_on_plateau:
                if metrics and self.metric_name in metrics:
                    self.scheduler.step(metrics[self.metric_name])
            else:
                self.scheduler.step()
    
    def get_last_lr(self) -> list[float]:
        """Get last computed learning rate."""
        if self._is_reduce_on_plateau:
            return [g["lr"] for g in self.scheduler.optimizer.param_groups]
        return self.scheduler.get_last_lr()
    
    def state_dict(self) -> dict:
        """Get scheduler state dict."""
        return self.scheduler.state_dict()
    
    def load_state_dict(self, state_dict: dict) -> None:
        """Load scheduler state dict."""
        self.scheduler.load_state_dict(state_dict)
