"""
Training callbacks for monitoring and controlling training.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """Current training state passed to callbacks."""
    
    epoch: int = 0
    global_step: int = 0
    train_loss: float = 0.0
    val_loss: float | None = None
    train_metrics: dict[str, float] = field(default_factory=dict)
    val_metrics: dict[str, float] = field(default_factory=dict)
    learning_rate: float = 0.0
    is_best: bool = False


class Callback(ABC):
    """Base class for training callbacks."""
    
    def on_train_start(self, trainer: "Trainer") -> None:
        """Called at the start of training."""
        pass
    
    def on_train_end(self, trainer: "Trainer") -> None:
        """Called at the end of training."""
        pass
    
    def on_epoch_start(self, trainer: "Trainer", state: TrainingState) -> None:
        """Called at the start of each epoch."""
        pass
    
    def on_epoch_end(self, trainer: "Trainer", state: TrainingState) -> None:
        """Called at the end of each epoch."""
        pass
    
    def on_batch_start(self, trainer: "Trainer", batch: Any, batch_idx: int) -> None:
        """Called at the start of each batch."""
        pass
    
    def on_batch_end(self, trainer: "Trainer", batch: Any, batch_idx: int, loss: float) -> None:
        """Called at the end of each batch."""
        pass
    
    def on_validation_start(self, trainer: "Trainer") -> None:
        """Called at the start of validation."""
        pass
    
    def on_validation_end(self, trainer: "Trainer", state: TrainingState) -> None:
        """Called at the end of validation."""
        pass


class EarlyStopping(Callback):
    """
    Early stopping callback to prevent overfitting.
    
    Monitors a metric and stops training if it doesn't improve
    for a specified number of epochs.
    """
    
    def __init__(
        self,
        monitor: str = "val_loss",
        min_delta: float = 1e-4,
        patience: int = 10,
        mode: str = "min",
        verbose: bool = True,
    ):
        """
        Initialize early stopping.
        
        Args:
            monitor: Metric to monitor.
            min_delta: Minimum change to qualify as improvement.
            patience: Number of epochs without improvement before stopping.
            mode: "min" or "max" for metric optimization direction.
            verbose: Whether to print messages.
        """
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.verbose = verbose
        
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.should_stop = False
    
    def _is_improvement(self, current: float) -> bool:
        """Check if current value is an improvement."""
        if self.mode == "min":
            return current < self.best_value - self.min_delta
        return current > self.best_value + self.min_delta
    
    def on_epoch_end(self, trainer: "Trainer", state: TrainingState) -> None:
        """Check for improvement at end of epoch."""
        # Get monitored value
        if self.monitor == "val_loss":
            current = state.val_loss
        elif self.monitor in state.val_metrics:
            current = state.val_metrics[self.monitor]
        elif self.monitor == "train_loss":
            current = state.train_loss
        else:
            current = state.train_metrics.get(self.monitor)
        
        if current is None:
            return
        
        if self._is_improvement(current):
            self.best_value = current
            self.counter = 0
            state.is_best = True
        else:
            self.counter += 1
            
            if self.verbose:
                logger.info(
                    f"EarlyStopping: {self.monitor} did not improve. "
                    f"Counter: {self.counter}/{self.patience}"
                )
            
            if self.counter >= self.patience:
                self.should_stop = True
                trainer.should_stop = True
                
                if self.verbose:
                    logger.info(
                        f"EarlyStopping: Stopping training. "
                        f"Best {self.monitor}: {self.best_value:.6f}"
                    )


class ModelCheckpoint(Callback):
    """
    Save model checkpoints during training.
    
    Can save:
    - Best model based on monitored metric
    - Latest model at each epoch
    - Top-k best models
    """
    
    def __init__(
        self,
        save_dir: str | Path,
        monitor: str = "val_loss",
        mode: str = "min",
        save_top_k: int = 3,
        save_last: bool = True,
        filename_format: str = "epoch={epoch:03d}-{monitor}={value:.4f}",
        verbose: bool = True,
    ):
        """
        Initialize checkpoint callback.
        
        Args:
            save_dir: Directory to save checkpoints.
            monitor: Metric to monitor for best model selection.
            mode: "min" or "max" for metric optimization.
            save_top_k: Number of best models to keep.
            save_last: Whether to always save the last model.
            filename_format: Format string for checkpoint filenames.
            verbose: Whether to print save messages.
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last
        self.filename_format = filename_format
        self.verbose = verbose
        
        # Track best checkpoints: [(value, path), ...]
        self.best_checkpoints: list[tuple[float, Path]] = []
        self.best_value = float("inf") if mode == "min" else float("-inf")
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current value is better than best."""
        if self.mode == "min":
            return current < best
        return current > best
    
    def _save_checkpoint(
        self,
        trainer: "Trainer",
        state: TrainingState,
        path: Path,
    ) -> None:
        """Save checkpoint to path."""
        checkpoint = {
            "epoch": state.epoch,
            "global_step": state.global_step,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "train_loss": state.train_loss,
            "val_loss": state.val_loss,
            "train_metrics": state.train_metrics,
            "val_metrics": state.val_metrics,
            "config": trainer.config.model_dump() if hasattr(trainer, "config") else {},
        }
        
        if trainer.scheduler is not None:
            checkpoint["scheduler_state_dict"] = trainer.scheduler.state_dict()
        
        if trainer.scaler is not None:
            checkpoint["scaler_state_dict"] = trainer.scaler.state_dict()
        
        torch.save(checkpoint, path)
        
        if self.verbose:
            logger.info(f"Saved checkpoint: {path}")
    
    def on_epoch_end(self, trainer: "Trainer", state: TrainingState) -> None:
        """Save checkpoint at end of epoch if needed."""
        # Get monitored value
        if self.monitor == "val_loss":
            current = state.val_loss
        elif self.monitor in state.val_metrics:
            current = state.val_metrics[self.monitor]
        else:
            current = state.train_loss
        
        if current is None:
            return
        
        # Check if this is a good checkpoint
        should_save = False
        
        if len(self.best_checkpoints) < self.save_top_k:
            should_save = True
        elif self._is_better(current, self.best_checkpoints[-1][0]):
            should_save = True
            
            # Remove worst checkpoint
            _, worst_path = self.best_checkpoints.pop()
            if worst_path.exists():
                worst_path.unlink()
        
        if should_save:
            # Create filename
            filename = self.filename_format.format(
                epoch=state.epoch,
                monitor=self.monitor,
                value=current,
            )
            path = self.save_dir / f"{filename}.pt"
            
            self._save_checkpoint(trainer, state, path)
            
            # Update best checkpoints list
            self.best_checkpoints.append((current, path))
            self.best_checkpoints.sort(
                key=lambda x: x[0],
                reverse=(self.mode == "max"),
            )
            
            # Update best value
            if self._is_better(current, self.best_value):
                self.best_value = current
                
                # Save as best model
                best_path = self.save_dir / "best_model.pt"
                self._save_checkpoint(trainer, state, best_path)
        
        # Save last model
        if self.save_last:
            last_path = self.save_dir / "last_model.pt"
            self._save_checkpoint(trainer, state, last_path)


class LearningRateMonitor(Callback):
    """Monitor and log learning rate during training."""
    
    def __init__(self, log_every_n_steps: int = 10):
        """
        Initialize LR monitor.
        
        Args:
            log_every_n_steps: How often to log LR.
        """
        self.log_every_n_steps = log_every_n_steps
        self.lr_history: list[tuple[int, float]] = []
    
    def on_batch_end(
        self,
        trainer: "Trainer",
        batch: Any,
        batch_idx: int,
        loss: float,
    ) -> None:
        """Log learning rate."""
        if trainer.global_step % self.log_every_n_steps == 0:
            lr = trainer.optimizer.param_groups[0]["lr"]
            self.lr_history.append((trainer.global_step, lr))


class MetricsLogger(Callback):
    """
    Log training metrics to file.
    
    Saves metrics history to JSON file for later analysis.
    """
    
    def __init__(
        self,
        log_dir: str | Path,
        filename: str = "metrics.json",
    ):
        """
        Initialize metrics logger.
        
        Args:
            log_dir: Directory for log files.
            filename: Name of metrics file.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / filename
        
        self.history: list[dict] = []
    
    def on_epoch_end(self, trainer: "Trainer", state: TrainingState) -> None:
        """Log metrics at end of epoch."""
        record = {
            "epoch": state.epoch,
            "global_step": state.global_step,
            "train_loss": state.train_loss,
            "val_loss": state.val_loss,
            "learning_rate": state.learning_rate,
            "train_metrics": state.train_metrics,
            "val_metrics": state.val_metrics,
            "timestamp": time.time(),
        }
        
        self.history.append(record)
        
        # Save to file
        with open(self.log_path, "w") as f:
            json.dump(self.history, f, indent=2)
    
    def on_train_end(self, trainer: "Trainer") -> None:
        """Final save at end of training."""
        with open(self.log_path, "w") as f:
            json.dump(self.history, f, indent=2)


class ProgressCallback(Callback):
    """
    Display training progress with rich formatting.
    """
    
    def __init__(self, show_metrics: list[str] | None = None):
        """
        Initialize progress callback.
        
        Args:
            show_metrics: List of metrics to display.
        """
        self.show_metrics = show_metrics or ["accuracy", "f1"]
        self.epoch_start_time: float = 0.0
    
    def on_epoch_start(self, trainer: "Trainer", state: TrainingState) -> None:
        """Record epoch start time."""
        self.epoch_start_time = time.time()
    
    def on_epoch_end(self, trainer: "Trainer", state: TrainingState) -> None:
        """Display epoch summary."""
        elapsed = time.time() - self.epoch_start_time
        
        # Build progress message
        msg = f"Epoch {state.epoch:3d} | "
        msg += f"Train Loss: {state.train_loss:.4f} | "
        
        if state.val_loss is not None:
            msg += f"Val Loss: {state.val_loss:.4f} | "
        
        # Add metrics
        for metric in self.show_metrics:
            if metric in state.val_metrics:
                msg += f"Val {metric}: {state.val_metrics[metric]:.4f} | "
        
        msg += f"LR: {state.learning_rate:.2e} | "
        msg += f"Time: {elapsed:.1f}s"
        
        if state.is_best:
            msg += " ★"
        
        logger.info(msg)


class GradientClipCallback(Callback):
    """
    Gradient clipping callback with logging.
    """
    
    def __init__(
        self,
        max_norm: float = 1.0,
        norm_type: float = 2.0,
        log_grad_norm: bool = False,
    ):
        """
        Initialize gradient clipping.
        
        Args:
            max_norm: Maximum gradient norm.
            norm_type: Type of norm to use.
            log_grad_norm: Whether to log gradient norms.
        """
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.log_grad_norm = log_grad_norm
        self.grad_norms: list[float] = []
    
    def on_batch_end(
        self,
        trainer: "Trainer",
        batch: Any,
        batch_idx: int,
        loss: float,
    ) -> None:
        """Clip gradients after backward pass."""
        if self.log_grad_norm:
            # Calculate gradient norm before clipping
            total_norm = 0.0
            for p in trainer.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(self.norm_type)
                    total_norm += param_norm.item() ** self.norm_type
            total_norm = total_norm ** (1.0 / self.norm_type)
            self.grad_norms.append(total_norm)


class WarmupCallback(Callback):
    """
    Learning rate warmup callback.
    """
    
    def __init__(
        self,
        warmup_epochs: int = 5,
        initial_lr_factor: float = 0.1,
    ):
        """
        Initialize warmup.
        
        Args:
            warmup_epochs: Number of warmup epochs.
            initial_lr_factor: Initial LR as fraction of target LR.
        """
        self.warmup_epochs = warmup_epochs
        self.initial_lr_factor = initial_lr_factor
        self.base_lrs: list[float] = []
    
    def on_train_start(self, trainer: "Trainer") -> None:
        """Store base learning rates."""
        self.base_lrs = [g["lr"] for g in trainer.optimizer.param_groups]
    
    def on_epoch_start(self, trainer: "Trainer", state: TrainingState) -> None:
        """Apply warmup learning rate."""
        if state.epoch < self.warmup_epochs:
            factor = self.initial_lr_factor + (1 - self.initial_lr_factor) * (
                state.epoch / self.warmup_epochs
            )
            
            for i, g in enumerate(trainer.optimizer.param_groups):
                g["lr"] = self.base_lrs[i] * factor
