"""
Main trainer class for model training.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import TrainingConfig
from .callbacks import Callback, TrainingState
from .schedulers import SchedulerWrapper, get_scheduler

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    
    loss: float = 0.0
    accuracy: float = 0.0
    f1_macro: float = 0.0
    f1_weighted: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    
    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "f1_macro": self.f1_macro,
            "f1_weighted": self.f1_weighted,
            "precision": self.precision,
            "recall": self.recall,
        }


class Trainer:
    """
    Modern PyTorch trainer with support for:
    - Mixed precision training (AMP)
    - Gradient accumulation
    - Gradient clipping
    - Multiple callbacks
    - Learning rate scheduling
    - Metric tracking
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | SchedulerWrapper | None = None,
        criterion: nn.Module | None = None,
        config: TrainingConfig | None = None,
        callbacks: list[Callback] | None = None,
        device: torch.device | str | None = None,
        class_weights: torch.Tensor | None = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            optimizer: Optimizer (created automatically if None).
            scheduler: Learning rate scheduler.
            criterion: Loss function (CrossEntropyLoss if None).
            config: Training configuration.
            callbacks: List of callbacks.
            device: Device to train on.
        """
        # Set device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        
        # Model
        self.model = model.to(self.device)
        
        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Configuration
        self.config = config or TrainingConfig()
        
        # Optimizer
        if optimizer is None:
            optimizer = self._create_optimizer()
        self.optimizer = optimizer
        
        # Scheduler
        self.scheduler = scheduler
        if self.scheduler is not None and not isinstance(self.scheduler, SchedulerWrapper):
            self.scheduler = SchedulerWrapper(self.scheduler)
        
        # Loss function with class weights and label smoothing support
        if criterion is None:
            label_smoothing = getattr(self.config, 'label_smoothing', 0.0)
            if class_weights is not None:
                class_weights = class_weights.to(self.device)
                self.criterion = nn.CrossEntropyLoss(
                    weight=class_weights,
                    label_smoothing=label_smoothing,
                )
                logger.info(f"Using class weights: {class_weights.tolist()}")
            else:
                self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            if label_smoothing > 0:
                logger.info(f"Using label smoothing: {label_smoothing}")
        else:
            self.criterion = criterion
        
        # Mixed precision - use device type for new API
        self.use_amp = self.config.use_amp and self.device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None
        self.amp_device_type = self.device.type  # for autocast
        
        # Callbacks
        self.callbacks = callbacks or []
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.should_stop = False
        
        # Best metrics
        self.best_val_loss = float("inf")
        self.best_val_accuracy = 0.0
        
        logger.info(f"Trainer initialized on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"Mixed precision: {self.use_amp}")
    
    def _create_optimizer(self) -> Optimizer:
        """Create optimizer from config."""
        optimizer_name = self.config.optimizer.lower()
        lr = self.config.learning_rate
        weight_decay = self.config.weight_decay
        
        if optimizer_name == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        elif optimizer_name == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        elif optimizer_name == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _call_callbacks(self, method: str, *args: Any, **kwargs: Any) -> None:
        """Call a method on all callbacks."""
        for callback in self.callbacks:
            getattr(callback, method)(*args, **kwargs)
    
    def _compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, float]:
        """
        Compute classification metrics.
        
        Args:
            predictions: Model predictions (logits or probabilities).
            targets: Ground truth labels.
        
        Returns:
            Dictionary of metrics.
        """
        with torch.no_grad():
            # Get predicted classes
            if predictions.dim() > 1:
                pred_classes = predictions.argmax(dim=1)
            else:
                pred_classes = predictions
            
            # Accuracy
            correct = (pred_classes == targets).sum().item()
            total = targets.size(0)
            accuracy = correct / total if total > 0 else 0.0
            
            # Per-class metrics for F1
            num_classes = predictions.size(1) if predictions.dim() > 1 else 3
            
            precisions = []
            recalls = []
            f1s = []
            
            for c in range(num_classes):
                pred_c = (pred_classes == c)
                true_c = (targets == c)
                
                tp = (pred_c & true_c).sum().item()
                fp = (pred_c & ~true_c).sum().item()
                fn = (~pred_c & true_c).sum().item()
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)
            
            return {
                "accuracy": accuracy,
                "precision": sum(precisions) / len(precisions),
                "recall": sum(recalls) / len(recalls),
                "f1_macro": sum(f1s) / len(f1s),
            }
    
    def train_epoch(self) -> dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        num_batches = 0
        
        accumulation_steps = self.config.gradient_accumulation_steps
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch}",
            leave=False,
        )
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress_bar):
            self._call_callbacks("on_batch_start", self, batch, batch_idx)
            
            # Get data
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0], batch[1]
            else:
                inputs, targets = batch["input"], batch["target"]
            
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass with AMP (using new API)
            with autocast(device_type=self.amp_device_type, enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss = loss / accumulation_steps
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                if self.config.gradient_clip_val and self.config.gradient_clip_val > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_val,
                    )
                
                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Scheduler step (if per-batch)
                if self.scheduler is not None:
                    self.scheduler.step_batch()
            
            # Track metrics
            batch_loss = loss.item() * accumulation_steps
            total_loss += batch_loss
            num_batches += 1
            
            with torch.no_grad():
                all_predictions.append(outputs.detach().cpu())
                all_targets.append(targets.detach().cpu())
            
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({"loss": batch_loss})
            
            self._call_callbacks("on_batch_end", self, batch, batch_idx, batch_loss)
        
        # Compute epoch metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = self._compute_metrics(all_predictions, all_targets)
        metrics["loss"] = avg_loss
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """
        Run validation.
        
        Returns:
            Dictionary of validation metrics.
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        self._call_callbacks("on_validation_start", self)
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            # Get data
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0], batch[1]
            else:
                inputs, targets = batch["input"], batch["target"]
            
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass with AMP (using new API)
            with autocast(device_type=self.amp_device_type, enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            
            total_loss += loss.item()
            num_batches += 1
            
            all_predictions.append(outputs.cpu())
            all_targets.append(targets.cpu())
        
        # Compute metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = self._compute_metrics(all_predictions, all_targets)
        metrics["loss"] = avg_loss
        
        return metrics
    
    def fit(
        self,
        num_epochs: int | None = None,
        resume_from: str | Path | None = None,
    ) -> dict[str, list[float]]:
        """
        Full training loop.
        
        Args:
            num_epochs: Number of epochs (uses config if None).
            resume_from: Path to checkpoint to resume from.
        
        Returns:
            Dictionary of training history.
        """
        num_epochs = num_epochs or self.config.epochs
        
        # Resume if checkpoint provided
        if resume_from is not None:
            self.load_checkpoint(resume_from)
        
        # Training history
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "learning_rate": [],
        }
        
        self._call_callbacks("on_train_start", self)
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            # Create training state
            state = TrainingState(
                epoch=epoch,
                global_step=self.global_step,
                learning_rate=current_lr,
            )
            
            self._call_callbacks("on_epoch_start", self, state)
            
            # Train epoch
            train_metrics = self.train_epoch()
            state.train_loss = train_metrics["loss"]
            state.train_metrics = train_metrics
            
            # Validate
            if self.val_loader is not None:
                val_metrics = self.validate()
                state.val_loss = val_metrics.get("loss")
                state.val_metrics = val_metrics
                
                self._call_callbacks("on_validation_end", self, state)
            
            # Scheduler step (if per-epoch)
            if self.scheduler is not None:
                metrics_dict = {"val_loss": state.val_loss} if state.val_loss else {}
                self.scheduler.step_epoch(metrics_dict)
            
            # Update history
            history["train_loss"].append(train_metrics["loss"])
            history["train_accuracy"].append(train_metrics.get("accuracy", 0))
            history["learning_rate"].append(current_lr)
            
            if self.val_loader is not None:
                history["val_loss"].append(val_metrics["loss"])
                history["val_accuracy"].append(val_metrics.get("accuracy", 0))
            
            # Logging
            log_msg = (
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics.get('accuracy', 0):.4f}"
            )
            if self.val_loader is not None:
                log_msg += (
                    f" | Val Loss: {val_metrics['loss']:.4f} | "
                    f"Val Acc: {val_metrics.get('accuracy', 0):.4f}"
                )
            logger.info(log_msg)
            
            self._call_callbacks("on_epoch_end", self, state)
            
            # Early stopping check
            if self.should_stop:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        self._call_callbacks("on_train_end", self)
        
        return history
    
    def save_checkpoint(self, path: str | Path) -> None:
        """
        Save training checkpoint.
        
        Args:
            path: Path to save checkpoint.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_val_accuracy": self.best_val_accuracy,
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str | Path) -> None:
        """
        Load training checkpoint.
        
        Args:
            path: Path to checkpoint.
        """
        path = Path(path)
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.best_val_accuracy = checkpoint.get("best_val_accuracy", 0.0)
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        logger.info(f"Checkpoint loaded: {path} (epoch {self.current_epoch})")


def create_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    config: TrainingConfig | None = None,
    callbacks: list[Callback] | None = None,
) -> Trainer:
    """
    Factory function to create a trainer with common configurations.
    
    Args:
        model: Model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        config: Training configuration.
        callbacks: List of callbacks.
    
    Returns:
        Configured Trainer instance.
    """
    config = config or TrainingConfig()
    
    # Create scheduler
    scheduler = None
    if config.scheduler:
        # Create dummy optimizer first to pass to scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        scheduler = get_scheduler(
            name=config.scheduler,
            optimizer=optimizer,
            total_epochs=config.epochs,
            warmup_epochs=config.warmup_epochs,
            min_lr=config.min_lr,
            steps_per_epoch=len(train_loader),
        )
        
        return Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            callbacks=callbacks,
        )
    
    return Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        callbacks=callbacks,
    )
