"""
Ray integration for distributed training.

Provides:
- Distributed data parallel training
- Hyperparameter tuning with Ray Tune
- Model serving with Ray Serve
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    import ray
    from ray import train
    from ray.train import Checkpoint, ScalingConfig
    from ray.train.torch import TorchTrainer
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

try:
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler
    RAY_TUNE_AVAILABLE = True
except ImportError:
    RAY_TUNE_AVAILABLE = False

logger = logging.getLogger(__name__)


def _check_ray():
    """Check if Ray is available."""
    if not RAY_AVAILABLE:
        raise ImportError("Ray not installed. Run: pip install 'ray[train]'")


class RayTrainer:
    """
    Distributed training with Ray Train.
    
    Features:
    - Data parallel training across GPUs/nodes
    - Automatic checkpointing
    - Fault tolerance
    - Integration with Ray Tune
    """
    
    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
        batch_size: int = 32,
        num_workers: int = 2,
        num_epochs: int = 10,
        learning_rate: float = 1e-3,
        use_gpu: bool = True,
    ):
        """
        Initialize Ray trainer.
        
        Args:
            model_factory: Function that creates model instance.
            train_dataset: Training dataset.
            val_dataset: Validation dataset.
            batch_size: Batch size per worker.
            num_workers: Number of distributed workers.
            num_epochs: Training epochs.
            learning_rate: Learning rate.
            use_gpu: Whether to use GPUs.
        """
        _check_ray()
        
        self.model_factory = model_factory
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.use_gpu = use_gpu
    
    def _train_func(self, config: dict) -> None:
        """Training function executed on each worker."""
        # Create model
        model = self.model_factory()
        model = train.torch.prepare_model(model)
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get("lr", self.learning_rate),
            weight_decay=config.get("weight_decay", 0.01),
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.get("batch_size", self.batch_size),
            shuffle=True,
            num_workers=4,
        )
        train_loader = train.torch.prepare_data_loader(train_loader)
        
        val_loader = None
        if self.val_dataset:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=config.get("batch_size", self.batch_size),
                shuffle=False,
                num_workers=4,
            )
            val_loader = train.torch.prepare_data_loader(val_loader)
        
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range(self.num_epochs):
            model.train()
            total_loss = 0.0
            num_batches = 0
            
            for batch in train_loader:
                inputs, targets = batch[0], batch[1]
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            train_loss = total_loss / num_batches
            
            # Validation
            val_loss = 0.0
            val_accuracy = 0.0
            
            if val_loader:
                model.eval()
                correct = 0
                total = 0
                val_total_loss = 0.0
                val_batches = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        inputs, targets = batch[0], batch[1]
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        
                        val_total_loss += loss.item()
                        val_batches += 1
                        
                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()
                
                val_loss = val_total_loss / val_batches
                val_accuracy = correct / total
            
            # Report metrics
            metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "epoch": epoch,
            }
            
            # Checkpoint
            checkpoint = Checkpoint.from_dict({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            })
            
            train.report(metrics, checkpoint=checkpoint)
    
    def train(self, config: dict | None = None) -> Any:
        """
        Run distributed training.
        
        Args:
            config: Training configuration.
        
        Returns:
            Ray Train result.
        """
        config = config or {}
        
        # Initialize Ray if not already
        if not ray.is_initialized():
            ray.init()
        
        # Configure scaling
        scaling_config = ScalingConfig(
            num_workers=self.num_workers,
            use_gpu=self.use_gpu,
        )
        
        # Create trainer
        trainer = TorchTrainer(
            train_loop_per_worker=lambda: self._train_func(config),
            scaling_config=scaling_config,
        )
        
        # Run training
        result = trainer.fit()
        
        return result


def distribute_training(
    model_factory: Callable[[], nn.Module],
    train_dataset: Dataset,
    val_dataset: Dataset | None = None,
    num_workers: int = 2,
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
) -> Any:
    """
    Convenience function for distributed training.
    
    Args:
        model_factory: Function that creates model.
        train_dataset: Training data.
        val_dataset: Validation data.
        num_workers: Number of workers.
        num_epochs: Number of epochs.
        batch_size: Batch size.
        learning_rate: Learning rate.
    
    Returns:
        Training result.
    """
    trainer = RayTrainer(
        model_factory=model_factory,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_workers=num_workers,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    
    return trainer.train()


class HyperparameterTuner:
    """
    Hyperparameter tuning with Ray Tune.
    """
    
    def __init__(
        self,
        train_func: Callable[[dict], None],
        param_space: dict,
        num_samples: int = 10,
        max_epochs: int = 10,
        metric: str = "val_accuracy",
        mode: str = "max",
    ):
        """
        Initialize tuner.
        
        Args:
            train_func: Training function accepting config dict.
            param_space: Hyperparameter search space.
            num_samples: Number of trials.
            max_epochs: Maximum epochs per trial.
            metric: Metric to optimize.
            mode: "min" or "max".
        """
        _check_ray()
        if not RAY_TUNE_AVAILABLE:
            raise ImportError("Ray Tune not available. Run: pip install 'ray[tune]'")
        
        self.train_func = train_func
        self.param_space = param_space
        self.num_samples = num_samples
        self.max_epochs = max_epochs
        self.metric = metric
        self.mode = mode
    
    def tune(self) -> Any:
        """
        Run hyperparameter tuning.
        
        Returns:
            Best configuration and results.
        """
        if not ray.is_initialized():
            ray.init()
        
        # ASHA scheduler for early stopping
        scheduler = ASHAScheduler(
            max_t=self.max_epochs,
            grace_period=1,
            reduction_factor=2,
        )
        
        # Reporter
        reporter = CLIReporter(
            metric_columns=[self.metric, "training_iteration"],
        )
        
        # Run tuning
        result = tune.run(
            self.train_func,
            config=self.param_space,
            num_samples=self.num_samples,
            scheduler=scheduler,
            progress_reporter=reporter,
            metric=self.metric,
            mode=self.mode,
        )
        
        best_config = result.get_best_config(metric=self.metric, mode=self.mode)
        best_result = result.get_best_result(metric=self.metric, mode=self.mode)
        
        return {
            "best_config": best_config,
            "best_result": best_result,
            "all_results": result,
        }


def get_tune_search_space() -> dict:
    """
    Get default hyperparameter search space.
    
    Returns:
        Search space dictionary.
    """
    if not RAY_TUNE_AVAILABLE:
        raise ImportError("Ray Tune not available")
    
    return {
        "lr": tune.loguniform(1e-5, 1e-2),
        "batch_size": tune.choice([16, 32, 64, 128]),
        "weight_decay": tune.loguniform(1e-6, 1e-2),
        "dropout": tune.uniform(0.1, 0.5),
        "hidden_dim": tune.choice([128, 256, 512]),
        "kernel_preset": tune.choice(["small", "medium", "large", "multi"]),
    }
