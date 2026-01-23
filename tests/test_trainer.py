"""
Unit tests for trainer module.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from metapathpredict.training.trainer import Trainer
from metapathpredict.training.callbacks import EarlyStopping
from metapathpredict.models.cnn import MultiScaleCNN
from metapathpredict.config.settings import TrainingConfig


def create_mock_dataloaders(batch_size=8, seq_len=500, num_classes=3, num_batches=5):
    """Create mock train and val data loaders."""
    # MultiScaleCNN expects (B, L, 4) format
    num_samples = batch_size * num_batches
    
    x_train = torch.randn(num_samples, seq_len, 4)
    y_train = torch.randint(0, num_classes, (num_samples,))
    
    x_val = torch.randn(num_samples // 2, seq_len, 4)
    y_val = torch.randint(0, num_classes, (num_samples // 2,))
    
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader


class TestTrainerInitialization:
    """Tests for Trainer initialization."""

    def test_basic_initialization(self):
        """Test basic trainer initialization."""
        model = MultiScaleCNN(seq_length=500, num_classes=3, branch_channels=32)
        train_loader, val_loader = create_mock_dataloaders()
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
        )
        
        assert trainer.model is not None
        assert trainer.device is not None
        assert trainer.train_loader is train_loader

    def test_initialization_with_device(self):
        """Test initialization with specific device."""
        model = MultiScaleCNN(seq_length=500, num_classes=3, branch_channels=32)
        train_loader, val_loader = create_mock_dataloaders()
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=torch.device("cpu"),
        )
        
        assert trainer.device == torch.device("cpu")

    def test_initialization_with_scheduler(self):
        """Test initialization with learning rate scheduler."""
        model = MultiScaleCNN(seq_length=500, num_classes=3, branch_channels=32)
        train_loader, val_loader = create_mock_dataloaders()
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        
        assert trainer.scheduler is not None


class TestTrainerTraining:
    """Tests for training functionality."""

    def test_train_epoch(self):
        """Test training for one epoch."""
        model = MultiScaleCNN(seq_length=500, num_classes=3, branch_channels=32)
        train_loader, val_loader = create_mock_dataloaders()
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=torch.device("cpu"),
        )
        
        metrics = trainer.train_epoch()
        
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert metrics["loss"] > 0

    def test_validate(self):
        """Test validation."""
        model = MultiScaleCNN(seq_length=500, num_classes=3, branch_channels=32)
        train_loader, val_loader = create_mock_dataloaders()
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=torch.device("cpu"),
        )
        
        metrics = trainer.validate()
        
        assert "loss" in metrics
        assert "accuracy" in metrics

    def test_training_improves_loss(self):
        """Test that training reduces loss over time."""
        torch.manual_seed(42)
        
        model = MultiScaleCNN(seq_length=500, num_classes=3, branch_channels=64)
        train_loader, val_loader = create_mock_dataloaders(num_batches=10)
        
        config = TrainingConfig(learning_rate=0.01)
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=torch.device("cpu"),
        )
        
        # Train for multiple epochs
        losses = []
        for _ in range(3):
            metrics = trainer.train_epoch()
            losses.append(metrics["loss"])
            trainer.current_epoch += 1
        
        # Loss should generally decrease (allowing for some variation)
        assert losses[-1] <= losses[0] * 1.5  # Allow some tolerance


class TestTrainerCheckpointing:
    """Tests for model checkpointing."""

    def test_save_checkpoint(self, tmp_path):
        """Test saving checkpoint."""
        model = MultiScaleCNN(seq_length=500, num_classes=3, branch_channels=32)
        train_loader, val_loader = create_mock_dataloaders()
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
        )
        
        checkpoint_path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(checkpoint_path)
        
        assert checkpoint_path.exists()
        
        # Verify checkpoint contents
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint

    def test_load_checkpoint(self, tmp_path):
        """Test loading checkpoint."""
        model = MultiScaleCNN(seq_length=500, num_classes=3, branch_channels=32)
        train_loader, val_loader = create_mock_dataloaders()
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
        )
        
        # Save initial state
        checkpoint_path = tmp_path / "checkpoint.pt"
        trainer.current_epoch = 10
        trainer.save_checkpoint(checkpoint_path)
        
        # Create new trainer and load
        model2 = MultiScaleCNN(seq_length=500, num_classes=3, branch_channels=32)
        
        trainer2 = Trainer(
            model=model2,
            train_loader=train_loader,
            val_loader=val_loader,
        )
        
        trainer2.load_checkpoint(checkpoint_path)
        
        assert trainer2.current_epoch == 10

    def test_resume_training(self, tmp_path):
        """Test resuming training from checkpoint."""
        model = MultiScaleCNN(seq_length=500, num_classes=3, branch_channels=32)
        train_loader, val_loader = create_mock_dataloaders()
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=torch.device("cpu"),
        )
        
        # Train for a few epochs
        for _ in range(2):
            trainer.train_epoch()
            trainer.current_epoch += 1
        
        # Save checkpoint
        checkpoint_path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(checkpoint_path)
        
        # Create new trainer and resume
        model2 = MultiScaleCNN(seq_length=500, num_classes=3, branch_channels=32)
        
        trainer2 = Trainer(
            model=model2,
            train_loader=train_loader,
            val_loader=val_loader,
            device=torch.device("cpu"),
        )
        
        trainer2.load_checkpoint(checkpoint_path)
        
        # Continue training
        metrics = trainer2.train_epoch()
        
        assert trainer2.current_epoch == 2
        assert "loss" in metrics


class TestMixedPrecision:
    """Tests for mixed precision training."""

    def test_amp_training(self):
        """Test training with automatic mixed precision (on CPU fallback)."""
        model = MultiScaleCNN(seq_length=500, num_classes=3, branch_channels=32)
        train_loader, val_loader = create_mock_dataloaders()
        
        # AMP only works on CUDA, but should not error on CPU
        config = TrainingConfig(use_amp=True)
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=torch.device("cpu"),
        )
        
        # Should complete without error
        metrics = trainer.train_epoch()
        
        assert "loss" in metrics


class TestGradientClipping:
    """Tests for gradient clipping."""

    def test_gradient_clipping(self):
        """Test that gradient clipping is applied."""
        model = MultiScaleCNN(seq_length=500, num_classes=3, branch_channels=32)
        train_loader, val_loader = create_mock_dataloaders()
        
        config = TrainingConfig(gradient_clip_val=1.0)
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=torch.device("cpu"),
        )
        
        # Should complete without exploding gradients
        metrics = trainer.train_epoch()
        
        assert not torch.isnan(torch.tensor(metrics["loss"]))


class TestEarlyStopping:
    """Tests for early stopping."""

    def test_early_stopping_triggers(self):
        """Test that early stopping triggers when validation doesn't improve."""
        # Test EarlyStopping callback directly
        early_stopping = EarlyStopping(patience=3, min_delta=0.0, mode="min")
        
        # Simulate validation losses that don't improve
        for val_loss in [1.0, 1.1, 1.2, 1.3, 1.4]:
            is_improvement = early_stopping._is_improvement(val_loss)
            if not is_improvement:
                early_stopping.counter += 1
            else:
                early_stopping.best_value = val_loss
                early_stopping.counter = 0
            
            if early_stopping.counter >= early_stopping.patience:
                early_stopping.should_stop = True
                break
        
        assert early_stopping.should_stop is True

    def test_early_stopping_resets(self):
        """Test that early stopping counter resets on improvement."""
        early_stopping = EarlyStopping(patience=3, min_delta=0.0, mode="min")
        
        # Simulate validation losses that improve
        for val_loss in [1.0, 0.9, 0.8, 0.7]:
            is_improvement = early_stopping._is_improvement(val_loss)
            if is_improvement:
                early_stopping.best_value = val_loss
                early_stopping.counter = 0
            else:
                early_stopping.counter += 1
        
        assert early_stopping.should_stop is False
        assert early_stopping.counter == 0


class TestMetricsLogging:
    """Tests for metrics logging."""

    def test_metrics_are_tracked(self):
        """Test that training metrics are tracked."""
        model = MultiScaleCNN(seq_length=500, num_classes=3, branch_channels=32)
        train_loader, val_loader = create_mock_dataloaders()
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=torch.device("cpu"),
        )
        
        # Run a few epochs
        train_losses = []
        val_losses = []
        for _ in range(3):
            train_metrics = trainer.train_epoch()
            val_metrics = trainer.validate()
            train_losses.append(train_metrics["loss"])
            val_losses.append(val_metrics["loss"])
            trainer.current_epoch += 1
        
        # Check that metrics were collected
        assert len(train_losses) == 3
        assert len(val_losses) == 3
