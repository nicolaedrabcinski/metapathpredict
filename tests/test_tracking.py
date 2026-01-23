"""
Tests for experiment tracking module.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from metapathpredict.training.tracking import (
    ExperimentConfig,
    ExperimentTracker,
    MLflowTracker,
    WandbTracker,
    DuckDBTracker,
    NoOpTracker,
    create_tracker,
    TrackedTrainer,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)


class TestExperimentConfig:
    """Tests for ExperimentConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = ExperimentConfig()
        
        assert config.tracker == "none"
        assert config.experiment_name == "metapathpredict"
        assert config.run_name is None
        assert config.tags == {}
    
    def test_mlflow_config(self):
        """Test MLflow configuration."""
        config = ExperimentConfig(
            tracker="mlflow",
            experiment_name="test_experiment",
            mlflow_tracking_uri="http://localhost:5000",
        )
        
        assert config.tracker == "mlflow"
        assert config.mlflow_tracking_uri == "http://localhost:5000"
    
    def test_wandb_config(self):
        """Test W&B configuration."""
        config = ExperimentConfig(
            tracker="wandb",
            wandb_project="my_project",
            wandb_entity="my_team",
            wandb_mode="offline",
        )
        
        assert config.tracker == "wandb"
        assert config.wandb_project == "my_project"
        assert config.wandb_entity == "my_team"
        assert config.wandb_mode == "offline"
    
    def test_duckdb_config(self):
        """Test DuckDB configuration."""
        config = ExperimentConfig(
            tracker="duckdb",
            duckdb_path="/tmp/experiments.duckdb",
        )
        
        assert config.tracker == "duckdb"
        assert config.duckdb_path == "/tmp/experiments.duckdb"


class TestNoOpTracker:
    """Tests for NoOpTracker."""
    
    def test_all_methods_are_noop(self):
        """Test that all methods do nothing."""
        tracker = NoOpTracker()
        
        # None of these should raise
        tracker.start_run("test")
        tracker.log_params({"lr": 0.001})
        tracker.log_metrics({"loss": 0.5}, step=1)
        tracker.log_artifact("/path/to/file")
        tracker.log_model(SimpleModel(), "model")
        tracker.set_tag("key", "value")
        tracker.end_run()


class TestDuckDBTracker:
    """Tests for DuckDBTracker."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.duckdb")
            yield db_path
    
    @pytest.fixture
    def tracker(self, temp_db):
        """Create DuckDB tracker."""
        config = ExperimentConfig(
            tracker="duckdb",
            experiment_name="test_experiment",
            duckdb_path=temp_db,
        )
        return DuckDBTracker(config)
    
    def test_start_and_end_run(self, tracker):
        """Test starting and ending a run."""
        run_id = tracker.start_run("test_run")
        
        assert run_id is not None
        assert tracker._run_id == run_id
        
        tracker.end_run()
        assert tracker._run_id is None
    
    def test_log_params(self, tracker):
        """Test logging parameters."""
        tracker.start_run("test_run")
        
        params = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "model": {
                "hidden_dim": 256,
                "num_layers": 3,
            }
        }
        tracker.log_params(params)
        
        tracker.end_run()
    
    def test_log_metrics(self, tracker):
        """Test logging metrics."""
        tracker.start_run("test_run")
        run_id = tracker._run_id
        
        tracker.log_metrics({"loss": 0.5, "accuracy": 0.8}, step=1)
        tracker.log_metrics({"loss": 0.3, "accuracy": 0.9}, step=2)
        
        metrics = tracker.get_run_metrics(run_id)
        assert len(metrics) == 4  # 2 metrics * 2 steps
        
        tracker.end_run()
    
    def test_log_artifact(self, tracker, temp_db):
        """Test logging artifacts."""
        tracker.start_run("test_run")
        
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"test content")
            temp_file = f.name
        
        try:
            tracker.log_artifact(temp_file, "test_artifact")
        finally:
            os.unlink(temp_file)
        
        tracker.end_run()
    
    def test_log_model(self, tracker):
        """Test logging model."""
        tracker.start_run("test_run")
        
        model = SimpleModel()
        tracker.log_model(model, "test_model")
        
        tracker.end_run()
    
    def test_set_tag(self, tracker):
        """Test setting tags."""
        tracker.start_run("test_run")
        
        tracker.set_tag("framework", "pytorch")
        tracker.set_tag("version", "2.0")
        
        tracker.end_run()
    
    def test_get_all_runs(self, tracker):
        """Test getting all runs."""
        tracker.start_run("run_1")
        tracker.end_run()
        
        tracker.start_run("run_2")
        tracker.end_run()
        
        runs = tracker.get_all_runs()
        assert len(runs) == 2
    
    def test_flatten_dict(self):
        """Test dictionary flattening."""
        nested = {
            "a": 1,
            "b": {
                "c": 2,
                "d": {
                    "e": 3
                }
            }
        }
        
        flat = DuckDBTracker._flatten_dict(nested)
        
        assert flat == {
            "a": "1",
            "b.c": "2",
            "b.d.e": "3"
        }


class TestCreateTracker:
    """Tests for tracker factory."""
    
    def test_create_noop_tracker(self):
        """Test creating no-op tracker."""
        config = ExperimentConfig(tracker="none")
        tracker = create_tracker(config)
        
        assert isinstance(tracker, NoOpTracker)
    
    def test_create_duckdb_tracker(self):
        """Test creating DuckDB tracker."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(
                tracker="duckdb",
                duckdb_path=os.path.join(tmpdir, "test.duckdb"),
            )
            tracker = create_tracker(config)
            
            assert isinstance(tracker, DuckDBTracker)
    
    def test_create_unknown_tracker(self):
        """Test creating unknown tracker raises error."""
        config = ExperimentConfig(tracker="unknown")
        
        with pytest.raises(ValueError, match="Unknown tracker type"):
            create_tracker(config)
    
    @patch.dict(os.environ, {"MLFLOW_TRACKING_URI": ""})
    def test_create_mlflow_tracker_import_error(self):
        """Test MLflow tracker with import error."""
        config = ExperimentConfig(tracker="mlflow")
        
        # This will either work (if mlflow installed) or raise ImportError
        try:
            tracker = create_tracker(config)
            assert isinstance(tracker, MLflowTracker)
        except ImportError:
            pass  # Expected if mlflow not installed
    
    def test_create_wandb_tracker_import_error(self):
        """Test W&B tracker with import error."""
        config = ExperimentConfig(tracker="wandb")
        
        # This will either work (if wandb installed) or raise ImportError
        try:
            tracker = create_tracker(config)
            assert isinstance(tracker, WandbTracker)
        except ImportError:
            pass  # Expected if wandb not installed


class MockTrainer:
    """Mock trainer for testing."""
    
    def __init__(self):
        self.model = SimpleModel()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.scheduler = None
        self.epoch = 0
    
    def train_epoch(self, train_loader):
        """Mock train epoch."""
        self.epoch += 1
        return {
            "loss": 1.0 / self.epoch,
            "accuracy": 1.0 - 1.0 / (self.epoch + 1),
        }
    
    def validate(self, val_loader):
        """Mock validation."""
        return {
            "loss": 1.0 / (self.epoch + 0.5),
            "accuracy": 1.0 - 1.0 / (self.epoch + 1.5),
        }


class TestTrackedTrainer:
    """Tests for TrackedTrainer."""
    
    @pytest.fixture
    def mock_tracker(self):
        """Create mock tracker."""
        tracker = MagicMock(spec=ExperimentTracker)
        return tracker
    
    @pytest.fixture
    def mock_trainer(self):
        """Create mock trainer."""
        return MockTrainer()
    
    def test_tracked_training(self, mock_tracker, mock_trainer):
        """Test training with tracking."""
        tracked = TrackedTrainer(mock_trainer, mock_tracker)
        
        # Mock data loaders
        train_loader = MagicMock()
        val_loader = MagicMock()
        
        config = {"learning_rate": 0.001}
        
        tracked.train(
            train_loader,
            val_loader,
            epochs=3,
            run_name="test_run",
            config=config,
        )
        
        # Verify tracking calls
        mock_tracker.start_run.assert_called_once_with(run_name="test_run")
        mock_tracker.log_params.assert_called_once_with(config)
        
        # 3 epochs of metrics
        assert mock_tracker.log_metrics.call_count == 3
        
        # Model and status logged
        mock_tracker.log_model.assert_called_once()
        mock_tracker.set_tag.assert_called_with("status", "completed")
        mock_tracker.end_run.assert_called_once()
    
    def test_tracked_training_with_error(self, mock_tracker):
        """Test training handles errors."""
        trainer = MagicMock()
        trainer.train_epoch.side_effect = RuntimeError("Training failed")
        
        tracked = TrackedTrainer(trainer, mock_tracker)
        
        with pytest.raises(RuntimeError, match="Training failed"):
            tracked.train(MagicMock(), MagicMock(), epochs=1)
        
        # Verify error handling
        mock_tracker.set_tag.assert_any_call("status", "failed")
        mock_tracker.end_run.assert_called_once()


class TestMLflowTrackerMocked:
    """Tests for MLflow tracker with mocking."""
    
    @pytest.fixture
    def mock_mlflow(self):
        """Mock mlflow module."""
        with patch.dict('sys.modules', {'mlflow': MagicMock()}):
            import sys
            mock = sys.modules['mlflow']
            mock.get_experiment_by_name.return_value = None
            yield mock
    
    def test_flatten_dict(self):
        """Test dictionary flattening."""
        nested = {
            "model": {
                "hidden_dim": 256,
                "layers": 3
            },
            "lr": 0.001
        }
        
        flat = MLflowTracker._flatten_dict(nested)
        
        assert flat["model.hidden_dim"] == "256"
        assert flat["model.layers"] == "3"
        assert flat["lr"] == "0.001"


class TestIntegrationDuckDB:
    """Integration tests for DuckDB tracker."""
    
    @pytest.fixture
    def tracker(self):
        """Create DuckDB tracker with temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(
                tracker="duckdb",
                experiment_name="integration_test",
                duckdb_path=os.path.join(tmpdir, "test.duckdb"),
            )
            tracker = DuckDBTracker(config)
            yield tracker
    
    def test_full_experiment_workflow(self, tracker):
        """Test complete experiment workflow."""
        # Start run
        run_id = tracker.start_run("experiment_1", tags={"version": "1.0"})
        
        # Log hyperparameters
        tracker.log_params({
            "learning_rate": 0.001,
            "batch_size": 32,
            "model_type": "cnn",
        })
        
        # Simulate training loop
        for epoch in range(5):
            loss = 1.0 / (epoch + 1)
            accuracy = 1.0 - loss / 2
            
            tracker.log_metrics({
                "train_loss": loss,
                "train_accuracy": accuracy,
            }, step=epoch)
        
        # Log final model
        model = SimpleModel()
        tracker.log_model(model, "best_model")
        
        # Add tags
        tracker.set_tag("best_accuracy", "0.9")
        tracker.set_tag("status", "completed")
        
        # End run
        tracker.end_run()
        
        # Verify
        runs = tracker.get_all_runs()
        assert len(runs) == 1
        assert runs[0]["run_name"] == "experiment_1"
        assert runs[0]["status"] == "completed"
        
        metrics = tracker.get_run_metrics(run_id)
        assert len(metrics) == 10  # 2 metrics * 5 epochs
