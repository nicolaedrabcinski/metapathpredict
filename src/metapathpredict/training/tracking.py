"""
Experiment tracking integration for MetaPathPredict.

Supports MLflow and Weights & Biases (wandb) for experiment tracking.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking."""
    
    # Tracker type: "mlflow", "wandb", "duckdb", or "none"
    tracker: str = "none"
    
    # Common settings
    experiment_name: str = "metapathpredict"
    run_name: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    
    # MLflow settings
    mlflow_tracking_uri: str = "mlruns"
    mlflow_artifact_location: Optional[str] = None
    
    # Weights & Biases settings
    wandb_project: str = "metapathpredict"
    wandb_entity: Optional[str] = None
    wandb_mode: str = "online"  # "online", "offline", "disabled"
    
    # DuckDB settings (local tracking)
    duckdb_path: str = "experiments.duckdb"


class ExperimentTracker(ABC):
    """Abstract base class for experiment trackers."""
    
    @abstractmethod
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """Start a new experiment run."""
        pass
    
    @abstractmethod
    def end_run(self):
        """End the current run."""
        pass
    
    @abstractmethod
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        pass
    
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        pass
    
    @abstractmethod
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log an artifact (file)."""
        pass
    
    @abstractmethod
    def log_model(self, model: nn.Module, artifact_path: str):
        """Log a PyTorch model."""
        pass
    
    @abstractmethod
    def set_tag(self, key: str, value: str):
        """Set a tag."""
        pass


class MLflowTracker(ExperimentTracker):
    """MLflow experiment tracker."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._run = None
        
        try:
            import mlflow
            self.mlflow = mlflow
            
            # Set tracking URI
            mlflow.set_tracking_uri(config.mlflow_tracking_uri)
            
            # Set or create experiment
            experiment = mlflow.get_experiment_by_name(config.experiment_name)
            if experiment is None:
                mlflow.create_experiment(
                    config.experiment_name,
                    artifact_location=config.mlflow_artifact_location,
                )
            mlflow.set_experiment(config.experiment_name)
            
        except ImportError:
            raise ImportError(
                "MLflow not installed. Install with: pip install mlflow"
            )
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """Start a new MLflow run."""
        run_name = run_name or self.config.run_name
        all_tags = {**self.config.tags, **(tags or {})}
        
        self._run = self.mlflow.start_run(run_name=run_name, tags=all_tags)
        return self._run
    
    def end_run(self):
        """End the current run."""
        if self._run:
            self.mlflow.end_run()
            self._run = None
    
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        # MLflow requires string values for nested dicts
        flat_params = self._flatten_dict(params)
        self.mlflow.log_params(flat_params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        self.mlflow.log_metrics(metrics, step=step)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log an artifact."""
        self.mlflow.log_artifact(local_path, artifact_path)
    
    def log_model(self, model: nn.Module, artifact_path: str):
        """Log a PyTorch model."""
        self.mlflow.pytorch.log_model(model, artifact_path)
    
    def set_tag(self, key: str, value: str):
        """Set a tag."""
        self.mlflow.set_tag(key, value)
    
    @staticmethod
    def _flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, str]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(MLflowTracker._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, str(v)))
        return dict(items)


class WandbTracker(ExperimentTracker):
    """Weights & Biases experiment tracker."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._run = None
        
        try:
            import wandb
            self.wandb = wandb
        except ImportError:
            raise ImportError(
                "wandb not installed. Install with: pip install wandb"
            )
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """Start a new W&B run."""
        run_name = run_name or self.config.run_name
        all_tags = list((tags or {}).keys()) + list(self.config.tags.keys())
        
        self._run = self.wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            name=run_name,
            tags=all_tags,
            mode=self.config.wandb_mode,
        )
        return self._run
    
    def end_run(self):
        """End the current run."""
        if self._run:
            self.wandb.finish()
            self._run = None
    
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        self.wandb.config.update(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        if step is not None:
            metrics["step"] = step
        self.wandb.log(metrics)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log an artifact."""
        artifact_name = artifact_path or Path(local_path).name
        artifact = self.wandb.Artifact(artifact_name, type="file")
        artifact.add_file(local_path)
        self.wandb.log_artifact(artifact)
    
    def log_model(self, model: nn.Module, artifact_path: str):
        """Log a PyTorch model."""
        # Save model temporarily
        temp_path = f"/tmp/{artifact_path}.pt"
        torch.save(model.state_dict(), temp_path)
        
        artifact = self.wandb.Artifact(artifact_path, type="model")
        artifact.add_file(temp_path)
        self.wandb.log_artifact(artifact)
        
        # Cleanup
        os.remove(temp_path)
    
    def set_tag(self, key: str, value: str):
        """Set a tag."""
        if self._run:
            self._run.tags.append(f"{key}:{value}")
    
    def watch_model(self, model: nn.Module, log: str = "gradients", log_freq: int = 100):
        """Watch model for gradient logging."""
        self.wandb.watch(model, log=log, log_freq=log_freq)


class DuckDBTracker(ExperimentTracker):
    """Local DuckDB experiment tracker."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._run_id: Optional[str] = None
        self._run_name: Optional[str] = None
        
        try:
            import duckdb
            self.duckdb = duckdb
            self.conn = duckdb.connect(config.duckdb_path)
            self._init_tables()
        except ImportError:
            raise ImportError(
                "DuckDB not installed. Install with: pip install duckdb"
            )
    
    def _init_tables(self):
        """Initialize database tables."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id VARCHAR PRIMARY KEY,
                experiment_name VARCHAR,
                run_name VARCHAR,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                status VARCHAR
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS params (
                run_id VARCHAR,
                key VARCHAR,
                value VARCHAR,
                PRIMARY KEY (run_id, key)
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                run_id VARCHAR,
                key VARCHAR,
                value DOUBLE,
                step INTEGER,
                timestamp TIMESTAMP
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tags (
                run_id VARCHAR,
                key VARCHAR,
                value VARCHAR,
                PRIMARY KEY (run_id, key)
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS artifacts (
                run_id VARCHAR,
                artifact_path VARCHAR,
                local_path VARCHAR,
                timestamp TIMESTAMP
            )
        """)
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """Start a new run."""
        import uuid
        
        self._run_id = str(uuid.uuid4())[:8]
        self._run_name = run_name or f"run_{self._run_id}"
        
        self.conn.execute("""
            INSERT INTO runs (run_id, experiment_name, run_name, start_time, status)
            VALUES (?, ?, ?, ?, 'running')
        """, [self._run_id, self.config.experiment_name, self._run_name, datetime.now()])
        
        # Log initial tags
        all_tags = {**self.config.tags, **(tags or {})}
        for key, value in all_tags.items():
            self.set_tag(key, value)
        
        return self._run_id
    
    def end_run(self):
        """End the current run."""
        if self._run_id:
            self.conn.execute("""
                UPDATE runs SET end_time = ?, status = 'completed'
                WHERE run_id = ?
            """, [datetime.now(), self._run_id])
            self._run_id = None
    
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        flat_params = self._flatten_dict(params)
        for key, value in flat_params.items():
            self.conn.execute("""
                INSERT OR REPLACE INTO params (run_id, key, value)
                VALUES (?, ?, ?)
            """, [self._run_id, key, str(value)])
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        for key, value in metrics.items():
            self.conn.execute("""
                INSERT INTO metrics (run_id, key, value, step, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, [self._run_id, key, value, step, datetime.now()])
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log an artifact."""
        artifact_path = artifact_path or Path(local_path).name
        self.conn.execute("""
            INSERT INTO artifacts (run_id, artifact_path, local_path, timestamp)
            VALUES (?, ?, ?, ?)
        """, [self._run_id, artifact_path, local_path, datetime.now()])
    
    def log_model(self, model: nn.Module, artifact_path: str):
        """Log a PyTorch model."""
        model_dir = Path(self.config.duckdb_path).parent / "models" / self._run_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / f"{artifact_path}.pt"
        torch.save(model.state_dict(), model_path)
        
        self.log_artifact(str(model_path), artifact_path)
    
    def set_tag(self, key: str, value: str):
        """Set a tag."""
        self.conn.execute("""
            INSERT OR REPLACE INTO tags (run_id, key, value)
            VALUES (?, ?, ?)
        """, [self._run_id, key, value])
    
    def get_run_metrics(self, run_id: str) -> List[Dict]:
        """Get metrics for a run."""
        result = self.conn.execute("""
            SELECT key, value, step, timestamp
            FROM metrics WHERE run_id = ?
            ORDER BY timestamp
        """, [run_id]).fetchall()
        
        return [
            {"key": r[0], "value": r[1], "step": r[2], "timestamp": r[3]}
            for r in result
        ]
    
    def get_all_runs(self) -> List[Dict]:
        """Get all runs."""
        result = self.conn.execute("""
            SELECT run_id, experiment_name, run_name, start_time, end_time, status
            FROM runs ORDER BY start_time DESC
        """).fetchall()
        
        return [
            {
                "run_id": r[0],
                "experiment_name": r[1],
                "run_name": r[2],
                "start_time": r[3],
                "end_time": r[4],
                "status": r[5],
            }
            for r in result
        ]
    
    @staticmethod
    def _flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, str]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(DuckDBTracker._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, str(v)))
        return dict(items)


class NoOpTracker(ExperimentTracker):
    """No-op tracker for when tracking is disabled."""
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        pass
    
    def end_run(self):
        pass
    
    def log_params(self, params: Dict[str, Any]):
        pass
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        pass
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        pass
    
    def log_model(self, model: nn.Module, artifact_path: str):
        pass
    
    def set_tag(self, key: str, value: str):
        pass


def create_tracker(config: ExperimentConfig) -> ExperimentTracker:
    """Factory function to create experiment tracker."""
    tracker_type = config.tracker.lower()
    
    if tracker_type == "mlflow":
        return MLflowTracker(config)
    elif tracker_type == "wandb":
        return WandbTracker(config)
    elif tracker_type == "duckdb":
        return DuckDBTracker(config)
    elif tracker_type == "none":
        return NoOpTracker()
    else:
        raise ValueError(f"Unknown tracker type: {tracker_type}")


class TrackedTrainer:
    """
    Trainer wrapper that adds experiment tracking.
    
    Usage:
        config = ExperimentConfig(tracker="mlflow")
        tracker = create_tracker(config)
        
        tracked_trainer = TrackedTrainer(trainer, tracker)
        tracked_trainer.train(train_loader, val_loader, epochs=100)
    """
    
    def __init__(self, trainer, tracker: ExperimentTracker):
        self.trainer = trainer
        self.tracker = tracker
    
    def train(
        self,
        train_loader,
        val_loader,
        epochs: int,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Train with experiment tracking."""
        # Start run
        self.tracker.start_run(run_name=run_name)
        
        # Log config/params
        if config:
            self.tracker.log_params(config)
        
        try:
            for epoch in range(epochs):
                # Train epoch
                train_metrics = self.trainer.train_epoch(train_loader)
                
                # Validate
                val_metrics = self.trainer.validate(val_loader)
                
                # Log metrics
                metrics = {
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics.get("accuracy", 0),
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics.get("accuracy", 0),
                }
                self.tracker.log_metrics(metrics, step=epoch)
                
                # Log learning rate if available
                if hasattr(self.trainer, "scheduler") and self.trainer.scheduler:
                    lr = self.trainer.optimizer.param_groups[0]["lr"]
                    self.tracker.log_metrics({"learning_rate": lr}, step=epoch)
            
            # Log final model
            self.tracker.log_model(self.trainer.model, "final_model")
            self.tracker.set_tag("status", "completed")
            
        except Exception as e:
            self.tracker.set_tag("status", "failed")
            self.tracker.set_tag("error", str(e))
            raise
        
        finally:
            self.tracker.end_run()
