# API Reference: Training

Training utilities and callbacks.

## Trainer

```python
class Trainer:
    """Main trainer class for model training.
    
    Args:
        model: PyTorch model to train.
        config: Training configuration.
        device: Device to train on ("cuda" or "cpu").
        callbacks: List of callback objects.
    
    Example:
        >>> trainer = Trainer(model, config, device="cuda")
        >>> history = trainer.fit(train_loader, val_loader)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        device: str = "cuda",
        callbacks: Optional[List[Callback]] = None,
    ):
        pass
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            epochs: Number of epochs (overrides config).
        
        Returns:
            Training history dictionary.
        """
        pass
    
    def train_epoch(
        self,
        train_loader: DataLoader,
    ) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary with epoch metrics.
        """
        pass
    
    def validate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """Validate the model.
        
        Returns:
            Dictionary with validation metrics.
        """
        pass
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        pass
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint and resume."""
        pass
```

## ContrastiveTrainer

```python
class ContrastiveTrainer:
    """Trainer for contrastive learning pretraining.
    
    Args:
        model: ContrastiveModel to train.
        temperature: NT-Xent loss temperature.
        device: Training device.
    
    Example:
        >>> trainer = ContrastiveTrainer(model, temperature=0.5)
        >>> trainer.pretrain(train_loader, epochs=100)
    """
    
    def __init__(
        self,
        model: ContrastiveModel,
        temperature: float = 0.5,
        device: str = "cuda",
    ):
        pass
    
    def pretrain(
        self,
        train_loader: DataLoader,
        epochs: int = 100,
        lr: float = 3e-4,
    ) -> Dict[str, List[float]]:
        """Pretrain with contrastive learning.
        
        Returns:
            Pretraining history.
        """
        pass
```

## Callbacks

### Callback (Base)

```python
class Callback:
    """Base callback class.
    
    Override methods to add custom behavior at training hooks.
    """
    
    def on_train_begin(self, trainer: Trainer):
        """Called at start of training."""
        pass
    
    def on_train_end(self, trainer: Trainer):
        """Called at end of training."""
        pass
    
    def on_epoch_begin(self, trainer: Trainer, epoch: int):
        """Called at start of each epoch."""
        pass
    
    def on_epoch_end(
        self,
        trainer: Trainer,
        epoch: int,
        logs: Dict[str, float],
    ):
        """Called at end of each epoch."""
        pass
    
    def on_batch_begin(self, trainer: Trainer, batch: int):
        """Called at start of each batch."""
        pass
    
    def on_batch_end(
        self,
        trainer: Trainer,
        batch: int,
        logs: Dict[str, float],
    ):
        """Called at end of each batch."""
        pass
```

### EarlyStopping

```python
class EarlyStopping(Callback):
    """Stop training when metric stops improving.
    
    Args:
        monitor: Metric to monitor ("val_loss" or "val_accuracy").
        patience: Epochs to wait before stopping.
        min_delta: Minimum change to qualify as improvement.
        mode: "min" for loss, "max" for accuracy.
    
    Example:
        >>> callback = EarlyStopping(patience=10, monitor="val_loss")
    """
    
    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        pass
```

### ModelCheckpoint

```python
class ModelCheckpoint(Callback):
    """Save model checkpoints during training.
    
    Args:
        dirpath: Directory to save checkpoints.
        filename: Checkpoint filename template.
        monitor: Metric to monitor for best model.
        mode: "min" or "max".
        save_top_k: Number of best models to keep.
        save_last: Whether to save last model.
    
    Example:
        >>> callback = ModelCheckpoint(
        ...     dirpath="checkpoints",
        ...     monitor="val_accuracy",
        ...     mode="max",
        ... )
    """
    
    def __init__(
        self,
        dirpath: str = "checkpoints",
        filename: str = "model_{epoch}_{val_loss:.4f}",
        monitor: str = "val_loss",
        mode: str = "min",
        save_top_k: int = 1,
        save_last: bool = True,
    ):
        pass
```

### LearningRateMonitor

```python
class LearningRateMonitor(Callback):
    """Log learning rate at each epoch.
    
    Example:
        >>> callback = LearningRateMonitor()
    """
    
    def on_epoch_end(self, trainer, epoch, logs):
        lr = trainer.optimizer.param_groups[0]["lr"]
        logs["learning_rate"] = lr
```

### MetricsLogger

```python
class MetricsLogger(Callback):
    """Log metrics to file.
    
    Args:
        log_dir: Directory to save logs.
        filename: Log filename.
    
    Example:
        >>> callback = MetricsLogger(log_dir="logs")
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        filename: str = "metrics.json",
    ):
        pass
```

### GradientClipCallback

```python
class GradientClipCallback(Callback):
    """Clip gradients during training.
    
    Args:
        max_norm: Maximum gradient norm.
    """
    
    def __init__(self, max_norm: float = 1.0):
        pass
```

## Schedulers

### WarmupCosineScheduler

```python
class WarmupCosineScheduler:
    """Cosine annealing with linear warmup.
    
    Args:
        optimizer: PyTorch optimizer.
        warmup_epochs: Number of warmup epochs.
        total_epochs: Total training epochs.
        min_lr: Minimum learning rate.
    
    Example:
        >>> scheduler = WarmupCosineScheduler(
        ...     optimizer, warmup_epochs=5, total_epochs=100
        ... )
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
    ):
        pass
    
    def step(self, epoch: int):
        """Update learning rate."""
        pass
```

### OneCycleScheduler

```python
class OneCycleScheduler:
    """One-cycle learning rate policy.
    
    Args:
        optimizer: PyTorch optimizer.
        max_lr: Maximum learning rate.
        total_steps: Total training steps.
        pct_start: Percentage of cycle spent increasing LR.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
    ):
        pass
```

## Experiment Tracking

### ExperimentTracker (Base)

```python
class ExperimentTracker(ABC):
    """Abstract base class for experiment trackers."""
    
    @abstractmethod
    def start_run(self, run_name: Optional[str] = None):
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
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ):
        """Log metrics."""
        pass
    
    @abstractmethod
    def log_artifact(self, local_path: str):
        """Log an artifact file."""
        pass
    
    @abstractmethod
    def log_model(self, model: nn.Module, artifact_path: str):
        """Log a PyTorch model."""
        pass
```

### TrackedTrainer

```python
class TrackedTrainer:
    """Trainer wrapper with automatic experiment tracking.
    
    Args:
        trainer: Base trainer.
        tracker: Experiment tracker.
    
    Example:
        >>> tracked = TrackedTrainer(trainer, tracker)
        >>> tracked.train(train_loader, val_loader, epochs=100)
    """
    
    def __init__(self, trainer: Trainer, tracker: ExperimentTracker):
        pass
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        run_name: Optional[str] = None,
        config: Optional[Dict] = None,
    ):
        """Train with automatic tracking."""
        pass
```

## Utility Functions

### create_optimizer

```python
def create_optimizer(
    model: nn.Module,
    config: TrainingConfig,
) -> torch.optim.Optimizer:
    """Create optimizer from config.
    
    Args:
        model: Model to optimize.
        config: Training configuration.
    
    Returns:
        Configured optimizer.
    """
    pass
```

### create_scheduler

```python
def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
) -> Optional[Any]:
    """Create learning rate scheduler from config.
    
    Args:
        optimizer: Optimizer to schedule.
        config: Training configuration.
    
    Returns:
        Configured scheduler or None.
    """
    pass
```

### compute_metrics

```python
def compute_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    y_prob: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute classification metrics.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_prob: Prediction probabilities (for AUC).
    
    Returns:
        Dictionary with accuracy, f1, precision, recall, auc.
    """
    pass
```
