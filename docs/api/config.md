# API Reference: Config

Configuration module using Pydantic v2.

## Config

::: metapathpredict.config.Config
    options:
      show_root_heading: true
      heading_level: 3

## ModelConfig

```python
from pydantic import BaseModel, Field

class ModelConfig(BaseModel):
    """Model architecture configuration."""
    
    architecture: Literal["cnn", "contrastive", "reinforcement"] = "cnn"
    """Architecture type: CNN, Contrastive Learning, or RL."""
    
    kernel_preset: Literal["small", "medium", "large"] = "medium"
    """CNN kernel size preset. small=5, medium=7, large=10."""
    
    hidden_channels: List[int] = Field(default=[32, 64, 128])
    """Hidden channel dimensions for conv layers."""
    
    dropout: float = Field(default=0.3, ge=0.0, le=0.9)
    """Dropout rate."""
    
    projection_dim: int = Field(default=128, ge=32)
    """Projection dimension for contrastive learning."""
    
    temperature: float = Field(default=0.5, gt=0.0)
    """Temperature for contrastive loss."""
    
    rl_algorithm: Literal["dqn", "reinforce", "a2c"] = "reinforce"
    """RL algorithm type."""
    
    hidden_dim: int = Field(default=256, ge=64)
    """Hidden dimension for RL networks."""
```

### Example

```python
from metapathpredict.config import ModelConfig

config = ModelConfig(
    architecture="cnn",
    kernel_preset="large",
    hidden_channels=[64, 128, 256],
    dropout=0.4,
)
```

## TrainingConfig

```python
class TrainingConfig(BaseModel):
    """Training hyperparameters."""
    
    epochs: int = Field(default=100, ge=1)
    """Number of training epochs."""
    
    batch_size: int = Field(default=32, ge=1)
    """Batch size for training."""
    
    learning_rate: float = Field(default=1e-3, gt=0.0)
    """Initial learning rate."""
    
    weight_decay: float = Field(default=1e-4, ge=0.0)
    """L2 regularization weight decay."""
    
    scheduler: Literal["cosine", "step", "plateau", "none"] = "cosine"
    """Learning rate scheduler type."""
    
    warmup_epochs: int = Field(default=5, ge=0)
    """Number of warmup epochs."""
    
    min_lr: float = Field(default=1e-6, ge=0.0)
    """Minimum learning rate for scheduler."""
    
    use_amp: bool = True
    """Use Automatic Mixed Precision."""
    
    early_stopping: bool = True
    """Enable early stopping."""
    
    patience: int = Field(default=10, ge=1)
    """Early stopping patience (epochs)."""
    
    gradient_clip: Optional[float] = Field(default=1.0, ge=0.0)
    """Gradient clipping max norm."""
    
    accumulation_steps: int = Field(default=1, ge=1)
    """Gradient accumulation steps."""
```

### Example

```python
from metapathpredict.config import TrainingConfig

config = TrainingConfig(
    epochs=200,
    batch_size=64,
    learning_rate=3e-4,
    scheduler="cosine",
    warmup_epochs=10,
    early_stopping=True,
    patience=20,
)
```

## DataConfig

```python
class DataConfig(BaseModel):
    """Data pipeline configuration."""
    
    max_length: int = Field(default=500, ge=100, le=10000)
    """Maximum sequence length."""
    
    min_length: int = Field(default=100, ge=50)
    """Minimum sequence length."""
    
    encoding: Literal["onehot", "kmer"] = "onehot"
    """Sequence encoding type."""
    
    kmer_size: int = Field(default=3, ge=1, le=7)
    """K-mer size for kmer encoding."""
    
    augmentation: bool = True
    """Enable data augmentation."""
    
    aug_crop_ratio: float = Field(default=0.9, gt=0.5, le=1.0)
    """Random crop ratio."""
    
    aug_mask_ratio: float = Field(default=0.1, ge=0.0, le=0.3)
    """Random mask ratio."""
    
    aug_noise_std: float = Field(default=0.1, ge=0.0)
    """Gaussian noise standard deviation."""
    
    num_workers: int = Field(default=4, ge=0)
    """DataLoader worker count."""
    
    pin_memory: bool = True
    """Pin memory for GPU transfer."""
    
    train_ratio: float = Field(default=0.8, gt=0.0, lt=1.0)
    """Training set ratio."""
    
    val_ratio: float = Field(default=0.1, gt=0.0, lt=1.0)
    """Validation set ratio."""
    
    test_ratio: float = Field(default=0.1, gt=0.0, lt=1.0)
    """Test set ratio."""
```

## ExperimentConfig

```python
class ExperimentConfig(BaseModel):
    """Experiment tracking configuration."""
    
    tracker: Literal["mlflow", "wandb", "duckdb", "none"] = "none"
    """Experiment tracker type."""
    
    experiment_name: str = "metapathpredict"
    """Experiment name."""
    
    run_name: Optional[str] = None
    """Run name (auto-generated if None)."""
    
    tags: Dict[str, str] = Field(default_factory=dict)
    """Tags for the experiment run."""
    
    mlflow_tracking_uri: str = "mlruns"
    """MLflow tracking server URI."""
    
    wandb_project: str = "metapathpredict"
    """W&B project name."""
    
    wandb_entity: Optional[str] = None
    """W&B team/entity name."""
    
    wandb_mode: Literal["online", "offline", "disabled"] = "online"
    """W&B sync mode."""
    
    duckdb_path: str = "experiments.duckdb"
    """DuckDB database path."""
```

## Full Config

```python
class Config(BaseModel):
    """Complete configuration."""
    
    model: ModelConfig = Field(default_factory=ModelConfig)
    """Model configuration."""
    
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    """Training configuration."""
    
    data: DataConfig = Field(default_factory=DataConfig)
    """Data configuration."""
    
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    """Experiment tracking configuration."""
    
    data_dir: str = "data"
    """Data directory path."""
    
    output_dir: str = "outputs"
    """Output directory path."""
    
    checkpoint_dir: str = "checkpoints"
    """Checkpoint directory path."""
    
    seed: int = Field(default=42)
    """Random seed for reproducibility."""
    
    deterministic: bool = False
    """Enable deterministic mode."""
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        pass
    
    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        pass
```

### Example

```python
from metapathpredict.config import Config

# Create from YAML
config = Config.from_yaml("config.yaml")

# Create programmatically
config = Config(
    model=ModelConfig(kernel_preset="large"),
    training=TrainingConfig(epochs=200),
    seed=123,
)

# Save to YAML
config.to_yaml("new_config.yaml")

# Access nested config
print(config.model.kernel_preset)
print(config.training.learning_rate)
```

## Validation

Pydantic v2 automatically validates all fields:

```python
from pydantic import ValidationError

try:
    config = TrainingConfig(learning_rate=-0.001)
except ValidationError as e:
    print(e)
    # learning_rate: Input should be greater than 0
```

## Environment Variables

Override config with environment variables:

```bash
export METAPATH_SEED=123
export METAPATH_TRACKER=mlflow
```

```python
from pydantic_settings import BaseSettings

class EnvConfig(BaseSettings):
    seed: int = 42
    tracker: str = "none"
    
    class Config:
        env_prefix = "METAPATH_"
```
