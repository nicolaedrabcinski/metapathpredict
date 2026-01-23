# Configuration

MetaPathPredict uses Pydantic v2 for type-safe configuration with validation.

## Configuration File Format

Configuration files use YAML format:

```yaml
# config.yaml
model:
  architecture: cnn
  kernel_preset: medium
  hidden_channels: [32, 64, 128]
  dropout: 0.3

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001

data:
  max_length: 500
  augmentation: true

experiment:
  tracker: mlflow
  experiment_name: my_experiment
```

## Configuration Sections

### Model Configuration

```yaml
model:
  # Architecture type: "cnn", "contrastive", "reinforcement"
  architecture: cnn
  
  # CNN kernel preset: "small" (5), "medium" (7), "large" (10)
  kernel_preset: medium
  
  # Hidden channel dimensions
  hidden_channels: [32, 64, 128]
  
  # Dropout rate (0.0 - 0.9)
  dropout: 0.3
  
  # Contrastive learning settings
  projection_dim: 128
  temperature: 0.5
  
  # RL settings
  rl_algorithm: reinforce  # "dqn", "reinforce", "a2c"
  hidden_dim: 256
```

### Training Configuration

```yaml
training:
  # Basic settings
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.0001
  
  # Learning rate scheduler
  scheduler: cosine  # "cosine", "step", "plateau", "none"
  warmup_epochs: 5
  min_lr: 0.000001
  
  # Mixed precision training
  use_amp: true
  
  # Early stopping
  early_stopping: true
  patience: 10
  
  # Gradient settings
  gradient_clip: 1.0
  accumulation_steps: 1
```

### Data Configuration

```yaml
data:
  # Sequence settings
  max_length: 500
  min_length: 100
  
  # Encoding type
  encoding: onehot  # "onehot", "kmer"
  kmer_size: 3
  
  # Augmentation
  augmentation: true
  aug_crop_ratio: 0.9
  aug_mask_ratio: 0.1
  aug_noise_std: 0.1
  
  # Data loading
  num_workers: 4
  pin_memory: true
  
  # Data splits
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
```

### Experiment Configuration

```yaml
experiment:
  # Tracker type: "mlflow", "wandb", "duckdb", "none"
  tracker: mlflow
  experiment_name: metapathpredict
  run_name: experiment_1
  tags:
    version: "2.0"
    dataset: unified
  
  # MLflow settings
  mlflow_tracking_uri: mlruns
  
  # W&B settings
  wandb_project: metapathpredict
  wandb_entity: my_team
  wandb_mode: online  # "online", "offline", "disabled"
  
  # DuckDB settings (local tracking)
  duckdb_path: experiments.duckdb
```

### Global Settings

```yaml
# Paths
data_dir: data
output_dir: outputs
checkpoint_dir: checkpoints

# Reproducibility
seed: 42
deterministic: false
```

## Loading Configuration

### From YAML File

```python
from metapathpredict.config import Config

config = Config.from_yaml("config.yaml")
```

### From Dictionary

```python
config = Config(
    model={"kernel_preset": "large"},
    training={"epochs": 200},
)
```

### From Environment Variables

```bash
export METAPATH_SEED=123
export METAPATH_TRACKER=mlflow
```

```python
from metapathpredict.config import Config

config = Config()  # Will use env vars
```

## Configuration Presets

### Quick Training (Testing)

```yaml
model:
  kernel_preset: small
  hidden_channels: [16, 32]

training:
  epochs: 5
  batch_size: 64
  early_stopping: false

data:
  max_length: 200
  augmentation: false
```

### Production Training

```yaml
model:
  kernel_preset: medium
  hidden_channels: [64, 128, 256, 512]
  dropout: 0.4

training:
  epochs: 200
  batch_size: 32
  learning_rate: 0.001
  scheduler: cosine
  warmup_epochs: 10
  early_stopping: true
  patience: 20
  use_amp: true

data:
  max_length: 1000
  augmentation: true
  num_workers: 8

experiment:
  tracker: mlflow
```

### Contrastive Pretraining

```yaml
model:
  architecture: contrastive
  kernel_preset: large
  projection_dim: 256
  temperature: 0.07

training:
  epochs: 100
  batch_size: 256  # Large batch for contrastive
  learning_rate: 0.0003
  scheduler: cosine
  warmup_epochs: 10

data:
  augmentation: true
  aug_crop_ratio: 0.85
  aug_mask_ratio: 0.15
```

## Validation

Configuration is automatically validated:

```python
from pydantic import ValidationError

try:
    config = Config(
        training={"learning_rate": -0.001}  # Invalid!
    )
except ValidationError as e:
    print(e)
    # learning_rate: Input should be greater than 0
```

## CLI Override

Override config values from CLI:

```bash
metapathpredict train \
    --config config.yaml \
    --epochs 50 \
    --batch-size 64 \
    --learning-rate 0.0001
```
