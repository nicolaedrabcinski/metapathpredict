# MetaPathPredict

Modern DNA sequence classification using deep learning with PyTorch 2.0+.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 🧬 **DNA Sequence Classification** - Classify sequences as virus, bacteria, or eukaryotic
- 🚀 **Modern PyTorch 2.0** - Mixed precision training, torch.compile support
- 🎯 **Three Training Approaches**:
  - **CNN** with configurable kernel sizes (5, 7, 10, multi-scale)
  - **Contrastive Learning** (SimCLR, SupCon) for powerful embeddings
  - **Deep Reinforcement Learning** (DQN, Policy Gradient, Actor-Critic)
- 📊 **Data Engineering Stack** - DuckDB, MinIO, Ray, Dagster integration
- ⚡ **High Performance** - Distributed training, GPU acceleration

## Installation

### Basic Installation

```bash
# Clone repository
git clone https://github.com/yourusername/metapathpredict.git
cd metapathpredict

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install package
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

### Full Installation (with DE stack)

```bash
pip install -e ".[dev,pipeline,distributed,orchestration]"
```

## Quick Start

### 1. Prepare Data

```bash
python -m metapathpredict prepare \
    --input data/input \
    --output data/datasets/unified \
    --fragment-size 1000
```

### 2. Train Model

**Standard CNN:**
```bash
python -m metapathpredict train \
    --config config.yaml \
    --model-type unified
```

**Configurable CNN (kernel 5, 7, 10):**
```bash
# Small kernel (5)
python -m metapathpredict train --model-type configurable_cnn --kernel-preset small

# Medium kernel (7)
python -m metapathpredict train --model-type configurable_cnn --kernel-preset medium

# Large kernel (10)
python -m metapathpredict train --model-type configurable_cnn --kernel-preset large

# Multi-scale (5, 7, 10 parallel)
python -m metapathpredict train --model-type configurable_cnn --kernel-preset multi
```

**Contrastive Learning:**
```bash
python -m metapathpredict train --model-type contrastive
```

**Reinforcement Learning:**
```bash
# DQN
python -m metapathpredict train --model-type rl --rl-algorithm dqn

# Policy Gradient (REINFORCE)
python -m metapathpredict train --model-type rl --rl-algorithm policy_gradient

# Actor-Critic
python -m metapathpredict train --model-type rl --rl-algorithm actor_critic
```

### 3. Run Predictions

```bash
python -m metapathpredict predict \
    --input data/input/test.fasta \
    --output data/output/predictions \
    --weights data/weights/unified/best_model.pt
```

## Training Approaches

### 1. Configurable CNN

CNN architecture with variable kernel sizes for capturing different k-mer patterns:

| Preset | Kernel Size | Best For |
|--------|-------------|----------|
| `small` | 5 | Short motifs, rapid patterns |
| `medium` | 7 | Balanced feature extraction |
| `large` | 10 | Long-range dependencies |
| `multi` | 5, 7, 10 | Multi-scale feature fusion |

```python
from metapathpredict.models import create_configurable_cnn

model = create_configurable_cnn(
    preset="multi",
    in_channels=4,
    num_classes=3,
    hidden_channels=128,
)
```

### 2. Contrastive Learning

SimCLR-style self-supervised learning with DNA-specific augmentations:

- Reverse complement
- Random mutations
- Random masking

```python
from metapathpredict.models import ContrastiveEncoder, NTXentLoss, SupConLoss

encoder = ContrastiveEncoder(
    in_channels=4,
    embedding_dim=256,
    projection_dim=128,
)

# Self-supervised
loss_fn = NTXentLoss(temperature=0.5)

# Supervised contrastive
loss_fn = SupConLoss(temperature=0.5)
```

### 3. Deep Reinforcement Learning

Sequence classification as a reinforcement learning problem:

```python
from metapathpredict.models import DQNAgent, PolicyGradientAgent, ActorCriticAgent

# DQN with experience replay
agent = DQNAgent(
    state_dim=(4, 1000),
    num_actions=3,
    hidden_dim=128,
)

# Policy Gradient (REINFORCE)
agent = PolicyGradientAgent(
    state_dim=(4, 1000),
    num_actions=3,
)

# Actor-Critic (A2C)
agent = ActorCriticAgent(
    state_dim=(4, 1000),
    num_actions=3,
)
```

## Data Engineering Stack

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Dagster                               │
│                   (Orchestration)                            │
├─────────────────────────────────────────────────────────────┤
│      Ray Cluster          │         DuckDB/DuckLake         │
│   (Distributed Training)  │      (Analytics/Catalog)        │
├─────────────────────────────────────────────────────────────┤
│                     MinIO (S3 Storage)                       │
│              Models | Datasets | Checkpoints                 │
├─────────────────────────────────────────────────────────────┤
│                      PostgreSQL                              │
│                  (Metadata Catalog)                          │
└─────────────────────────────────────────────────────────────┘
```

### Start DE Stack

```bash
# Basic stack
docker-compose up -d

# With monitoring (Prometheus + Grafana)
docker-compose --profile monitoring up -d

# With data governance (DataHub)
docker-compose --profile governance up -d
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| MinIO | 9000/9001 | S3-compatible storage |
| PostgreSQL | 5432 | Metadata database |
| Ray Dashboard | 8265 | Ray cluster UI |
| Dagster UI | 3000 | Pipeline orchestration |
| Prometheus | 9090 | Metrics collection |
| Grafana | 3001 | Monitoring dashboards |

### Distributed Training with Ray

```python
from metapathpredict.pipeline import RayTrainer, HyperparameterTuner

# Distributed training
trainer = RayTrainer(config_path="config.yaml")
trainer.train(num_workers=4, use_gpu=True)

# Hyperparameter tuning
tuner = HyperparameterTuner()
best_config = tuner.tune(num_samples=50)
```

### Run Dagster Pipeline

```bash
# Start Dagster UI
dagster dev -m metapathpredict.pipeline.dagster_assets

# Execute pipeline
dagster job execute -m metapathpredict.pipeline.dagster_assets -j training_pipeline_job
```

## Configuration

### YAML Configuration

```yaml
# config.yaml
paths:
  data_dir: /app/data
  weights_dir: /app/data/weights

data:
  sequence_length: 1000
  batch_size: 64
  train_split: 0.8

model:
  type: unified
  kernel_preset: medium

training:
  epochs: 100
  learning_rate: 0.001
  use_mixed_precision: true
  early_stopping_patience: 15

pipeline:
  storage:
    endpoint_url: http://localhost:9000
    bucket: metapathpredict
  ray:
    address: ray://localhost:10001
```

## Project Structure

```
metapathpredict/
├── src/metapathpredict/
│   ├── config/           # Pydantic configuration
│   ├── data/             # Dataset & preprocessing
│   │   ├── preprocessing.py
│   │   ├── augmentation.py
│   │   └── dataset.py
│   ├── models/           # Neural network architectures
│   │   ├── cnn.py
│   │   ├── configurable_cnn.py
│   │   ├── contrastive.py
│   │   ├── reinforcement.py
│   │   └── unified.py
│   ├── training/         # Training logic
│   │   ├── trainer.py
│   │   ├── callbacks.py
│   │   └── schedulers.py
│   ├── inference/        # Prediction
│   │   ├── predictor.py
│   │   └── ensemble.py
│   ├── pipeline/         # DE stack integration
│   │   ├── duckdb_connector.py
│   │   ├── storage.py
│   │   ├── ray_training.py
│   │   └── dagster_assets.py
│   └── cli.py            # Command-line interface
├── tests/                # Unit tests
├── docker/               # Docker configurations
├── config.yaml           # Default configuration
├── docker-compose.yml    # DE stack deployment
└── pyproject.toml        # Package configuration
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/metapathpredict --cov-report=html

# Run specific test file
pytest tests/test_models.py

# Skip slow tests
pytest -m "not slow"
```

## Development

### Code Quality

```bash
# Lint
ruff check src/

# Format
ruff format src/

# Type check
mypy src/metapathpredict
```

### Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Performance Tips

1. **Enable Mixed Precision**: Set `use_mixed_precision: true` in config
2. **Use torch.compile**: Automatically enabled for PyTorch 2.0+
3. **Increase Workers**: Set `num_workers: 4` or higher for data loading
4. **Use Distributed Training**: Scale with Ray for multi-GPU/multi-node

## Citation

```bibtex
@software{metapathpredict2026,
  title = {MetaPathPredict: Modern DNA Sequence Classification},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/metapathpredict}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
