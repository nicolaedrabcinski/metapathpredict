# MetaPathPredict

<p align="center">
  <img src="assets/logo.png" alt="MetaPathPredict Logo" width="200">
</p>

<p align="center">
  <strong>Deep Learning for Metagenomic Sequence Classification</strong>
</p>

<p align="center">
  <a href="https://github.com/metapathpredict/metapathpredict/actions"><img src="https://github.com/metapathpredict/metapathpredict/workflows/CI/badge.svg" alt="CI Status"></a>
  <a href="https://pypi.org/project/metapathpredict/"><img src="https://img.shields.io/pypi/v/metapathpredict" alt="PyPI"></a>
  <a href="https://pypi.org/project/metapathpredict/"><img src="https://img.shields.io/pypi/pyversions/metapathpredict" alt="Python Version"></a>
  <a href="https://github.com/metapathpredict/metapathpredict/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
</p>

---

## Overview

MetaPathPredict is a modern deep learning framework for classifying metagenomic DNA sequences into their taxonomic origins (viral, bacterial, eukaryotic). Built with **PyTorch 2.0+** and featuring three distinct training approaches.

## Key Features

- 🧬 **DNA Sequence Classification** - Classify sequences as viral, bacterial, or eukaryotic
- 🚀 **PyTorch 2.0+** - Modern deep learning with AMP, torch.compile
- 🎯 **Three Training Approaches**:
    - Configurable CNN (kernel sizes 5, 7, 10)
    - Contrastive Learning (SimCLR/SupCon)
    - Deep Reinforcement Learning (DQN/A2C/PG)
- 📊 **Experiment Tracking** - MLflow, W&B, or DuckDB
- ☁️ **Cloud Ready** - Terraform configs for GCP deployment
- 🔧 **Type-Safe Config** - Pydantic v2 validation

## Quick Start

### Installation

```bash
pip install metapathpredict
```

Or with development dependencies:

```bash
pip install metapathpredict[dev]
```

### Basic Usage

```python
from metapathpredict.models import ConfigurableCNN
from metapathpredict.training import Trainer
from metapathpredict.config import Config

# Load configuration
config = Config.from_yaml("config.yaml")

# Create model
model = ConfigurableCNN(
    kernel_preset="medium",  # 5, 7, or 10
    num_classes=3,
)

# Train
trainer = Trainer(model, config)
trainer.fit(train_loader, val_loader)
```

### CLI Usage

```bash
# Prepare dataset
metapathpredict prepare --input data/input --output data/datasets

# Train model
metapathpredict train --config configs/train.yaml

# Predict
metapathpredict predict --input sequences.fasta --output predictions/
```

## Training Approaches

| Approach | Best For | Key Feature |
|----------|----------|-------------|
| **Configurable CNN** | Standard classification | Adjustable kernel sizes |
| **Contrastive Learning** | Limited labeled data | Self-supervised pretraining |
| **Deep RL** | Custom rewards | Exploration-based learning |

## Documentation Structure

- **[Getting Started](getting-started/installation.md)** - Installation and quick start
- **[User Guide](user-guide/data-preparation.md)** - Detailed usage instructions
- **[Training Approaches](training-approaches/overview.md)** - Deep dive into each method
- **[API Reference](api/config.md)** - Complete API documentation
- **[Deployment](deployment/docker.md)** - Docker and cloud deployment

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (optional, for GPU training)

## License

MIT License - see [LICENSE](https://github.com/metapathpredict/metapathpredict/blob/main/LICENSE) for details.
