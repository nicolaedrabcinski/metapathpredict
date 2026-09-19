# MetaPathPredict

Deep learning framework for metagenomic DNA sequence classification using **Contrastive Learning** and **Deep Reinforcement Learning**.

Classifies DNA fragments into three categories: **bacteria**, **eukaryotic**, and **virus**.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-184%20passed-brightgreen.svg)]()

## How It Works

```
FASTA files ──> One-hot encode ──> Contrastive Pretrain (SimCLR/SupCon)
                   (ACGT → 4ch)         │
                                         ▼
                                   Transfer encoder weights
                                         │
                                         ▼
                                   RL Fine-tune (Actor-Critic/DQN/REINFORCE)
                                         │
                                         ▼
                                   bacteria / eukaryotic / virus
```

**Pipeline:** The CNN backbone is first pretrained with contrastive learning to learn robust DNA representations, then fine-tuned with reinforcement learning for classification.

## Installation

```bash
git clone https://github.com/nicolaedrabcinski/metapathpredict.git
cd metapathpredict
pip install -e .
```

With all optional dependencies:

```bash
pip install -e ".[all]"
```

## Quick Start

### 1. Download genomes

One genome per species (at most two per genus) across eight NCBI RefSeq groups:
bacteria, archaea, fungi, protozoa, plant, invertebrate, vertebrate, virus (~30 GB).

```bash
python scripts/download_diverse_genomes.py --output data/genomes
```

### 2. Prepare dataset (genomes → HDF5)

Whole genomes are assigned to train/val/test, so val and test contain species the model
never saw in training. The script verifies that no species appears in two splits.

```bash
metapathpredict prepare \
    --manifest data/genomes/manifest.tsv \
    --config configs/train_gpu.yaml \
    --output data/datasets/taxa8 \
    --length 500 \
    --fragments-per-class 30000
```

### 3. Train (Full Pipeline)

```bash
# CPU
metapathpredict train --pipeline full --config configs/train_cpu.yaml

# GPU
metapathpredict train --pipeline full --config configs/train_gpu.yaml --device cuda
```

This runs:
1. **Contrastive pretraining** — learns DNA representations via NT-Xent loss, then fits a linear probe on the frozen encoder
2. **RL fine-tuning** — transfers encoder weights to Actor-Critic agent

Checkpoints saved to `data/weights/taxa8/`:
- `contrastive_best.pt` — best encoder
- `rl_best.pt` — best RL agent

### 4. Predict

```bash
metapathpredict predict \
    --input sequences.fasta \
    --model data/weights/taxa8/rl_best.pt
```

### 5. Evaluate

```bash
metapathpredict evaluate \
    --model data/weights/taxa8/rl_best.pt \
    --data data/datasets/taxa8/encoded_test_500.hdf5
```

## Training Pipelines

| Pipeline | Command | Description |
|----------|---------|-------------|
| `full` | `--pipeline full` | Contrastive pretrain → RL fine-tune (default) |
| `contrastive` | `--pipeline contrastive` | Contrastive pretraining only |
| `rl` | `--pipeline rl` | RL training only (needs encoder checkpoint) |
| `supervised` | `--pipeline supervised` | Legacy supervised CNN |

## Architecture

### Contrastive Learning (Phase 1)

- **Backbone:** ConfigurableCNN (`small`/`medium`/`large` presets)
- **Projection head:** 3-layer MLP with BatchNorm
- **Loss:** NTXent (SimCLR) or SupCon (supervised contrastive)
- **Augmentations:** Random mutation, random masking, reverse complement

### Reinforcement Learning (Phase 2)

- **Algorithms:** DQN, REINFORCE (Policy Gradient), Actor-Critic
- **Weight transfer:** Encoder weights from contrastive phase → RL agent backbone (`strict=False`)
- **Environment:** Each DNA fragment is a state, classification is the action

### CNN Backbone Presets

| Preset | Layers | Base Channels | Parameters |
|--------|--------|---------------|------------|
| `small` | 3 conv blocks | 64 | ~250K |
| `medium` | 4 conv blocks | 64 | ~500K |
| `large` | 5 conv blocks | 128 | ~2M |

## Configuration

### CPU Config (`configs/train_cpu.yaml`)

```yaml
contrastive:
  backbone: "medium"
  num_epochs: 20
  batch_size: 128
  learning_rate: 0.001

rl:
  algorithm: "actor_critic"
  num_epochs: 10
  episodes_per_epoch: 1000
```

### GPU Config (`configs/train_gpu.yaml`)

```yaml
contrastive:
  backbone: "large"
  num_epochs: 50
  batch_size: 512
  learning_rate: 0.001

rl:
  algorithm: "actor_critic"
  num_epochs: 30
  episodes_per_epoch: 5000

training:
  use_amp: true  # Mixed precision
```

## Docker

```bash
# Full stack (MinIO + PostgreSQL + Ray + Dagster + App)
docker compose up -d

# CPU-only
docker compose -f docker-compose.cpu.yml up -d

# With monitoring (Prometheus + Grafana)
docker compose --profile monitoring up -d
```

| Service | Port | Description |
|---------|------|-------------|
| App | 8000 | MetaPathPredict API |
| MinIO | 9000/9001 | S3-compatible storage |
| PostgreSQL | 5432 | Metadata catalog |
| Ray Dashboard | 8265 | Distributed compute |
| Dagster UI | 3000 | Pipeline orchestration |
| Prometheus | 9090 | Metrics |
| Grafana | 3001 | Dashboards |

## Dashboard (Frontend)

React + TypeScript dashboard for model interpretability:

```bash
cd frontend
npm install
npm run dev
```

Features: attribution heatmaps, prediction comparison, sequence viewer, motif analysis.

## Project Structure

```
metapathpredict/
├── src/metapathpredict/
│   ├── models/
│   │   ├── contrastive.py      # ContrastiveEncoder, NTXent, SupCon, augmentation
│   │   ├── reinforcement.py    # DQN, PolicyGradient, ActorCritic, RLTrainer
│   │   ├── configurable_cnn.py # Backbone CNN with presets
│   │   └── unified.py          # Legacy supervised classifier
│   ├── data/
│   │   ├── dataset.py          # HDF5 dataset loaders
│   │   ├── datamodule.py       # DataModule with train/val/test splits
│   │   └── preprocessing.py    # One-hot encoding, fragmentation
│   ├── training/               # Legacy supervised trainer, callbacks
│   ├── api/                    # FastAPI backend
│   ├── config/                 # Pydantic settings
│   └── cli.py                  # CLI entry point
├── frontend/                   # React dashboard
├── configs/
│   ├── train_cpu.yaml
│   └── train_gpu.yaml
├── scripts/
│   ├── download_diverse_genomes.py  # Genome downloader (manifest)
│   └── run_experiment.py       # Hydra experiment runner
├── docker/                     # Dockerfiles
├── tests/                      # 184+ tests
└── pyproject.toml
```

## Testing

```bash
# All tests
pytest

# With coverage
pytest --cov=src/metapathpredict

# Specific module
pytest tests/test_contrastive.py
pytest tests/test_reinforcement.py
pytest tests/test_pipeline.py
```

## Python API

```python
from metapathpredict.models import (
    ContrastiveEncoder,
    ContrastiveTrainer,
    ContrastiveAugmentation,
    ActorCriticAgent,
    RLTrainer,
    SequenceEnvironment,
)

# Contrastive pretraining
encoder = ContrastiveEncoder(in_channels=4, backbone="medium")
trainer = ContrastiveTrainer(encoder, optimizer, augmentation, temperature=0.07)
loss = trainer.train_epoch(dataloader)

# RL fine-tuning
agent = ActorCriticAgent(in_channels=4, num_actions=3, backbone="medium")
agent.load_state_dict(encoder_weights, strict=False)  # Transfer weights
env = SequenceEnvironment(sequences, labels)
rl_trainer = RLTrainer(agent, env, optimizer, algorithm="actor_critic")
metrics = rl_trainer.train_epoch(num_episodes=1000)
```

## License

MIT License — see [LICENSE](LICENSE) for details.
