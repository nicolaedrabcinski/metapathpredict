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

# add genomes to reach the given totals per RefSeq group; each new genome is picked from the
# family with the fewest genomes so far (--dry-run shows the plan without downloading)
python scripts/download_diverse_genomes.py --extend plant=120,invertebrate=140,vertebrate_other=80,vertebrate_mammalian=40
```

NCBI lineages of the genomes are cached in `data/genomes/lineages.json` (fetched on demand).

### 2. Prepare dataset (genomes → HDF5)

Whole genomes are assigned to train/val/test, and by default whole **families** stay together
(`--split-by family`): val and test hold organisms with no relative of the same family in training.
A species-only split (`--split-by genome`) still lets a genus or family sit on both sides, which
makes accuracy look better than it is on new organisms. The script verifies that no species and no
family appears in two splits. `--split-seed` gives a different random split; use several to see how
much the results depend on which genomes ended up in the test set.

```bash
metapathpredict prepare \
    --manifest data/genomes/manifest.tsv \
    --config configs/train_gpu.yaml \
    --output data/datasets/taxa8fam_s1 \
    --length 500 \
    --fragments-per-class 30000 \
    --split-by family --split-seed 1
```

`split_assignments.tsv` in the output directory lists every genome with its split and lineage.
For a dataset made before lineages were recorded: `python scripts/annotate_lineages.py <dataset_dir>`.

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
    --model data/weights/taxa8/contrastive_best.pt
```

In our own runs a plain CNN trained by `scripts/baselines.py supervised` (cross-entropy, no
contrastive pretraining) has consistently matched or beaten this pipeline's `contrastive_best.pt`,
and RL fine-tuning (`rl_best.pt`) has never beaten the plain contrastive checkpoint either — see
"Reference points and model comparisons" below. `predict`/`evaluate` accept checkpoints from either
source (`contrastive_best.pt`, `rl_best.pt`, or a `scripts/baselines.py --save-checkpoint` model.pt).

### 5. Evaluate

```bash
metapathpredict evaluate \
    --model data/weights/taxa8/contrastive_best.pt \
    --data data/datasets/taxa8/encoded_test_500.hdf5
```

Fragments of one genome are right or wrong together, so a plain accuracy overstates how certain
it is. With `test_fragments.fasta` next to the test file, `evaluate` (and every run made by
`scripts/run_experiment.py`) also reports a 95% bootstrap interval that resamples whole genomes, and
accuracy split by how close the nearest training genome of the class is (genus, family, order, ...;
"near" = genus or family, "far" = the rest).

Explaining a trained CNN classifier (`metapathpredict.explain`, figures only):

```bash
python scripts/explain.py                                    # writes figures/xai/*.png, data/*.csv, summary.json
python scripts/explain.py --model experiments/baselines/<run>/model.pt --data-dir data/datasets/<split> \
    --fragment-size 500 --out figures/xai_500                # another model / dataset
```

Attribution maps (Integrated Gradients, gradient x input, Grad-CAM), deletion curves and a
random-weights check that say whether the maps can be trusted, a shuffle test (how much of the
prediction is base or dinucleotide composition), predictions by GC content, first-layer motifs and
the test genomes the model gets most wrong. Needs a model saved with
`scripts/baselines.py supervised --save-checkpoint`.

Reference points and model comparisons:

```bash
python scripts/baselines.py kmer --k 4                       # 4-mer composition + gradient boosting
python scripts/baselines.py supervised --augment rc          # same CNN, plain cross-entropy
python scripts/compare_predictions.py A.npy B.npy --only far # is B better than A? paired bootstrap over genomes
```

## Training Pipelines

| Pipeline | Command | Description |
|----------|---------|-------------|
| `full` | `--pipeline full` | Contrastive pretrain → RL fine-tune (default) |
| `contrastive` | `--pipeline contrastive` | Contrastive pretraining only |
| `rl` | `--pipeline rl` | RL training only (needs encoder checkpoint) |

A plain supervised CNN (no contrastive pretraining, no RL) is trained separately via
`scripts/baselines.py supervised`, not through `metapathpredict train` — see "Reference points and
model comparisons" above. It is currently our best-performing recipe.

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
│   │   └── configurable_cnn.py # Backbone CNN with presets (used by scripts/baselines.py too)
│   ├── data/
│   │   ├── dataset.py          # HDF5 dataset loaders
│   │   ├── datamodule.py       # DataModule with train/val/test splits
│   │   └── preprocessing.py    # One-hot encoding, fragmentation
│   ├── baselines.py            # Plain CNN / k-mer reference models (scripts/baselines.py)
│   ├── explain.py              # Attribution methods (scripts/explain.py)
│   ├── genome_eval.py          # Genome-level bootstrap evaluation
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

## Experiment tracking

Experiments run through Hydra (`scripts/run_experiment.py`) are tracked twice: each run gets its own
directory under `experiments/` (resolved config, log, test scores) and one MLflow run with the same
parameters, per-epoch metrics (losses, alignment, uniformity, effective rank, probe/RL accuracy),
CPU/GPU/memory metrics and the test-split scores per class.

```bash
python scripts/run_experiment.py experiment_name=bs512 contrastive.batch_size=512
python scripts/run_experiment.py -m experiment_name=lr contrastive.learning_rate=1e-4,3e-4   # a sweep
mlflow ui --backend-store-uri sqlite:///mlflow.db                # http://localhost:5000
python scripts/import_runs_to_mlflow.py                           # add finished runs from their logs
```

Use `tracking.enabled=false` to skip MLflow. Plain `metapathpredict train` does not log to MLflow.

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
