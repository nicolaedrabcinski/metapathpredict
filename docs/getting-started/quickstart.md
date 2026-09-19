# Quick Start

This guide will help you train your first model in under 5 minutes.

## 1. Prepare Your Data

MetaPathPredict expects FASTA files with sequences organized by class:

```
data/input/
├── bacteria.fasta
├── viruses.fasta
└── eucaryotic.fasta
```

### Convert to HDF5

```bash
metapathpredict prepare \
    --input data/input \
    --output data/datasets/unified \
    --max-length 500 \
    --train-ratio 0.8
```

Or using Python:

```python
from metapathpredict.data import prepare_dataset

prepare_dataset(
    input_dir="data/input",
    output_dir="data/datasets/unified",
    max_length=500,
    train_ratio=0.8,
)
```

## 2. Create Configuration

Create a `config.yaml` file:

```yaml
model:
  architecture: cnn
  kernel_preset: medium  # small=5, medium=7, large=10
  hidden_channels: [32, 64, 128]
  dropout: 0.3

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  scheduler: cosine
  early_stopping: true
  patience: 10

data:
  max_length: 500
  augmentation: true

experiment:
  tracker: none  # or "mlflow", "wandb"
  experiment_name: my_first_model
```

## 3. Train Model

### Using CLI

```bash
metapathpredict train \
    --config config.yaml \
    --data data/datasets/unified
```

# Load configuration
config = Config.from_yaml("config.yaml")

# Create data module
datamodule = SequenceDataModule(
    data_dir="data/datasets/unified",
    batch_size=config.training.batch_size,
)
datamodule.setup()

# Create model
model = ConfigurableCNN(
    input_channels=4,
    num_classes=3,
    kernel_preset=config.model.kernel_preset,
    hidden_channels=config.model.hidden_channels,
)

# Train
trainer = Trainer(model, config)
trainer.fit(
    datamodule.train_dataloader(),
    datamodule.val_dataloader(),
)

# Save model
torch.save(model.state_dict(), "model.pt")
```

## 4. Make Predictions

### Using CLI

```bash
metapathpredict predict \
    --model checkpoints/best_model.pt \
    --input sequences.fasta \
    --output predictions.csv
```

### Using Python

```python
from metapathpredict.inference import Predictor

# Load predictor
predictor = Predictor.from_checkpoint("checkpoints/best_model.pt")

# Predict from FASTA
results = predictor.predict_fasta("sequences.fasta")

# Print results
for seq_id, prediction in results.items():
    print(f"{seq_id}: {prediction['class']} ({prediction['confidence']:.2%})")
```

## 5. View Results

Training metrics are saved to `outputs/metrics.json`:

```python
import json

with open("outputs/metrics.json") as f:
    metrics = json.load(f)

print(f"Best validation accuracy: {metrics['best_val_accuracy']:.2%}")
print(f"Training time: {metrics['training_time']:.1f} seconds")
```

## Quick Training Examples

### Fast Training (Testing)

```bash
metapathpredict train \
    --config config.yaml \
    --epochs 5 \
    --batch-size 64
```

### Production Training

```bash
metapathpredict train \
    --config config.yaml \
    --epochs 200 \
    --use-amp \
    --early-stopping
```

### GPU Training

```bash
CUDA_VISIBLE_DEVICES=0 metapathpredict train --config config.yaml
```

## Next Steps

- [Configuration Guide](configuration.md) - Detailed configuration options
- [Training Approaches](../training-approaches/overview.md) - Learn about CNN, Contrastive, and RL
- [Experiment Tracking](../user-guide/tracking.md) - Set up MLflow or W&B
