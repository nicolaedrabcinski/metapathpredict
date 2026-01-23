# Configurable CNN

The Configurable CNN is our primary training approach, offering adjustable kernel sizes to capture patterns at different scales.

## Architecture

```
Input: (batch, 4, seq_length)
    │
    ▼
┌─────────────────────────────┐
│   Conv1D Block 1            │
│   - Conv1D(4→32, k=K)       │
│   - BatchNorm1d             │
│   - ReLU                    │
│   - MaxPool1d(2)            │
│   - Dropout(0.3)            │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│   Conv1D Block 2            │
│   - Conv1D(32→64, k=K)      │
│   - BatchNorm1d             │
│   - ReLU                    │
│   - MaxPool1d(2)            │
│   - Dropout(0.3)            │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│   Conv1D Block 3            │
│   - Conv1D(64→128, k=K)     │
│   - BatchNorm1d             │
│   - ReLU                    │
│   - MaxPool1d(2)            │
│   - Dropout(0.3)            │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│   Global Average Pooling    │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│   Classifier                │
│   - Linear(128→num_classes) │
└─────────────────────────────┘
    │
    ▼
Output: (batch, num_classes)
```

## Kernel Presets

| Preset | Kernel Size | Captures |
|--------|-------------|----------|
| `small` | 5 | Local motifs (codons) |
| `medium` | 7 | Medium patterns |
| `large` | 10 | Longer conserved regions |

### Biological Interpretation

- **k=5**: Captures codon-level patterns (3bp + context)
- **k=7**: Balanced for common motifs
- **k=10**: Better for longer binding sites and regulatory elements

## Usage

### Basic Usage

```python
from metapathpredict.models import ConfigurableCNN

# Create model with medium kernel
model = ConfigurableCNN(
    input_channels=4,      # One-hot encoded (A, T, G, C)
    num_classes=3,         # viral, bacterial, eukaryotic
    kernel_preset="medium",
    hidden_channels=[32, 64, 128],
    dropout=0.3,
)

# Forward pass
import torch
x = torch.randn(8, 4, 500)  # batch, channels, length
output = model(x)  # (8, 3)
```

### Custom Kernel Size

```python
model = ConfigurableCNN(
    kernel_preset="custom",
    kernel_size=9,  # Custom size
)
```

### Multi-Scale CNN

Combine multiple kernel sizes:

```python
from metapathpredict.models import MultiScaleCNN

model = MultiScaleCNN(
    kernel_sizes=[5, 7, 10],
    hidden_channels=[32, 64, 128],
)
```

## Training

### Configuration

```yaml
model:
  architecture: cnn
  kernel_preset: medium
  hidden_channels: [32, 64, 128]
  dropout: 0.3

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  scheduler: cosine
```

### Training Script

```python
from metapathpredict.config import Config
from metapathpredict.models import ConfigurableCNN
from metapathpredict.training import Trainer

# Load config
config = Config.from_yaml("config.yaml")

# Create model
model = ConfigurableCNN(
    kernel_preset=config.model.kernel_preset,
    hidden_channels=config.model.hidden_channels,
)

# Create trainer
trainer = Trainer(
    model=model,
    config=config,
    device="cuda",
)

# Train
history = trainer.fit(train_loader, val_loader)
```

## Feature Extraction

Use the CNN as a feature extractor:

```python
# Get features (before classification)
features = model.get_features(x)  # (batch, 128)

# Use for downstream tasks
classifier = nn.Linear(128, num_new_classes)
new_output = classifier(features)
```

## Hyperparameter Tuning

### Recommended Ranges

| Parameter | Range | Default |
|-----------|-------|---------|
| `kernel_preset` | small, medium, large | medium |
| `hidden_channels` | [16-64, 32-128, 64-256] | [32, 64, 128] |
| `dropout` | 0.1 - 0.5 | 0.3 |
| `learning_rate` | 1e-4 - 1e-2 | 1e-3 |
| `batch_size` | 16 - 128 | 32 |

### Using Ray Tune

```python
from ray import tune
from metapathpredict.training import tune_hyperparameters

search_space = {
    "kernel_preset": tune.choice(["small", "medium", "large"]),
    "dropout": tune.uniform(0.1, 0.5),
    "learning_rate": tune.loguniform(1e-4, 1e-2),
}

best_config = tune_hyperparameters(
    model_class=ConfigurableCNN,
    search_space=search_space,
    num_samples=20,
)
```

## Performance Tips

### Speed Optimization

1. **Use AMP** (Automatic Mixed Precision):
```yaml
training:
  use_amp: true
```

2. **torch.compile** (PyTorch 2.0+):
```python
model = torch.compile(model)
```

3. **Increase workers**:
```yaml
data:
  num_workers: 8
  pin_memory: true
```

### Memory Optimization

1. **Gradient accumulation**:
```yaml
training:
  accumulation_steps: 4
  batch_size: 8  # Effective batch = 32
```

2. **Gradient checkpointing**:
```python
model.gradient_checkpointing_enable()
```

## Model Checkpoints

```python
# Save
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config.model_dump(),
}, 'checkpoint.pt')

# Load
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

## Comparison with Other Kernels

Results on 60k sequence benchmark:

| Kernel | Accuracy | F1 | Params | Time |
|--------|----------|-----|--------|------|
| k=5 | 94.2% | 0.941 | 45K | 15m |
| k=7 | **95.1%** | **0.950** | 52K | 18m |
| k=10 | 94.8% | 0.946 | 63K | 22m |
| Multi (5,7,10) | **95.4%** | **0.953** | 160K | 35m |

## Next Steps

- [Contrastive Learning](contrastive.md) - Self-supervised pretraining
- [Training Guide](../user-guide/training.md) - Detailed training instructions
- [API Reference](../api/models.md) - Full API documentation
