# Installation

## Requirements

- Python 3.10 or higher
- PyTorch 2.0 or higher
- CUDA 11.8+ (optional, for GPU acceleration)

## Installation Methods

### Using pip (Recommended)

```bash
pip install metapathpredict
```

With optional dependencies:

```bash
# Development dependencies
pip install metapathpredict[dev]

# All optional dependencies
pip install metapathpredict[all]
```

### Using Conda

```bash
# Create environment
conda env create -f envs/environment.yml
conda activate metapathpredict

# Install package
pip install -e .
```

### From Source

```bash
# Clone repository
git clone https://github.com/metapathpredict/metapathpredict.git
cd metapathpredict

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install in development mode
pip install -e ".[dev]"
```

## GPU Support

### CUDA Installation

For GPU training, ensure you have CUDA installed:

```bash
# Check CUDA version
nvidia-smi

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Verifying Installation

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Optional Dependencies

| Package | Purpose | Install |
|---------|---------|---------|
| `mlflow` | Experiment tracking | `pip install mlflow` |
| `wandb` | Weights & Biases tracking | `pip install wandb` |
| `ray[tune]` | Hyperparameter tuning | `pip install ray[tune]` |
| `dagster` | Pipeline orchestration | `pip install dagster` |

## Troubleshooting

### Common Issues

#### CUDA Out of Memory

Reduce batch size in configuration:

```yaml
training:
  batch_size: 16  # Reduce from 32
```

#### Import Errors

Ensure all dependencies are installed:

```bash
pip install -e ".[all]"
```

#### Permission Denied (Linux)

```bash
chmod +x scripts/*.sh
```

## Next Steps

- [Quick Start Guide](quickstart.md) - Get started with your first model
- [Configuration](configuration.md) - Learn about configuration options
