# Contributing to MetaPathPredict

Thank you for your interest in contributing to MetaPathPredict! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. We expect all contributors to:

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community and project

## Getting Started

### Prerequisites

- Python 3.10+
- Git
- Conda or pip for environment management
- CUDA-capable GPU (optional, but recommended for training)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:

```bash
git clone https://github.com/YOUR_USERNAME/metapathpredict.git
cd metapathpredict
```

3. Add the upstream repository:

```bash
git remote add upstream https://github.com/ORIGINAL_OWNER/metapathpredict.git
```

## Development Setup

### Using Conda (Recommended)

```bash
# Create environment
conda env create -f envs/environment.yml
conda activate metapathpredict

# Install package in development mode
pip install -e ".[dev]"
```

### Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package in development mode
pip install -e ".[dev]"
```

### Install Pre-commit Hooks

```bash
pre-commit install
```

This will run code quality checks automatically before each commit.

## Making Changes

### Branching Strategy

1. Create a new branch for your feature or bugfix:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bugfix-name
```

2. Keep your branch up to date with upstream:

```bash
git fetch upstream
git rebase upstream/main
```

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(model): add configurable kernel sizes for CNN

fix(preprocessing): handle edge case in sequence encoding

docs(readme): add installation instructions for GPU support

test(training): add integration tests for contrastive learning
```

## Coding Standards

### Code Style

- Follow [PEP 8](https://pep8.org/) for Python code
- Use [ruff](https://github.com/astral-sh/ruff) for linting and formatting
- Maximum line length: 100 characters
- Use type hints for all function signatures

### Import Order

```python
# Standard library imports
import os
import sys
from pathlib import Path

# Third-party imports
import torch
import numpy as np

# Local imports
from metapathpredict.models import ConfigurableCNN
from metapathpredict.training import Trainer
```

### Docstrings

Use Google-style docstrings:

```python
def process_sequence(sequence: str, max_length: int = 500) -> torch.Tensor:
    """Process a DNA sequence for model input.
    
    Encodes the input sequence using one-hot encoding and pads/truncates
    to the specified maximum length.
    
    Args:
        sequence: Input DNA sequence string (A, T, G, C, N).
        max_length: Maximum sequence length for padding/truncation.
    
    Returns:
        One-hot encoded tensor of shape (4, max_length).
    
    Raises:
        ValueError: If sequence contains invalid characters.
    
    Example:
        >>> tensor = process_sequence("ATGC", max_length=10)
        >>> tensor.shape
        torch.Size([4, 10])
    """
    ...
```

### Type Hints

```python
from typing import Dict, List, Optional, Tuple, Union

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    callbacks: Optional[List[Callback]] = None,
) -> Dict[str, List[float]]:
    ...
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=metapathpredict --cov-report=html

# Run specific test file
pytest tests/test_models.py

# Run specific test
pytest tests/test_models.py::TestConfigurableCNN::test_forward_pass

# Run integration tests only
pytest -m integration

# Run fast tests only (no GPU)
pytest -m "not slow"
```

### Writing Tests

- Place tests in the `tests/` directory
- Mirror the source code structure
- Use pytest fixtures for reusable test components
- Mark slow tests with `@pytest.mark.slow`
- Mark GPU tests with `@pytest.mark.gpu`

```python
import pytest
import torch

from metapathpredict.models import ConfigurableCNN


class TestConfigurableCNN:
    """Tests for ConfigurableCNN model."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return ConfigurableCNN(
            input_channels=4,
            num_classes=3,
            kernel_preset="small",
        )
    
    def test_forward_pass(self, model):
        """Test forward pass produces correct output shape."""
        batch = torch.randn(8, 4, 500)
        output = model(batch)
        
        assert output.shape == (8, 3)
    
    @pytest.mark.gpu
    def test_cuda_forward(self, model):
        """Test forward pass on GPU."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        model = model.cuda()
        batch = torch.randn(8, 4, 500).cuda()
        output = model(batch)
        
        assert output.device.type == "cuda"
```

### Test Coverage

- Aim for >80% code coverage for new code
- Focus on testing critical paths and edge cases
- Include both unit tests and integration tests

## Pull Request Process

### Before Submitting

1. Ensure all tests pass:
   ```bash
   pytest
   ```

2. Run linting and formatting:
   ```bash
   ruff check --fix .
   ruff format .
   mypy src/
   ```

3. Update documentation if needed

4. Update CHANGELOG.md with your changes

### Submitting a PR

1. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Create a Pull Request on GitHub

3. Fill in the PR template with:
   - Description of changes
   - Related issues
   - Testing performed
   - Screenshots (if applicable)

4. Request review from maintainers

### PR Review Process

- Maintainers will review your PR within a few days
- Address any feedback or requested changes
- Once approved, maintainers will merge your PR

### After Merge

- Delete your feature branch
- Pull the latest changes from upstream:
  ```bash
  git checkout main
  git pull upstream main
  ```

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

1. **Environment details:**
   - OS and version
   - Python version
   - PyTorch version
   - CUDA version (if applicable)

2. **Steps to reproduce:**
   - Minimal code example
   - Input data (if applicable)
   - Expected behavior
   - Actual behavior

3. **Error messages:**
   - Full traceback
   - Any relevant logs

### Feature Requests

When requesting features, please include:

1. **Use case:** Why is this feature needed?
2. **Proposed solution:** How should it work?
3. **Alternatives:** What alternatives have you considered?

## Project Structure

```
metapathpredict/
├── src/metapathpredict/      # Source code
│   ├── config/               # Configuration management
│   ├── data/                 # Data loading and preprocessing
│   ├── models/               # Neural network models
│   │   ├── cnn.py           # Configurable CNN
│   │   ├── contrastive.py   # Contrastive learning
│   │   └── reinforcement.py # RL agents
│   └── training/            # Training utilities
├── tests/                   # Test files
├── data/                    # Data directory
├── configs/                 # Configuration files
└── terraform/               # Infrastructure as code
```

## Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [pytest Documentation](https://docs.pytest.org/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Conventional Commits](https://www.conventionalcommits.org/)

## Questions?

If you have questions, feel free to:

1. Open an issue with the "question" label
2. Start a discussion on GitHub Discussions
3. Reach out to maintainers

Thank you for contributing to MetaPathPredict! 🎉
