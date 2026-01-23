# Contributing to MetaPathPredict

Thank you for your interest in contributing! Please see our full [CONTRIBUTING.md](https://github.com/your-org/metapathpredict/blob/main/CONTRIBUTING.md) file for detailed guidelines.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/your-org/metapathpredict.git
cd metapathpredict

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install development dependencies
pip install -e ".[dev,test]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

## Development Workflow

1. **Fork** the repository
2. Create a **feature branch**: `git checkout -b feature/your-feature`
3. Make your changes
4. Run **tests**: `pytest`
5. Run **linting**: `ruff check .`
6. **Commit** with conventional commits: `git commit -m "feat: add new feature"`
7. **Push** and create a Pull Request

## Code Style

- Follow **PEP 8** guidelines
- Use **type hints** for all functions
- Write **docstrings** in Google style
- Format with **ruff format**
- Lint with **ruff check**

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=metapathpredict

# Run specific tests
pytest tests/test_models.py -v
```

## Documentation

```bash
# Build docs locally
mkdocs serve

# View at http://localhost:8000
```

## Questions?

- Open an [Issue](https://github.com/your-org/metapathpredict/issues)
- Start a [Discussion](https://github.com/your-org/metapathpredict/discussions)
