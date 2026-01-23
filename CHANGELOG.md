# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.1.0] - 2026-01-23

### Added
- **Three Training Approaches**:
  - Configurable CNN with kernel presets (small=5, medium=7, large=10, multi-scale)
  - Contrastive Learning (SimCLR, SupCon) with DNA-specific augmentations
  - Deep Reinforcement Learning (DQN, Policy Gradient, Actor-Critic)
- **Data Engineering Stack**:
  - DuckDB connector for OLAP analytics
  - DuckLake integration for lake format with ACID transactions
  - MinIO/S3 storage for models and datasets
  - Ray distributed training and hyperparameter tuning
  - Dagster pipeline orchestration with software-defined assets
- Docker Compose deployment for full DE stack
- Prometheus/Grafana monitoring integration
- DataHub governance integration (optional)
- Unit tests for core modules
- GitHub Actions CI/CD pipeline
- Pre-commit hooks configuration

### Changed
- Updated project structure for better modularity
- Improved documentation (README)

## [2.0.0] - 2026-01-23

### Added
- Complete rewrite to PyTorch 2.0
- Pydantic v2 configuration with validation
- Modern CNN architectures:
  - Multi-scale CNN
  - Residual CNN
  - Attention CNN
- Advanced data augmentation:
  - MixUp
  - CutMix
  - Reverse Complement
- Mixed Precision Training (AMP)
- Test-Time Augmentation (TTA)
- Ensemble predictions with K-Fold
- Modern learning rate schedulers:
  - Warmup Cosine Annealing
  - OneCycleLR
- Comprehensive callbacks:
  - EarlyStopping
  - ModelCheckpoint
  - MetricsLogger
- CLI interface with Click/Typer
- Full type hints coverage

### Changed
- Migrated from TensorFlow/Keras to PyTorch
- Standardized spelling to "eukaryotic" (was "eucaryotic")
- Unified preprocessing into single module
- Consolidated duplicate models

### Removed
- TensorFlow/Keras dependencies
- Duplicate preprocess files
- Unused Keras models (model_5.py, model_7.py, model_10.py)

### Fixed
- Hardcoded absolute paths
- Memory leaks in visualization
- Division by zero in metrics
- Race conditions in logging

## [1.0.0] - 2025-05-22

### Added
- Initial release
- CNN models for sequence classification
- Three-class classification (virus, bacteria, eukaryotic)
- FASTA file processing
- HDF5 dataset storage
- Basic training pipeline
- Prediction scripts

---

[Unreleased]: https://github.com/yourusername/metapathpredict/compare/v2.1.0...HEAD
[2.1.0]: https://github.com/yourusername/metapathpredict/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/yourusername/metapathpredict/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/yourusername/metapathpredict/releases/tag/v1.0.0
