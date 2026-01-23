# Changelog

All notable changes to MetaPathPredict will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- MkDocs documentation with Material theme
- Example Jupyter notebooks for quick start and configuration
- CONTRIBUTING.md guidelines

## [2.0.0] - 2024-XX-XX

### Added
- **PyTorch 2.0 Migration**: Complete rewrite from TensorFlow to PyTorch 2.0+
- **Three Training Approaches**:
  - Configurable CNN with kernel sizes 5, 7, 10
  - Contrastive Learning (SimCLR, SupCon)
  - Deep Reinforcement Learning (DQN, REINFORCE, A2C)
- **Pydantic v2 Configuration**: Type-safe configuration with validation
- **Experiment Tracking**: MLflow, Weights & Biases, and DuckDB support
- **Modern Data Pipeline**: 
  - DuckDB for local data processing
  - MinIO for object storage
  - Ray for distributed processing
  - Dagster for orchestration
- **Infrastructure as Code**: 
  - Terraform modules for GCP deployment
  - Docker and Docker Compose configurations
  - Kubernetes manifests for GKE
- **CI/CD Pipeline**:
  - GitHub Actions workflows
  - Pre-commit hooks (ruff, mypy)
  - Automated testing
- **Mixed Precision Training**: AMP support for faster training
- **torch.compile()**: JIT compilation for optimized inference
- **Ensemble and TTA**: Improved prediction accuracy

### Changed
- Migrated from TensorFlow/Keras to PyTorch 2.0+
- Configuration from JSON/YAML to Pydantic v2 models
- Project structure to src-layout
- Package management to Poetry/pyproject.toml

### Removed
- TensorFlow dependencies
- Legacy configuration format
- Deprecated training scripts

## [1.0.0] - 2023-XX-XX

### Added
- Initial release with TensorFlow implementation
- CNN model for sequence classification
- Basic training pipeline
- FASTA input support

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 2.0.0 | 2024-XX | PyTorch 2.0+, Three training approaches, Modern infrastructure |
| 1.0.0 | 2023-XX | Initial TensorFlow release |
