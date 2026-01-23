"""
MetaPathPredict - Deep Learning for Metagenomic Sequence Classification

A PyTorch-based framework for classifying metagenomic sequences into
virus, bacteria, and eukaryotic categories.

Features:
- Modern CNN architectures with attention mechanisms
- Mixed precision training support
- Test-time augmentation
- Ensemble predictions
- CLI interface for training and inference
"""

__version__ = "2.0.0"
__author__ = "MetaPathPredict Team"

from metapathpredict.config import Settings
from metapathpredict.data import (
    HDF5SequenceDataset,
    SequenceDataModule,
    SequenceDataset,
    SequencePreprocessor,
)
from metapathpredict.inference import BatchPredictor, EnsemblePredictor, Predictor
from metapathpredict.models import (
    MultiScaleCNN,
    ResidualCNN,
    SimpleCNN,
    UnifiedClassifier,
    create_cnn_model,
)
from metapathpredict.training import (
    EarlyStopping,
    ModelCheckpoint,
    Trainer,
    create_trainer,
    get_scheduler,
)

__all__ = [
    # Config
    "Settings",
    # Models
    "UnifiedClassifier",
    "MultiScaleCNN",
    "ResidualCNN",
    "SimpleCNN",
    "create_cnn_model",
    # Data
    "SequenceDataset",
    "HDF5SequenceDataset",
    "SequenceDataModule",
    "SequencePreprocessor",
    # Training
    "Trainer",
    "create_trainer",
    "EarlyStopping",
    "ModelCheckpoint",
    "get_scheduler",
    # Inference
    "Predictor",
    "BatchPredictor",
    "EnsemblePredictor",
]
