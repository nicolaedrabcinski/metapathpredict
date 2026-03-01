"""
MetaPathPredict - Deep Learning for Metagenomic Sequence Classification

A PyTorch-based framework for classifying metagenomic sequences into
virus, bacteria, and eukaryotic categories using Contrastive Learning
and Deep Reinforcement Learning.

Features:
- Contrastive pretraining (SimCLR / SupCon)
- Deep RL fine-tuning (DQN, REINFORCE, Actor-Critic)
- Full pipeline: contrastive pretrain -> RL fine-tune
- CLI interface for training and inference
- FastAPI dashboard for interpretability
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
from metapathpredict.models import (
    # Contrastive Learning (primary)
    ContrastiveAugmentation,
    ContrastiveEncoder,
    ContrastiveTrainer,
    NTXentLoss,
    SupConLoss,
    # Reinforcement Learning (primary)
    ActorCriticAgent,
    DQNAgent,
    PolicyGradientAgent,
    RLTrainer,
    SequenceEnvironment,
    ReplayBuffer,
    # CNN backbone
    ConfigurableCNN,
    create_configurable_cnn,
    # Legacy CNN models (kept for backward compatibility)
    MultiScaleCNN,
    ResidualCNN,
    SimpleCNN,
    UnifiedClassifier,
    create_cnn_model,
)

__all__ = [
    # Config
    "Settings",
    # Contrastive Learning
    "ContrastiveEncoder",
    "ContrastiveTrainer",
    "ContrastiveAugmentation",
    "NTXentLoss",
    "SupConLoss",
    # Reinforcement Learning
    "DQNAgent",
    "PolicyGradientAgent",
    "ActorCriticAgent",
    "RLTrainer",
    "SequenceEnvironment",
    "ReplayBuffer",
    # CNN backbone
    "ConfigurableCNN",
    "create_configurable_cnn",
    # Data
    "SequenceDataset",
    "HDF5SequenceDataset",
    "SequenceDataModule",
    "SequencePreprocessor",
    # Legacy
    "UnifiedClassifier",
    "MultiScaleCNN",
    "ResidualCNN",
    "SimpleCNN",
    "create_cnn_model",
]
