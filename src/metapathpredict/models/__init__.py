"""PyTorch models for sequence classification."""

from metapathpredict.models.attention import AttentionBlock, SelfAttention
from metapathpredict.models.base import BaseModel
from metapathpredict.models.cnn import (
    MultiScaleCNN,
    ResidualBlock,
    ResidualCNN,
    SimpleCNN,
    create_cnn_model,
)
from metapathpredict.models.configurable_cnn import (
    ConfigurableCNN,
    create_configurable_cnn,
)
from metapathpredict.models.contrastive import (
    ContrastiveAugmentation,
    ContrastiveEncoder,
    ContrastiveTrainer,
    NTXentLoss,
    ProjectionHead,
    SupConLoss,
)
from metapathpredict.models.reinforcement import (
    ActorCriticAgent,
    DQNAgent,
    PolicyGradientAgent,
    ReplayBuffer,
    RLTrainer,
    SequenceEnvironment,
)
from metapathpredict.models.unified import UnifiedClassifier

__all__ = [
    # Base
    "BaseModel",
    # CNN
    "UnifiedClassifier",
    "MultiScaleCNN",
    "ResidualCNN",
    "SimpleCNN",
    "ResidualBlock",
    "create_cnn_model",
    # Configurable CNN
    "ConfigurableCNN",
    "create_configurable_cnn",
    # Attention
    "AttentionBlock",
    "SelfAttention",
    # Contrastive Learning
    "ContrastiveEncoder",
    "ContrastiveTrainer",
    "ContrastiveAugmentation",
    "ProjectionHead",
    "NTXentLoss",
    "SupConLoss",
    # Reinforcement Learning
    "DQNAgent",
    "PolicyGradientAgent",
    "ActorCriticAgent",
    "RLTrainer",
    "SequenceEnvironment",
    "ReplayBuffer",
]
