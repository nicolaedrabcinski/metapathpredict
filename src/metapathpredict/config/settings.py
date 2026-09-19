"""
Pydantic configuration settings for MetaPathPredict.

This module provides type-safe, validated configuration management
using Pydantic v2 with support for YAML/JSON config files and environment variables.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class DeviceType(str, Enum):
    """Supported device types for computation."""
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon


class ModelType(str, Enum):
    """Available model architectures."""
    SIMPLE_CNN = "simple_cnn"
    MULTI_SCALE_CNN = "multi_scale_cnn"
    RESIDUAL_CNN = "residual_cnn"
    ATTENTION_CNN = "attention_cnn"
    TRANSFORMER = "transformer"


class SchedulerType(str, Enum):
    """Learning rate scheduler types."""
    COSINE = "cosine"
    COSINE_WARMUP = "cosine_warmup"
    STEP = "step"
    EXPONENTIAL = "exponential"
    PLATEAU = "plateau"
    ONE_CYCLE = "one_cycle"


class OptimizerType(str, Enum):
    """Optimizer types."""
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RADAM = "radam"
    LION = "lion"


class ClassLabel(str, Enum):
    """Sequence class labels - alphabetical order for consistency."""
    BACTERIA = "bacteria"
    EUKARYOTIC = "eukaryotic"
    VIRUS = "virus"


# Constants
NUCLEOTIDES = ["A", "C", "G", "T"]
NUM_NUCLEOTIDES = 4
DEFAULT_FRAGMENT_SIZES = [500, 1000]

# Class names in consistent alphabetical order - SINGLE SOURCE OF TRUTH
CLASS_NAMES = [ClassLabel.BACTERIA.value, ClassLabel.EUKARYOTIC.value, ClassLabel.VIRUS.value]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}
NUM_CLASSES = 3

# Fine-grained taxonomic classes used when a dataset is prepared from a genome
# manifest (scripts/download_diverse_genomes.py), and how they roll up into the
# three coarse classes that other tools (DeepMicroClass, Tiara) and the API report.
TAXON_CLASSES = [
    "bacteria", "archaea", "fungi", "protozoa", "plant", "invertebrate", "vertebrate", "virus",
]
NCBI_GROUP_TO_TAXON = {
    "bacteria": "bacteria",
    "archaea": "archaea",
    "fungi": "fungi",
    "protozoa": "protozoa",
    "plant": "plant",
    "invertebrate": "invertebrate",
    "vertebrate_other": "vertebrate",
    "vertebrate_mammalian": "vertebrate",
    "viral": "virus",
}
SUPERCLASSES = ["prokaryote", "eukaryote", "virus"]
TAXON_TO_SUPERCLASS = {
    "bacteria": "prokaryote", "archaea": "prokaryote",
    "fungi": "eukaryote", "protozoa": "eukaryote", "plant": "eukaryote",
    "invertebrate": "eukaryote", "vertebrate": "eukaryote",
    "virus": "virus",
    # legacy 3-class datasets
    "eukaryotic": "eukaryote",
}


def superclass_index_map(class_names: list[str]) -> list[int] | None:
    """Map each class index to its prokaryote/eukaryote/virus index, or None if a name has no mapping."""
    if not all(name in TAXON_TO_SUPERCLASS for name in class_names):
        return None
    return [SUPERCLASSES.index(TAXON_TO_SUPERCLASS[name]) for name in class_names]


class PathConfig(BaseModel):
    """Configuration for file and directory paths."""
    
    # Base directories
    base_dir: Path = Field(default=Path("."), description="Base directory for the project")
    data_dir: Path = Field(default=Path("data"), description="Data directory")
    
    # Input paths
    virus_fasta: Path = Field(default=Path("data/input/viruses.fasta"))
    bacteria_fasta: Path = Field(default=Path("data/input/bacteria.fasta"))
    eukaryotic_fasta: Path = Field(default=Path("data/input/eukaryotic.fasta"))
    
    # Output paths
    datasets_dir: Path = Field(default=Path("data/datasets/unified"))
    weights_dir: Path = Field(default=Path("data/weights/unified"))
    predictions_dir: Path = Field(default=Path("data/output/predictions"))
    logs_dir: Path = Field(default=Path("logs"))
    
    @field_validator("*", mode="before")
    @classmethod
    def expand_path(cls, v: Any) -> Any:
        """Expand user home directory and resolve relative paths."""
        if isinstance(v, (str, Path)):
            return Path(v).expanduser()
        return v
    
    def make_absolute(self, base: Path | None = None) -> "PathConfig":
        """Convert all paths to absolute paths relative to base."""
        base = (base or self.base_dir).resolve()
        data = {}
        for field_name, field_value in self:
            if isinstance(field_value, Path) and not field_value.is_absolute():
                data[field_name] = base / field_value
            else:
                data[field_name] = field_value
        return PathConfig(**data)
    
    def ensure_directories(self) -> None:
        """Create all output directories if they don't exist."""
        for path in [self.datasets_dir, self.weights_dir, self.predictions_dir, self.logs_dir]:
            path.mkdir(parents=True, exist_ok=True)


class DataConfig(BaseModel):
    """Configuration for data processing."""
    
    # Fragment settings
    fragment_sizes: list[int] = Field(
        default=[500, 1000],
        description="Fragment sizes in base pairs"
    )
    default_fragment_size: int = Field(default=1000)
    
    # Sampling settings
    fragments_per_class: int = Field(
        default=20000,
        ge=100,
        description="Number of fragments per class"
    )
    min_sequence_length: int = Field(default=100, ge=50)
    max_n_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    # BIO-003: Lowered from 0.8 to 0.7 for better compatibility with real metagenomic data
    min_valid_nt_ratio: float = Field(
        default=0.7, 
        ge=0.0, 
        le=1.0,
        description="Minimum ratio of valid nucleotides (ACGT). Metagenomic data often has 10-30% N."
    )
    
    # Fragmentation
    fragment_step_ratio: float = Field(
        default=0.5,
        ge=0.1,
        le=1.0,
        description="Step size as ratio of fragment size (0.5 = 50% overlap)"
    )
    
    # Augmentation
    use_reverse_complement: bool = Field(default=True)
    augmentation_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Split settings
    train_ratio: float = Field(default=0.8, ge=0.5, le=0.95)
    val_ratio: float = Field(default=0.1, ge=0.05, le=0.3)
    test_ratio: float = Field(default=0.1, ge=0.05, le=0.3)
    
    # Class labels - using centralized alphabetical order from CLASS_NAMES
    class_names: list[str] = Field(default_factory=lambda: CLASS_NAMES.copy())
    class_to_idx: dict[str, int] = Field(default_factory=lambda: CLASS_TO_IDX.copy())
    
    # Processing
    num_workers: int = Field(default=4, ge=0)
    
    @model_validator(mode="after")
    def validate_split_ratios(self) -> "DataConfig":
        """Ensure split ratios sum to 1.0."""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        return self


class ModelConfig(BaseModel):
    """Configuration for model architecture."""
    
    # Disable protected namespace warning for model_type field
    model_config = {"protected_namespaces": ()}
    
    # Architecture
    model_type: ModelType = Field(default=ModelType.MULTI_SCALE_CNN)
    num_classes: int = Field(default=3, ge=2)
    
    # CNN settings
    kernel_sizes: list[int] = Field(
        default=[5, 7, 11],
        description="Kernel sizes for multi-scale CNN"
    )
    num_filters: list[int] = Field(
        default=[64, 128, 256],
        description="Number of filters for each conv layer"
    )
    
    # Network architecture
    hidden_dims: list[int] = Field(
        default=[512, 256],
        description="Hidden layer dimensions for classifier"
    )
    
    # Regularization
    dropout_rate: float = Field(default=0.3, ge=0.0, le=0.8)
    use_batch_norm: bool = Field(default=True)
    use_layer_norm: bool = Field(default=False)
    
    # Attention (if applicable)
    num_attention_heads: int = Field(default=4, ge=1)
    attention_dropout: float = Field(default=0.1, ge=0.0, le=0.5)
    
    # Residual connections
    use_residual: bool = Field(default=True)
    
    # Weight initialization
    init_method: Literal["xavier", "kaiming", "orthogonal"] = Field(default="kaiming")


class TrainingConfig(BaseModel):
    """Configuration for model training."""
    
    # Basic training params
    num_epochs: int = Field(default=50, ge=1)
    batch_size: int = Field(default=64, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    
    # Optimizer
    optimizer: OptimizerType = Field(default=OptimizerType.ADAMW)
    learning_rate: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    
    # AdamW specific
    adam_beta1: float = Field(default=0.9)
    adam_beta2: float = Field(default=0.999)
    adam_eps: float = Field(default=1e-8)
    
    # SGD specific
    sgd_momentum: float = Field(default=0.9)
    sgd_nesterov: bool = Field(default=True)
    
    # Learning rate scheduler
    scheduler: SchedulerType = Field(default=SchedulerType.COSINE_WARMUP)
    warmup_epochs: int = Field(default=5, ge=0)
    min_lr: float = Field(default=1e-6, ge=0)
    
    # Early stopping
    early_stopping: bool = Field(default=True)
    patience: int = Field(default=10, ge=1)
    min_delta: float = Field(default=1e-4, ge=0)
    
    # Gradient clipping
    gradient_clip_val: float | None = Field(default=1.0, ge=0)
    gradient_clip_algorithm: Literal["norm", "value"] = Field(default="norm")
    
    # Mixed precision
    use_amp: bool = Field(default=True, description="Use automatic mixed precision")
    amp_dtype: Literal["float16", "bfloat16"] = Field(default="float16")
    
    # Checkpointing
    save_top_k: int = Field(default=3, ge=1)
    checkpoint_metric: str = Field(default="val_loss")
    checkpoint_mode: Literal["min", "max"] = Field(default="min")
    
    # Logging
    log_every_n_steps: int = Field(default=10, ge=1)
    val_check_interval: float = Field(default=1.0, gt=0)
    
    # Reproducibility
    seed: int = Field(default=42)
    deterministic: bool = Field(default=True)
    
    # Label smoothing
    label_smoothing: float = Field(default=0.1, ge=0.0, le=0.5)
    
    # Class weights for imbalanced data
    use_class_weights: bool = Field(default=False)
    class_weights: list[float] | None = Field(default=None)


class ContrastiveConfig(BaseModel):
    """Configuration for contrastive learning phase."""

    # Encoder
    backbone: Literal["small", "medium", "large", "progressive", "multi"] = Field(
        default="medium", description="CNN backbone preset for ContrastiveEncoder"
    )
    base_channels: int = Field(default=64, ge=16)
    projection_dim: int = Field(default=128, ge=32)
    hidden_dim: int = Field(default=256, ge=64)

    # Loss
    loss_type: Literal["ntxent", "supcon"] = Field(
        default="supcon", description="ntxent = SimCLR unsupervised, supcon = supervised contrastive"
    )
    temperature: float = Field(default=0.07, gt=0, le=1.0)

    # Augmentation
    mutation_rate: float = Field(default=0.1, ge=0.0, le=0.5)
    mask_rate: float = Field(default=0.15, ge=0.0, le=0.5)

    # Training
    num_epochs: int = Field(default=20, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    batch_size: int = Field(default=128, ge=1)
    early_stopping_patience: int = Field(
        default=3, ge=0,
        description="Stop if val loss doesn't improve for this many epochs. 0 disables it.",
    )
    probe_epochs: int = Field(
        default=3, ge=1,
        description="Epochs for the linear-probe classifier head fit after contrastive pretraining.",
    )


class RLConfig(BaseModel):
    """Configuration for reinforcement learning phase."""

    # Agent
    algorithm: Literal["dqn", "policy_gradient", "actor_critic"] = Field(
        default="actor_critic"
    )
    backbone: Literal["small", "medium", "large", "progressive", "multi"] = Field(
        default="medium"
    )
    hidden_dim: int = Field(default=256, ge=64)

    # Environment rewards
    reward_correct: float = Field(default=1.0)
    reward_incorrect: float = Field(default=-0.5)
    reward_uncertain: float = Field(default=-0.1)

    # Training
    num_epochs: int = Field(default=10, ge=1)
    episodes_per_epoch: int = Field(default=1000, ge=100)
    batch_size: int = Field(
        default=32, ge=1,
        description="Episodes per gradient update for actor_critic (reduces update variance vs. 1)",
    )
    learning_rate: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    gamma: float = Field(default=0.99, ge=0, le=1.0)
    early_stopping_patience: int = Field(
        default=3, ge=0,
        description="Stop if val accuracy doesn't improve for this many epochs. 0 disables it.",
    )

    # DQN-specific
    epsilon_start: float = Field(default=1.0, ge=0, le=1.0)
    epsilon_end: float = Field(default=0.05, ge=0, le=1.0)
    epsilon_decay_epochs: int = Field(default=5, ge=1)
    replay_buffer_size: int = Field(default=10000, ge=1000)
    target_update_freq: int = Field(default=5, ge=1)

    # Transfer learning
    load_encoder_from: str | None = Field(
        default=None, description="Path to contrastive checkpoint to initialize encoder weights"
    )


class InferenceConfig(BaseModel):
    """Configuration for model inference."""
    
    batch_size: int = Field(default=128, ge=1)
    num_workers: int = Field(default=4, ge=0)
    
    # Prediction settings
    min_contig_length: int = Field(default=750, ge=100)
    overlap_ratio: float = Field(default=0.5, ge=0.0, le=0.9)
    
    # Thresholds
    high_confidence_threshold: float = Field(default=0.8, ge=0.5, le=1.0)
    medium_confidence_threshold: float = Field(default=0.6, ge=0.3, le=0.9)
    
    # Aggregation method for fragments
    aggregation_method: Literal["mean", "max", "voting"] = Field(default="mean")
    
    # Output
    save_fragment_predictions: bool = Field(default=True)
    save_probabilities: bool = Field(default=True)
    output_format: Literal["csv", "tsv", "json"] = Field(default="csv")
    
    # Ensemble
    use_ensemble: bool = Field(default=False)
    ensemble_weights: list[float] | None = Field(default=None)
    
    # Test Time Augmentation
    use_tta: bool = Field(default=False)
    tta_transforms: list[str] = Field(default=["reverse_complement"])


class Settings(BaseModel):
    """Main settings class combining all configuration sections."""
    
    # Sub-configurations
    paths: PathConfig = Field(default_factory=PathConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    contrastive: ContrastiveConfig = Field(default_factory=ContrastiveConfig)
    rl: RLConfig = Field(default_factory=RLConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    
    # Runtime settings
    device: DeviceType = Field(default=DeviceType.AUTO)
    verbose: bool = Field(default=True)
    debug: bool = Field(default=False)
    
    # Project info
    project_name: str = Field(default="metapathpredict")
    experiment_name: str = Field(default="default")
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Settings":
        """Load settings from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        return cls(**data)
    
    @classmethod
    def from_json(cls, path: str | Path) -> "Settings":
        """Load settings from a JSON file."""
        import json
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, "r") as f:
            data = json.load(f)
        
        return cls(**data)
    
    def to_yaml(self, path: str | Path) -> None:
        """Save settings to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            yaml.dump(self.model_dump(mode="json"), f, default_flow_style=False, sort_keys=False)
    
    def to_json(self, path: str | Path, indent: int = 2) -> None:
        """Save settings to a JSON file."""
        import json
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(self.model_dump(mode="json"), f, indent=indent)
    
    def get_device(self) -> str:
        """Get the actual device string based on availability."""
        import torch
        
        if self.device == DeviceType.AUTO:
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return self.device.value
    
    def setup_reproducibility(self) -> None:
        """Set up random seeds and deterministic behavior."""
        import random
        import numpy as np
        import torch
        
        seed = self.training.seed
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        if self.training.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # For PyTorch 2.0+
            if hasattr(torch, "use_deterministic_algorithms"):
                try:
                    torch.use_deterministic_algorithms(True)
                except RuntimeError:
                    pass  # Some operations don't support deterministic mode
    
    def model_post_init(self, __context: Any) -> None:
        """Post-initialization setup."""
        # Make paths absolute
        self.paths = self.paths.make_absolute()


def load_config(path: str | Path | None = None) -> Settings:
    """
    Load configuration from file or return defaults.
    
    Args:
        path: Path to config file (YAML or JSON). If None, returns defaults.
    
    Returns:
        Settings object with configuration.
    """
    if path is None:
        return Settings()
    
    path = Path(path)
    
    if path.suffix in {".yaml", ".yml"}:
        return Settings.from_yaml(path)
    elif path.suffix == ".json":
        return Settings.from_json(path)
    else:
        raise ValueError(f"Unsupported config file format: {path.suffix}")


def create_default_config(path: str | Path = "config.yaml") -> Settings:
    """Create and save a default configuration file."""
    settings = Settings()
    settings.to_yaml(path)
    return settings
