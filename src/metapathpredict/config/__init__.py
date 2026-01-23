"""Configuration module with Pydantic settings."""

from metapathpredict.config.settings import (
    Settings,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    InferenceConfig,
    PathConfig,
    # Class label constants - single source of truth
    CLASS_NAMES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    NUM_CLASSES,
    ClassLabel,
)

__all__ = [
    "Settings",
    "DataConfig",
    "ModelConfig", 
    "TrainingConfig",
    "InferenceConfig",
    "PathConfig",
    "CLASS_NAMES",
    "CLASS_TO_IDX",
    "IDX_TO_CLASS",
    "NUM_CLASSES",
    "ClassLabel",
]
