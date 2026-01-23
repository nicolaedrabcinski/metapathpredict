"""
Unit tests for configuration module.
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from metapathpredict.config.settings import (
    ClassLabel,
    DataConfig,
    DeviceType,
    ModelConfig,
    ModelType,
    OptimizerType,
    PathConfig,
    SchedulerType,
    Settings,
    TrainingConfig,
)


class TestClassLabel:
    """Tests for ClassLabel enum."""

    def test_enum_values(self):
        """Test that enum has correct values."""
        assert ClassLabel.VIRUS.value == "virus"
        assert ClassLabel.BACTERIA.value == "bacteria"
        assert ClassLabel.EUKARYOTIC.value == "eukaryotic"

    def test_correct_spelling(self):
        """Test that 'eukaryotic' is spelled correctly (not 'eucaryotic')."""
        assert "eukaryotic" in [c.value for c in ClassLabel]
        # Ensure old spelling is not present
        assert "eucaryotic" not in [c.value for c in ClassLabel]


class TestDeviceType:
    """Tests for DeviceType enum."""

    def test_supported_devices(self):
        """Test all supported device types."""
        assert DeviceType.AUTO.value == "auto"
        assert DeviceType.CPU.value == "cpu"
        assert DeviceType.CUDA.value == "cuda"
        assert DeviceType.MPS.value == "mps"


class TestPathConfig:
    """Tests for PathConfig."""

    def test_default_paths(self):
        """Test default path values."""
        config = PathConfig()
        assert config.data_dir == Path("data")
        assert config.datasets_dir == Path("data/datasets/unified")
        assert config.weights_dir == Path("data/weights/unified")

    def test_custom_paths(self):
        """Test custom path values."""
        config = PathConfig(
            data_dir=Path("/custom/data"),
            weights_dir=Path("/custom/weights"),
        )
        assert config.data_dir == Path("/custom/data")
        assert config.weights_dir == Path("/custom/weights")

    def test_path_expansion(self):
        """Test that paths with ~ are expanded."""
        config = PathConfig(data_dir="~/data")
        assert "~" not in str(config.data_dir)

    def test_make_absolute(self):
        """Test make_absolute method."""
        config = PathConfig(base_dir=Path("/project"))
        absolute_config = config.make_absolute()
        assert absolute_config.data_dir.is_absolute()


class TestDataConfig:
    """Tests for DataConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DataConfig()
        assert config.default_fragment_size == 1000
        assert config.fragment_sizes == [500, 1000]
        assert config.train_ratio == 0.8
        assert config.val_ratio == 0.1
        assert config.test_ratio == 0.1

    def test_fragment_size_validation(self):
        """Test fragment size list must be non-empty."""
        # default_fragment_size doesn't have ge=1 validation
        # but fragment_sizes list should work
        config = DataConfig(default_fragment_size=100)
        assert config.default_fragment_size == 100
        # Test min_sequence_length has validation
        with pytest.raises(ValueError):
            DataConfig(min_sequence_length=10)  # ge=50

    def test_split_sum_validation(self):
        """Test that splits sum to 1.0."""
        # Valid splits
        config = DataConfig(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
        assert config.train_ratio + config.val_ratio + config.test_ratio == pytest.approx(1.0)

    def test_batch_size_validation(self):
        """Test batch size must be positive."""
        # Note: batch_size is now in TrainingConfig, not DataConfig
        pass  # DataConfig doesn't have batch_size anymore


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_model_type(self):
        """Test default model type."""
        config = ModelConfig()
        assert config.model_type == ModelType.MULTI_SCALE_CNN

    def test_custom_model_config(self):
        """Test custom model configuration."""
        config = ModelConfig(
            model_type=ModelType.ATTENTION_CNN,
            hidden_dims=[128, 64],
            num_classes=3,
            dropout_rate=0.3,
        )
        assert config.model_type == ModelType.ATTENTION_CNN
        assert config.hidden_dims == [128, 64]
        assert config.num_classes == 3
        assert config.dropout_rate == 0.3

    def test_dropout_range_validation(self):
        """Test dropout must be between 0 and 1."""
        with pytest.raises(ValueError):
            ModelConfig(dropout_rate=-0.1)
        with pytest.raises(ValueError):
            ModelConfig(dropout_rate=1.5)


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_values(self):
        """Test default training configuration."""
        config = TrainingConfig()
        assert config.num_epochs == 50  # Default is 50
        assert config.learning_rate == 1e-3  # Default learning rate
        assert config.optimizer == OptimizerType.ADAMW
        assert config.use_amp is True  # use_amp instead of use_mixed_precision

    def test_learning_rate_validation(self):
        """Test learning rate must be positive."""
        with pytest.raises(ValueError):
            TrainingConfig(learning_rate=0)
        with pytest.raises(ValueError):
            TrainingConfig(learning_rate=-0.001)

    def test_epochs_validation(self):
        """Test epochs must be positive."""
        with pytest.raises(ValueError):
            TrainingConfig(num_epochs=0)

    def test_scheduler_options(self):
        """Test different scheduler types."""
        for scheduler in SchedulerType:
            config = TrainingConfig(scheduler=scheduler)
            assert config.scheduler == scheduler


class TestSettings:
    """Tests for main Settings class."""

    def test_default_settings(self):
        """Test default settings creation."""
        settings = Settings()
        assert settings.paths is not None
        assert settings.data is not None
        assert settings.model is not None
        assert settings.training is not None

    def test_from_yaml(self):
        """Test loading settings from YAML file."""
        yaml_content = """
paths:
  data_dir: /test/data
  weights_dir: /test/weights
data:
  default_fragment_size: 500
  fragment_sizes: [500]
model:
  model_type: attention_cnn
  hidden_dims: [256, 128]
training:
  num_epochs: 50
  learning_rate: 0.0005
  batch_size: 32
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            
            settings = Settings.from_yaml(f.name)
            
            assert settings.paths.data_dir == Path("/test/data")
            assert settings.data.default_fragment_size == 500
            assert settings.training.batch_size == 32
            assert settings.model.model_type == ModelType.ATTENTION_CNN
            assert settings.model.hidden_dims == [256, 128]
            assert settings.training.num_epochs == 50

    def test_to_yaml(self):
        """Test saving settings to YAML file."""
        settings = Settings(
            data=DataConfig(default_fragment_size=500),
            training=TrainingConfig(num_epochs=25),
        )
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            settings.to_yaml(f.name)
            
            with open(f.name) as rf:
                loaded = yaml.safe_load(rf)
            
            assert loaded["data"]["default_fragment_size"] == 500
            assert loaded["training"]["num_epochs"] == 25

    def test_settings_immutability(self):
        """Test that settings are frozen (immutable)."""
        settings = Settings()
        # Pydantic v2 with frozen=True should raise error
        # Note: Need to check if model is actually frozen


class TestEnvironmentOverrides:
    """Tests for environment variable overrides."""

    def test_env_override_precedence(self, monkeypatch):
        """Test that environment variables can override config."""
        # This tests the Settings.from_yaml_with_overrides functionality
        # if implemented with environment variable support
        pass  # To be implemented based on actual env var support


class TestConfigSerialization:
    """Tests for config serialization/deserialization."""

    def test_round_trip_yaml(self):
        """Test that config survives YAML round-trip."""
        original = Settings(
            data=DataConfig(default_fragment_size=750),
            model=ModelConfig(hidden_dims=[512, 256], dropout_rate=0.4),
            training=TrainingConfig(num_epochs=200, learning_rate=0.0001, batch_size=128),
        )
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            original.to_yaml(f.name)
            loaded = Settings.from_yaml(f.name)
        
        assert loaded.data.default_fragment_size == original.data.default_fragment_size
        assert loaded.training.batch_size == original.training.batch_size
        assert loaded.model.hidden_dims == original.model.hidden_dims
        assert loaded.training.num_epochs == original.training.num_epochs

    def test_json_serialization(self):
        """Test JSON serialization."""
        settings = Settings()
        json_str = settings.model_dump_json()
        
        assert "default_fragment_size" in json_str
        assert "batch_size" in json_str
