"""
Unit tests for CNN models.
"""

import pytest
import torch
import torch.nn as nn

from metapathpredict.models.cnn import (
    MultiScaleCNN,
    ResidualBlock,
    ResidualCNN,
    SimpleCNN,
)
from metapathpredict.models.base import BaseModel, ConvBlock, SEBlock
from metapathpredict.models.configurable_cnn import (
    ConfigurableCNN,
    KERNEL_PRESETS,
    create_configurable_cnn,
)


class TestConvBlock:
    """Tests for ConvBlock."""

    def test_output_shape(self):
        """Test output shape matches expected."""
        block = ConvBlock(in_channels=4, out_channels=64, kernel_size=7)
        x = torch.randn(8, 4, 1000)  # batch, channels, seq_len
        
        out = block(x)
        
        assert out.shape[0] == 8  # batch preserved
        assert out.shape[1] == 64  # channels changed
        assert out.shape[2] == 1000  # seq_len preserved (same padding)

    def test_with_dropout(self):
        """Test output with dropout enabled."""
        block = ConvBlock(in_channels=4, out_channels=64, kernel_size=7, dropout=0.5)
        x = torch.randn(8, 4, 1000)
        
        block.train()
        out = block(x)
        
        assert out.shape == (8, 64, 1000)


class TestSEBlock:
    """Tests for Squeeze-and-Excitation block."""

    def test_output_shape(self):
        """Test SE block preserves shape."""
        se = SEBlock(channels=64, reduction=16)
        x = torch.randn(8, 64, 1000)
        
        out = se(x)
        
        assert out.shape == x.shape

    def test_attention_mechanism(self):
        """Test that SE applies channel-wise attention."""
        se = SEBlock(channels=64, reduction=16)
        x = torch.randn(8, 64, 1000)
        
        with torch.no_grad():
            out = se(x)
        
        # Output should be modified (not equal to input)
        assert not torch.allclose(out, x)


class TestResidualBlock:
    """Tests for ResidualBlock."""

    def test_residual_connection(self):
        """Test residual connection preserves information."""
        block = ResidualBlock(channels=64, kernel_size=3)
        x = torch.randn(8, 64, 1000)
        
        out = block(x)
        
        assert out.shape == x.shape

    def test_with_se(self):
        """Test residual block with SE."""
        block = ResidualBlock(channels=64, use_se=True)
        x = torch.randn(8, 64, 100)
        
        out = block(x)
        
        assert out.shape == x.shape

    def test_with_dilation(self):
        """Test dilated convolutions."""
        block = ResidualBlock(channels=64, dilation=2)
        x = torch.randn(8, 64, 100)
        
        out = block(x)
        
        assert out.shape == x.shape


class TestMultiScaleCNN:
    """Tests for MultiScaleCNN."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        model = MultiScaleCNN(
            seq_length=1000,
            num_classes=3,
            branch_channels=64,
            kernel_sizes=[5, 7, 10],
        )
        # Note: MultiScaleCNN expects input (B, L, 4) - length first, then channels
        x = torch.randn(8, 1000, 4)
        
        out = model(x)
        
        assert out.shape == (8, 3)

    def test_different_kernel_sizes(self):
        """Test with different kernel size configurations."""
        for kernels in [[5], [5, 7], [5, 7, 10], [3, 5, 7, 11]]:
            model = MultiScaleCNN(
                seq_length=500,
                num_classes=3,
                kernel_sizes=kernels,
            )
            x = torch.randn(4, 500, 4)
            out = model(x)
            
            assert out.shape == (4, 3)

    def test_variable_sequence_length(self):
        """Test model handles different sequence lengths."""
        for seq_len in [100, 500, 1000]:
            model = MultiScaleCNN(seq_length=seq_len, num_classes=3)
            x = torch.randn(4, seq_len, 4)
            out = model(x)
            
            assert out.shape == (4, 3)

    def test_gradient_flow(self):
        """Test gradients flow through the model."""
        model = MultiScaleCNN(seq_length=500, num_classes=3)
        x = torch.randn(4, 500, 4, requires_grad=True)
        
        out = model(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestResidualCNN:
    """Tests for ResidualCNN."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        model = ResidualCNN(
            seq_length=1000,
            num_classes=3,
            base_channels=64,
            num_blocks=[2, 2, 2],
        )
        x = torch.randn(8, 1000, 4)
        
        out = model(x)
        
        assert out.shape == (8, 3)

    def test_different_depths(self):
        """Test with different numbers of residual blocks."""
        for num_blocks in [[2], [2, 2], [2, 2, 2, 2]]:
            model = ResidualCNN(
                seq_length=500,
                num_classes=3,
                num_blocks=num_blocks,
            )
            x = torch.randn(4, 500, 4)
            out = model(x)
            
            assert out.shape == (4, 3)


class TestSimpleCNN:
    """Tests for SimpleCNN."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        model = SimpleCNN(seq_length=1000, num_classes=3)
        x = torch.randn(8, 1000, 4)
        
        out = model(x)
        
        assert out.shape == (8, 3)

    def test_custom_filters(self):
        """Test with custom filter sizes."""
        model = SimpleCNN(
            seq_length=500,
            num_classes=3,
            num_filters=[32, 64, 128],
        )
        x = torch.randn(4, 500, 4)
        out = model(x)
        
        assert out.shape == (4, 3)


class TestConfigurableCNN:
    """Tests for ConfigurableCNN with kernel presets."""

    def test_kernel_presets_exist(self):
        """Test that all kernel presets are defined."""
        assert "small" in KERNEL_PRESETS
        assert "medium" in KERNEL_PRESETS
        assert "large" in KERNEL_PRESETS
        assert "multi" in KERNEL_PRESETS
        
        # KERNEL_PRESETS are lists of kernel sizes
        assert KERNEL_PRESETS["small"] == [5, 5, 5]
        assert KERNEL_PRESETS["medium"] == [7, 7, 7]
        assert KERNEL_PRESETS["large"] == [10, 10, 10]

    def test_small_kernel(self):
        """Test CNN with small kernel (5)."""
        model = ConfigurableCNN(
            in_channels=4,
            num_classes=3,
            kernel_preset="small",
        )
        x = torch.randn(8, 4, 1000)
        
        out = model(x)
        
        assert out.shape == (8, 3)
        assert model.kernel_sizes == [5, 5, 5]

    def test_medium_kernel(self):
        """Test CNN with medium kernel (7)."""
        model = ConfigurableCNN(
            in_channels=4,
            num_classes=3,
            kernel_preset="medium",
        )
        x = torch.randn(8, 4, 1000)
        
        out = model(x)
        
        assert out.shape == (8, 3)
        assert model.kernel_sizes == [7, 7, 7]

    def test_large_kernel(self):
        """Test CNN with large kernel (10)."""
        model = ConfigurableCNN(
            in_channels=4,
            num_classes=3,
            kernel_preset="large",
        )
        x = torch.randn(8, 4, 1000)
        
        out = model(x)
        
        assert out.shape == (8, 3)
        assert model.kernel_sizes == [10, 10, 10]

    def test_multi_scale_kernel(self):
        """Test CNN with multi-scale kernels."""
        model = ConfigurableCNN(
            in_channels=4,
            num_classes=3,
            kernel_preset="multi",
        )
        x = torch.randn(8, 4, 1000)
        
        out = model(x)
        
        assert out.shape == (8, 3)

    def test_custom_kernel_size(self):
        """Test CNN with custom kernel sizes."""
        model = ConfigurableCNN(
            in_channels=4,
            num_classes=3,
            custom_kernels=[3, 5, 9],
        )
        x = torch.randn(8, 4, 1000)
        
        out = model(x)
        
        assert out.shape == (8, 3)
        assert model.kernel_sizes == [3, 5, 9]

    def test_invalid_preset_raises(self):
        """Test that invalid preset raises ValueError."""
        with pytest.raises((ValueError, KeyError)):
            ConfigurableCNN(
                in_channels=4,
                num_classes=3,
                kernel_preset="invalid",
            )

    def test_factory_function(self):
        """Test create_configurable_cnn factory."""
        # With preset name
        model = create_configurable_cnn(kernel_size="medium")
        assert model.kernel_sizes == [7, 7, 7]
        
        # With integer kernel size
        model = create_configurable_cnn(kernel_size=5)
        assert model.kernel_sizes == [5, 5, 5]


class TestModelTraining:
    """Tests for model training behavior."""

    def test_train_mode(self):
        """Test model in training mode."""
        model = ConfigurableCNN(in_channels=4, num_classes=3)
        model.train()
        
        x = torch.randn(8, 4, 1000)
        out = model(x)
        
        assert out.shape == (8, 3)

    def test_eval_mode(self):
        """Test model in evaluation mode."""
        model = ConfigurableCNN(in_channels=4, num_classes=3)
        model.eval()
        
        x = torch.randn(8, 4, 1000)
        with torch.no_grad():
            out = model(x)
        
        assert out.shape == (8, 3)

    def test_parameter_count(self):
        """Test model has reasonable parameter count."""
        model = ConfigurableCNN(in_channels=4, num_classes=3)
        
        num_params = sum(p.numel() for p in model.parameters())
        
        # Should have a reasonable number of parameters
        assert num_params > 1000  # At least some parameters
        assert num_params < 100_000_000  # Not too many


class TestModelSaveLoad:
    """Tests for model serialization."""

    def test_state_dict_save_load(self):
        """Test saving and loading model state dict."""
        model = ConfigurableCNN(in_channels=4, num_classes=3)
        x = torch.randn(4, 4, 500)
        
        # Get initial output
        model.eval()
        with torch.no_grad():
            out1 = model(x)
        
        # Save and load state dict
        state = model.state_dict()
        
        model2 = ConfigurableCNN(in_channels=4, num_classes=3)
        model2.load_state_dict(state)
        model2.eval()
        
        with torch.no_grad():
            out2 = model2(x)
        
        assert torch.allclose(out1, out2)


class TestBatchNormalization:
    """Tests for batch normalization behavior."""

    def test_batch_size_one(self):
        """Test model works with batch size 1 in eval mode."""
        model = ConfigurableCNN(in_channels=4, num_classes=3)
        model.eval()
        
        x = torch.randn(1, 4, 500)
        with torch.no_grad():
            out = model(x)
        
        assert out.shape == (1, 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGPUCompatibility:
    """Tests for GPU compatibility."""

    def test_cuda_forward(self):
        """Test forward pass on GPU."""
        model = ConfigurableCNN(in_channels=4, num_classes=3).cuda()
        x = torch.randn(8, 4, 1000).cuda()
        
        out = model(x)
        
        assert out.device.type == "cuda"
        assert out.shape == (8, 3)

    def test_cuda_training(self):
        """Test training step on GPU."""
        model = ConfigurableCNN(in_channels=4, num_classes=3).cuda()
        model.train()
        
        x = torch.randn(8, 4, 1000).cuda()
        y = torch.randint(0, 3, (8,)).cuda()
        
        out = model(x)
        loss = nn.CrossEntropyLoss()(out, y)
        loss.backward()
        
        assert loss.device.type == "cuda"
