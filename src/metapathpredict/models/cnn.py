"""
CNN architectures for sequence classification.

Implements various CNN architectures including:
- Multi-scale CNN with different kernel sizes
- Residual CNN
- Dilated CNN for larger receptive field
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from metapathpredict.models.base import BaseModel, ConvBlock, DropPath, SEBlock


class ResidualBlock(nn.Module):
    """
    Residual block for 1D convolutions.

    Architecture:
        x -> Conv -> BN -> ReLU -> Conv -> BN -> (+x) -> ReLU
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
        use_se: bool = True,
        drop_path: float = 0.0,
    ):
        super().__init__()

        padding = (kernel_size - 1) // 2 * dilation

        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)

        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)

        self.se = SEBlock(channels) if use_se else None
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else None

        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with residual connection."""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.se is not None:
            out = self.se(out)

        if self.drop_path is not None:
            out = self.drop_path(out)

        out = out + identity
        out = self.activation(out)

        return out


class MultiScaleBranch(nn.Module):
    """
    Single branch of multi-scale CNN with specific kernel size.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []

        # First layer changes channels
        layers.append(ConvBlock(
            in_channels,
            out_channels,
            kernel_size,
            dropout=dropout,
        ))

        # Additional layers maintain channels
        for _ in range(num_layers - 1):
            layers.append(ConvBlock(
                out_channels,
                out_channels,
                kernel_size,
                dropout=dropout,
            ))

        self.layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through branch."""
        x = self.layers(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        return x


class MultiScaleCNN(BaseModel):
    """
    Multi-scale CNN that processes input with multiple kernel sizes.

    Inspired by InceptionNet, this model uses parallel convolutional
    branches with different kernel sizes to capture features at
    different scales simultaneously.

    Architecture:
        Input (B, L, 4)
           ↓ transpose
        (B, 4, L)
           ↓
        ┌─────────┬─────────┬─────────┐
        │ k=5     │ k=7     │ k=11    │  Parallel branches
        │ branch  │ branch  │ branch  │
        └────┬────┴────┬────┴────┬────┘
             │         │         │
             └─────────┴─────────┘
                      ↓ concat
                  (B, total_features)
                      ↓
                  Classifier
                      ↓
                  (B, num_classes)
    """

    def __init__(
        self,
        seq_length: int = 1000,
        num_classes: int = 3,
        kernel_sizes: list[int] | None = None,
        branch_channels: int = 256,
        num_conv_layers: int = 3,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
    ):
        """
        Initialize Multi-scale CNN.

        Args:
            seq_length: Input sequence length.
            num_classes: Number of output classes.
            kernel_sizes: List of kernel sizes for each branch.
            branch_channels: Number of channels per branch.
            num_conv_layers: Number of conv layers per branch.
            hidden_dims: Hidden dimensions for classifier.
            dropout: Dropout rate.
            use_batch_norm: Whether to use batch normalization.
        """
        super().__init__()

        self.seq_length = seq_length
        self.num_classes = num_classes
        self.kernel_sizes = kernel_sizes or [5, 7, 11]

        # Input channels (one-hot encoded nucleotides)
        in_channels = 4

        # Create parallel branches
        self.branches = nn.ModuleList([
            MultiScaleBranch(
                in_channels=in_channels,
                out_channels=branch_channels,
                kernel_size=k,
                num_layers=num_conv_layers,
                dropout=dropout,
            )
            for k in self.kernel_sizes
        ])

        # Total features after concatenation
        total_features = branch_channels * len(self.kernel_sizes)

        # Classifier head
        hidden_dims = hidden_dims or [512, 256]
        classifier_layers = []

        prev_dim = total_features
        for dim in hidden_dims:
            classifier_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim) if use_batch_norm else nn.Identity(),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim

        classifier_layers.append(nn.Linear(prev_dim, num_classes))

        self.classifier = nn.Sequential(*classifier_layers)

        # Initialize weights
        self.initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, 4) or (batch, 4, seq_len).

        Returns:
            Logits of shape (batch, num_classes).
        """
        # Handle both input formats
        if x.dim() == 3 and x.shape[-1] == 4:
            x = x.transpose(1, 2)  # (B, L, 4) -> (B, 4, L)

        # Process through each branch
        branch_outputs = [branch(x) for branch in self.branches]

        # Concatenate branch outputs
        merged = torch.cat(branch_outputs, dim=1)

        # Classify
        logits = self.classifier(merged)

        return logits

    def get_features(self, x: Tensor) -> Tensor:
        """Extract features before classifier."""
        if x.dim() == 3 and x.shape[-1] == 4:
            x = x.transpose(1, 2)

        branch_outputs = [branch(x) for branch in self.branches]
        return torch.cat(branch_outputs, dim=1)


class ResidualCNN(BaseModel):
    """
    Residual CNN with stacked residual blocks.

    Uses dilated convolutions for larger receptive field without
    increasing parameters.
    """

    def __init__(
        self,
        seq_length: int = 1000,
        num_classes: int = 3,
        base_channels: int = 64,
        num_blocks: list[int] | None = None,
        kernel_size: int = 7,
        dropout: float = 0.3,
        use_se: bool = True,
        drop_path_rate: float = 0.1,
    ):
        """
        Initialize Residual CNN.

        Args:
            seq_length: Input sequence length.
            num_classes: Number of output classes.
            base_channels: Base number of channels.
            num_blocks: Number of residual blocks at each stage.
            kernel_size: Kernel size for convolutions.
            dropout: Dropout rate.
            use_se: Whether to use Squeeze-and-Excitation blocks.
            drop_path_rate: Maximum drop path rate.
        """
        super().__init__()

        self.seq_length = seq_length
        self.num_classes = num_classes
        num_blocks = num_blocks or [2, 2, 2, 2]

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(4, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # Build stages with increasing channels and dilation
        self.stages = nn.ModuleList()
        channels = base_channels

        # Calculate drop path rates for stochastic depth
        total_blocks = sum(num_blocks)
        drop_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        block_idx = 0
        for stage_idx, n_blocks in enumerate(num_blocks):
            stage_layers = []

            # Downsample at start of each stage (except first)
            if stage_idx > 0:
                stage_layers.append(nn.Sequential(
                    nn.Conv1d(channels, channels * 2, kernel_size=1),
                    nn.BatchNorm1d(channels * 2),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                ))
                channels *= 2

            # Add residual blocks with increasing dilation
            for i in range(n_blocks):
                dilation = 2 ** min(i, 3)  # Cap dilation at 8
                stage_layers.append(ResidualBlock(
                    channels=channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    use_se=use_se,
                    drop_path=drop_rates[block_idx],
                ))
                block_idx += 1

            self.stages.append(nn.Sequential(*stage_layers))

        # Global pooling and classifier
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels // 2, num_classes),
        )

        self.initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        if x.dim() == 3 and x.shape[-1] == 4:
            x = x.transpose(1, 2)

        x = self.stem(x)

        for stage in self.stages:
            x = stage(x)

        x = self.pool(x)
        logits = self.classifier(x)

        return logits

    def get_features(self, x: Tensor) -> Tensor:
        """Extract features before classifier."""
        if x.dim() == 3 and x.shape[-1] == 4:
            x = x.transpose(1, 2)

        x = self.stem(x)

        for stage in self.stages:
            x = stage(x)

        x = self.pool(x)
        return x.flatten(1)


class SimpleCNN(BaseModel):
    """
    Simple CNN baseline for comparison.

    A straightforward CNN without complex mechanisms, useful
    for quick experiments and as a baseline.
    """

    def __init__(
        self,
        seq_length: int = 1000,
        num_classes: int = 3,
        num_filters: list[int] | None = None,
        kernel_size: int = 5,
        dropout: float = 0.3,
    ):
        """
        Initialize Simple CNN.

        Args:
            seq_length: Input sequence length.
            num_classes: Number of output classes.
            num_filters: Number of filters at each layer.
            kernel_size: Kernel size for convolutions.
            dropout: Dropout rate.
        """
        super().__init__()

        num_filters = num_filters or [64, 128, 256]

        # Build conv layers
        layers = []
        in_channels = 4

        for out_channels in num_filters:
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(dropout),
            ])
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_filters[-1], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        self.initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        if x.dim() == 3 and x.shape[-1] == 4:
            x = x.transpose(1, 2)

        x = self.features(x)
        x = self.pool(x)
        logits = self.classifier(x)

        return logits


def create_cnn_model(
    model_type: Literal["simple", "multi_scale", "residual"] = "multi_scale",
    seq_length: int = 1000,
    num_classes: int = 3,
    **kwargs,
) -> BaseModel:
    """
    Factory function to create CNN models.

    Args:
        model_type: Type of model to create.
        seq_length: Input sequence length.
        num_classes: Number of output classes.
        **kwargs: Additional model-specific arguments.

    Returns:
        Instantiated model.
    """
    models = {
        "simple": SimpleCNN,
        "multi_scale": MultiScaleCNN,
        "residual": ResidualCNN,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")

    return models[model_type](seq_length=seq_length, num_classes=num_classes, **kwargs)
