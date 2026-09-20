"""
Base model class and utilities for PyTorch models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all models in MetaPathPredict.

    Provides common utilities for weight initialization, parameter counting,
    and model information.
    """

    def __init__(self):
        super().__init__()
        self._is_initialized = False

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass - must be implemented by subclasses."""
        pass

    def initialize_weights(self, method: str = "kaiming") -> None:
        """
        Initialize model weights.

        Args:
            method: Initialization method ("xavier", "kaiming", "orthogonal").
        """
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                if method == "xavier":
                    nn.init.xavier_uniform_(module.weight)
                elif method == "kaiming":
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                elif method == "orthogonal":
                    nn.init.orthogonal_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        self._is_initialized = True

    @property
    def num_parameters(self) -> int:
        """Total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_parameter_summary(self) -> dict[str, int]:
        """Get parameter summary by layer type."""
        summary = {}
        for _name, module in self.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                module_type = type(module).__name__
                num_params = sum(p.numel() for p in module.parameters())
                summary[module_type] = summary.get(module_type, 0) + num_params
        return summary

    def freeze(self) -> None:
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def freeze_backbone(self) -> None:
        """Freeze backbone (all except classifier head)."""
        for name, param in self.named_parameters():
            if "classifier" not in name and "head" not in name and "fc" not in name:
                param.requires_grad = False

    def get_device(self) -> torch.device:
        """Get the device of the model parameters."""
        return next(self.parameters()).device

    def __repr__(self) -> str:
        """String representation with parameter count."""
        base_repr = super().__repr__()
        return f"{base_repr}\n\nTotal parameters: {self.num_parameters:,}\nTrainable: {self.num_trainable_parameters:,}"


def group_count(channels: int, target: int = 32) -> int:
    """Largest number of groups <= target that divides `channels`."""
    groups = min(target, channels)
    while channels % groups:
        groups -= 1
    return groups


def reverse_complement(x: Tensor) -> Tensor:
    """Reverse complement of one-hot sequences [batch, 4, length] with channels A, C, G, T."""
    x = torch.flip(x, dims=[-1])
    return x.index_select(dim=1, index=torch.tensor([3, 2, 1, 0], device=x.device))  # A<->T, C<->G


class ConvBlock(nn.Module):
    """
    Convolutional block with optional batch norm, activation, and pooling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | str = "same",
        dilation: int = 1,
        groups: int = 1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        use_group_norm: bool = False,
        activation: nn.Module | None = None,
        dropout: float = 0.0,
        pool_size: int | None = None,
    ):
        super().__init__()

        # Handle padding
        if padding == "same":
            padding = (kernel_size - 1) // 2 * dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=not (use_batch_norm or use_group_norm),  # normalisation layer supplies the shift
        )

        self.norm = None
        if use_batch_norm:
            self.norm = nn.BatchNorm1d(out_channels)
        elif use_group_norm:
            self.norm = nn.GroupNorm(group_count(out_channels), out_channels)
        elif use_layer_norm:
            self.norm = nn.LayerNorm(out_channels)

        self.activation = activation or nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.pool = nn.MaxPool1d(pool_size) if pool_size is not None else None

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = self.conv(x)

        if self.norm is not None:
            if isinstance(self.norm, nn.LayerNorm):
                x = x.transpose(1, 2)
                x = self.norm(x)
                x = x.transpose(1, 2)
            else:
                x = self.norm(x)

        x = self.activation(x)

        if self.dropout is not None:
            x = self.dropout(x)

        if self.pool is not None:
            x = self.pool(x)

        return x


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.

    From "Squeeze-and-Excitation Networks" (Hu et al., 2018).
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        reduced_channels = max(channels // reduction, 8)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply channel attention."""
        b, c, _ = x.shape

        # Squeeze
        y = self.pool(x).view(b, c)

        # Excitation
        y = self.fc(y).view(b, c, 1)

        # Scale
        return x * y


class DropPath(nn.Module):
    """
    Drop path (stochastic depth) for residual connections.

    From "Deep Networks with Stochastic Depth" (Huang et al., 2016).
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        """Apply drop path during training."""
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()

        return x.div(keep_prob) * random_tensor
