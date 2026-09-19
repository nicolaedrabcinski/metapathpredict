"""
Configurable CNN with variable kernel sizes.

Supports kernel sizes: 5, 7, 10 (or custom).
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel, ConvBlock, SEBlock, group_count, reverse_complement


class ConfigurableCNN(BaseModel):
    """
    CNN with configurable kernel sizes for DNA sequence classification.
    
    Supports multiple kernel size configurations:
    - Small (5): Better for short motifs
    - Medium (7): Balanced approach  
    - Large (10): Better for longer patterns
    - Multi-scale: Combines multiple kernel sizes
    """
    
    KERNEL_PRESETS = {
        "small": [5, 5, 5],
        "medium": [7, 7, 7],
        "large": [10, 10, 10],
        "progressive": [5, 7, 10],
        "multi": [5, 7, 10],  # Multi-scale in parallel
    }
    
    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 3,
        kernel_preset: Literal["small", "medium", "large", "progressive", "multi"] = "medium",
        custom_kernels: list[int] | None = None,
        base_channels: int = 64,
        num_blocks: int = 3,
        use_se: bool = True,
        dropout: float = 0.3,
        norm: Literal["batch", "group"] = "batch",
        pool: Literal["avg", "max", "avgmax"] = "avg",
    ):
        """
        Initialize configurable CNN.
        
        Args:
            in_channels: Input channels (4 for DNA one-hot).
            num_classes: Number of output classes.
            kernel_preset: Preset kernel configuration.
            custom_kernels: Custom kernel sizes (overrides preset).
            base_channels: Base number of channels.
            num_blocks: Number of conv blocks.
            use_se: Whether to use SE attention.
            dropout: Dropout rate.
            norm: "batch" (BatchNorm) or "group" (GroupNorm, statistics never mix samples).
            pool: how the feature map is reduced over positions: "avg" (composition-like), "max"
                (presence of a motif) or "avgmax" (both, twice as many features).
        """
        super().__init__()
        if norm not in ("batch", "group"):
            raise ValueError(f"norm must be 'batch' or 'group', got {norm!r}")
        self.norm_type = norm
        if pool not in ("avg", "max", "avgmax"):
            raise ValueError(f"pool must be 'avg', 'max' or 'avgmax', got {pool!r}")
        self.pool_type = pool
        self._pool_factor = 2 if pool == "avgmax" else 1
        
        # Determine kernel sizes
        if custom_kernels is not None:
            self.kernel_sizes = custom_kernels
        else:
            self.kernel_sizes = self.KERNEL_PRESETS[kernel_preset]
        
        self.kernel_preset = kernel_preset
        self.use_multi_scale = (kernel_preset == "multi")
        
        if self.use_multi_scale:
            self._build_multiscale(in_channels, num_classes, base_channels, use_se, dropout)
        else:
            self._build_sequential(in_channels, num_classes, base_channels, num_blocks, use_se, dropout)
    
    def _build_sequential(
        self,
        in_channels: int,
        num_classes: int,
        base_channels: int,
        num_blocks: int,
        use_se: bool,
        dropout: float,
    ) -> None:
        """Build sequential CNN architecture."""
        layers = []
        
        current_channels = in_channels
        
        for i, kernel_size in enumerate(self.kernel_sizes[:num_blocks]):
            out_channels = base_channels * (2 ** i)
            
            layers.append(
                ConvBlock(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    pool_size=2,
                    dropout=dropout if i > 0 else 0.0,
                    **self._norm_flags(),
                )
            )
            
            if use_se:
                layers.append(SEBlock(out_channels))
            
            current_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(current_channels * self._pool_factor, current_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(current_channels // 2, num_classes),
        )
        
        self._final_channels = current_channels * self._pool_factor
    
    def _norm_flags(self) -> dict[str, bool]:
        return {"use_batch_norm": self.norm_type == "batch", "use_group_norm": self.norm_type == "group"}

    def _build_multiscale(
        self,
        in_channels: int,
        num_classes: int,
        base_channels: int,
        use_se: bool,
        dropout: float,
    ) -> None:
        """Build multi-scale parallel CNN architecture."""
        self.branches = nn.ModuleList()
        
        for kernel_size in self.kernel_sizes:
            branch = nn.Sequential(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=base_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    pool_size=2,
                    dropout=0.0,
                    **self._norm_flags(),
                ),
                SEBlock(base_channels) if use_se else nn.Identity(),
                ConvBlock(
                    in_channels=base_channels,
                    out_channels=base_channels * 2,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    pool_size=2,
                    dropout=dropout,
                    **self._norm_flags(),
                ),
            )
            self.branches.append(branch)
        
        # Fusion layer
        combined_channels = base_channels * 2 * len(self.kernel_sizes)
        
        self.fusion = nn.Sequential(
            nn.Conv1d(combined_channels, base_channels * 4, kernel_size=1),
            (
                nn.BatchNorm1d(base_channels * 4)
                if self.norm_type == "batch"
                else nn.GroupNorm(group_count(base_channels * 4), base_channels * 4)
            ),
            nn.ReLU(inplace=True),
        )
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base_channels * 4 * self._pool_factor, base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 2, num_classes),
        )
        
        self._final_channels = base_channels * 4 * self._pool_factor
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.use_multi_scale:
            # Process through parallel branches
            branch_outputs = [branch(x) for branch in self.branches]
            
            # Align lengths (take minimum)
            min_len = min(out.size(2) for out in branch_outputs)
            branch_outputs = [out[:, :, :min_len] for out in branch_outputs]
            
            # Concatenate and fuse
            x = torch.cat(branch_outputs, dim=1)
            x = self.fusion(x)
        else:
            x = self.features(x)
        
        x = self._pooled(x)
        x = self.classifier(x)
        
        return x
    
    def _pooled(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool_type == "avg":
            return self.pool(x).flatten(1)
        if self.pool_type == "max":
            return self.max_pool(x).flatten(1)
        return torch.cat([self.pool(x).flatten(1), self.max_pool(x).flatten(1)], dim=1)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get feature embeddings before classifier."""
        if self.use_multi_scale:
            branch_outputs = [branch(x) for branch in self.branches]
            min_len = min(out.size(2) for out in branch_outputs)
            branch_outputs = [out[:, :, :min_len] for out in branch_outputs]
            x = torch.cat(branch_outputs, dim=1)
            x = self.fusion(x)
        else:
            x = self.features(x)
        
        x = self._pooled(x)
        
        return x


class RCShared(nn.Module):
    """
    A ConfigurableCNN that looks at both strands with the same weights. The embeddings of a sequence and of
    its reverse complement are combined (mean or max), so the prediction is identical for the two strands
    by construction, instead of being learned from reverse-complement augmentation. Costs two forward passes.
    """

    def __init__(self, base: "ConfigurableCNN", mode: str = "mean"):
        super().__init__()
        if mode not in ("mean", "max"):
            raise ValueError(f"mode must be 'mean' or 'max', got {mode!r}")
        self.base, self.mode = base, mode

    @property
    def classifier(self) -> nn.Module:
        return self.base.classifier

    @property
    def _final_channels(self) -> int:
        return self.base._final_channels

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        forward, reverse = self.base.get_embeddings(x), self.base.get_embeddings(reverse_complement(x))
        return (forward + reverse) / 2 if self.mode == "mean" else torch.maximum(forward, reverse)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base.classifier(self.get_embeddings(x))


# Expose KERNEL_PRESETS at module level for backwards compatibility
KERNEL_PRESETS = ConfigurableCNN.KERNEL_PRESETS


def create_configurable_cnn(
    kernel_size: int | str = 7,
    in_channels: int = 4,
    num_classes: int = 3,
    **kwargs,
) -> ConfigurableCNN:
    """
    Factory function to create ConfigurableCNN.
    
    Args:
        kernel_size: Kernel size (5, 7, 10) or preset name.
        in_channels: Input channels.
        num_classes: Number of classes.
        **kwargs: Additional arguments.
    
    Returns:
        ConfigurableCNN instance.
    """
    if isinstance(kernel_size, str):
        return ConfigurableCNN(
            in_channels=in_channels,
            num_classes=num_classes,
            kernel_preset=kernel_size,
            **kwargs,
        )
    elif isinstance(kernel_size, int):
        preset_map = {5: "small", 7: "medium", 10: "large"}
        preset = preset_map.get(kernel_size, "medium")
        return ConfigurableCNN(
            in_channels=in_channels,
            num_classes=num_classes,
            kernel_preset=preset,
            custom_kernels=[kernel_size] * 3,
            **kwargs,
        )
    else:
        # Assume it's a list
        return ConfigurableCNN(
            in_channels=in_channels,
            num_classes=num_classes,
            custom_kernels=list(kernel_size),
            **kwargs,
        )
