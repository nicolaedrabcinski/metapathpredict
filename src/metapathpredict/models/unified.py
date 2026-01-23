"""
Unified classifier model combining CNN and attention mechanisms.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from metapathpredict.models.base import BaseModel, ConvBlock, SEBlock
from metapathpredict.models.attention import (
    AttentionBlock,
    GlobalAttentionPooling,
    PositionalEncoding,
)


class UnifiedClassifier(BaseModel):
    """
    Unified classifier combining multi-scale CNN with attention.
    
    This is the main model architecture that combines:
    1. Multi-scale CNN branches for local feature extraction
    2. Attention mechanism for global context
    3. SE blocks for channel attention
    4. Residual connections throughout
    
    Architecture:
        Input (B, L, 4)
              ↓
        Stem (initial conv + norm)
              ↓
        ┌─────────┬─────────┬─────────┐
        │ k=5     │ k=9     │ k=15    │  Multi-scale CNN branches
        │ branch  │ branch  │ branch  │
        └────┬────┴────┬────┴────┬────┘
             │         │         │
             └─────────┴─────────┘
                      ↓ concat + SE
                  Feature fusion
                      ↓
              Attention blocks (optional)
                      ↓
              Global attention pooling
                      ↓
                  Classifier
                      ↓
                  (B, num_classes)
    """
    
    def __init__(
        self,
        seq_length: int = 1000,
        num_classes: int = 3,
        # CNN settings
        kernel_sizes: list[int] | None = None,
        base_channels: int = 64,
        num_conv_layers: int = 4,
        # Attention settings
        use_attention: bool = True,
        num_attention_blocks: int = 2,
        num_attention_heads: int = 8,
        attention_dropout: float = 0.1,
        # Classifier settings
        hidden_dims: list[int] | None = None,
        # Regularization
        dropout: float = 0.3,
        drop_path_rate: float = 0.1,
        use_se: bool = True,
        # Other
        pooling: Literal["attention", "avg", "max"] = "attention",
    ):
        """
        Initialize Unified Classifier.
        
        Args:
            seq_length: Input sequence length.
            num_classes: Number of output classes.
            kernel_sizes: Kernel sizes for multi-scale branches.
            base_channels: Base number of channels.
            num_conv_layers: Number of conv layers per branch.
            use_attention: Whether to use attention blocks.
            num_attention_blocks: Number of attention blocks.
            num_attention_heads: Number of attention heads.
            attention_dropout: Dropout for attention.
            hidden_dims: Hidden dimensions for classifier.
            dropout: Dropout rate.
            drop_path_rate: Drop path rate for stochastic depth.
            use_se: Whether to use SE blocks.
            pooling: Pooling method ("attention", "avg", "max").
        """
        super().__init__()
        
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.kernel_sizes = kernel_sizes or [5, 9, 15]
        self.use_attention = use_attention
        self.pooling_type = pooling
        
        # Stem: initial projection
        self.stem = nn.Sequential(
            nn.Conv1d(4, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.GELU(),
        )
        
        # Multi-scale CNN branches
        branch_channels = base_channels * 2
        self.branches = nn.ModuleList()
        
        for k in self.kernel_sizes:
            branch = self._make_branch(
                in_channels=base_channels,
                out_channels=branch_channels,
                kernel_size=k,
                num_layers=num_conv_layers,
                dropout=dropout,
                use_se=use_se,
            )
            self.branches.append(branch)
        
        # Feature fusion
        fused_channels = branch_channels * len(self.kernel_sizes)
        self.fusion = nn.Sequential(
            nn.Conv1d(fused_channels, branch_channels, kernel_size=1),
            nn.BatchNorm1d(branch_channels),
            nn.GELU(),
        )
        
        if use_se:
            self.fusion_se = SEBlock(branch_channels)
        else:
            self.fusion_se = nn.Identity()
        
        # Attention blocks (optional)
        if use_attention:
            self.pos_encoding = PositionalEncoding(
                branch_channels,
                max_len=seq_length,
                dropout=dropout,
            )
            
            # Stochastic depth rates
            drop_rates = [x.item() for x in torch.linspace(0, drop_path_rate, num_attention_blocks)]
            
            self.attention_blocks = nn.ModuleList([
                AttentionBlock(
                    dim=branch_channels,
                    num_heads=num_attention_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    drop_path=drop_rates[i],
                )
                for i in range(num_attention_blocks)
            ])
            
            self.attention_norm = nn.LayerNorm(branch_channels)
        
        # Pooling
        if pooling == "attention":
            self.pool = GlobalAttentionPooling(branch_channels, num_heads=num_attention_heads)
        elif pooling == "avg":
            self.pool = nn.AdaptiveAvgPool1d(1)
        else:
            self.pool = nn.AdaptiveMaxPool1d(1)
        
        # Classifier head
        hidden_dims = hidden_dims or [256, 128]
        pooled_dim = branch_channels
        
        classifier_layers = []
        prev_dim = pooled_dim
        
        for i, dim in enumerate(hidden_dims):
            classifier_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(dropout if i < len(hidden_dims) - 1 else dropout / 2),
            ])
            prev_dim = dim
        
        classifier_layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*classifier_layers)
        
        # Initialize weights
        self.initialize_weights()
    
    def _make_branch(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num_layers: int,
        dropout: float,
        use_se: bool,
    ) -> nn.Module:
        """Create a single CNN branch."""
        layers = []
        
        # First layer: change channels
        layers.append(ConvBlock(
            in_channels,
            out_channels,
            kernel_size,
            dropout=dropout,
        ))
        
        # Subsequent layers: maintain channels with residual connections
        for i in range(num_layers - 1):
            # Conv block
            layers.append(ConvBlock(
                out_channels,
                out_channels,
                kernel_size,
                dropout=dropout,
            ))
            
            # SE block every 2 layers
            if use_se and (i + 1) % 2 == 0:
                layers.append(SEBlock(out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, 4) or (batch, 4, seq_len).
        
        Returns:
            Logits of shape (batch, num_classes).
        """
        # Handle input format
        if x.dim() == 3 and x.shape[-1] == 4:
            x = x.transpose(1, 2)  # (B, L, 4) -> (B, 4, L)
        
        # Stem
        x = self.stem(x)
        
        # Multi-scale branches
        branch_outputs = [branch(x) for branch in self.branches]
        
        # Concatenate and fuse
        x = torch.cat(branch_outputs, dim=1)
        x = self.fusion(x)
        x = self.fusion_se(x)
        
        # Attention (if enabled)
        if self.use_attention:
            # Transpose for attention: (B, C, L) -> (B, L, C)
            x = x.transpose(1, 2)
            x = self.pos_encoding(x)
            
            for attn_block in self.attention_blocks:
                x = attn_block(x)
            
            x = self.attention_norm(x)
            
            # Pool
            if self.pooling_type == "attention":
                x = self.pool(x)  # (B, L, C) -> (B, C)
            else:
                x = x.transpose(1, 2)  # (B, L, C) -> (B, C, L)
                x = self.pool(x).squeeze(-1)  # (B, C)
        else:
            # Direct pooling without attention
            if self.pooling_type == "attention":
                x = x.transpose(1, 2)
                x = self.pool(x)
            else:
                x = self.pool(x).squeeze(-1)
        
        # Classify
        logits = self.classifier(x)
        
        return logits
    
    def get_features(self, x: Tensor, return_attention: bool = False) -> Tensor | tuple[Tensor, list]:
        """
        Extract features before classifier.
        
        Args:
            x: Input tensor.
            return_attention: Whether to return attention weights.
        
        Returns:
            Features tensor, optionally with attention weights.
        """
        if x.dim() == 3 and x.shape[-1] == 4:
            x = x.transpose(1, 2)
        
        x = self.stem(x)
        branch_outputs = [branch(x) for branch in self.branches]
        x = torch.cat(branch_outputs, dim=1)
        x = self.fusion(x)
        x = self.fusion_se(x)
        
        attention_weights = []
        
        if self.use_attention:
            x = x.transpose(1, 2)
            x = self.pos_encoding(x)
            
            for attn_block in self.attention_blocks:
                x = attn_block(x)
                # Could collect attention weights here if needed
            
            x = self.attention_norm(x)
            
            if self.pooling_type == "attention":
                x = self.pool(x)
            else:
                x = x.transpose(1, 2)
                x = self.pool(x).squeeze(-1)
        else:
            if self.pooling_type == "attention":
                x = x.transpose(1, 2)
                x = self.pool(x)
            else:
                x = self.pool(x).squeeze(-1)
        
        if return_attention:
            return x, attention_weights
        return x
    
    def predict_proba(self, x: Tensor) -> Tensor:
        """
        Get prediction probabilities.
        
        Args:
            x: Input tensor.
        
        Returns:
            Probability tensor of shape (batch, num_classes).
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)
    
    def predict(self, x: Tensor) -> Tensor:
        """
        Get class predictions.
        
        Args:
            x: Input tensor.
        
        Returns:
            Class indices of shape (batch,).
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)


def create_unified_model(
    seq_length: int = 1000,
    num_classes: int = 3,
    size: Literal["small", "base", "large"] = "base",
    **kwargs,
) -> UnifiedClassifier:
    """
    Factory function to create unified models of different sizes.
    
    Args:
        seq_length: Input sequence length.
        num_classes: Number of output classes.
        size: Model size ("small", "base", "large").
        **kwargs: Additional arguments.
    
    Returns:
        UnifiedClassifier instance.
    """
    configs = {
        "small": {
            "kernel_sizes": [5, 9],
            "base_channels": 32,
            "num_conv_layers": 2,
            "num_attention_blocks": 1,
            "num_attention_heads": 4,
            "hidden_dims": [128],
        },
        "base": {
            "kernel_sizes": [5, 9, 15],
            "base_channels": 64,
            "num_conv_layers": 3,
            "num_attention_blocks": 2,
            "num_attention_heads": 8,
            "hidden_dims": [256, 128],
        },
        "large": {
            "kernel_sizes": [5, 9, 15, 21],
            "base_channels": 96,
            "num_conv_layers": 4,
            "num_attention_blocks": 4,
            "num_attention_heads": 12,
            "hidden_dims": [512, 256],
        },
    }
    
    if size not in configs:
        raise ValueError(f"Unknown size: {size}. Choose from {list(configs.keys())}")
    
    config = {**configs[size], **kwargs}
    
    return UnifiedClassifier(
        seq_length=seq_length,
        num_classes=num_classes,
        **config,
    )
