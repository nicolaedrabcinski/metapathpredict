"""
Attention mechanisms for sequence classification.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SelfAttention(nn.Module):
    """
    Self-attention mechanism for 1D sequences.

    Computes attention over sequence positions to capture long-range dependencies.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        """
        Initialize self-attention.

        Args:
            dim: Input dimension.
            num_heads: Number of attention heads.
            head_dim: Dimension per head (default: dim // num_heads).
            dropout: Dropout rate for attention weights.
            bias: Whether to use bias in projections.
        """
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim or dim // num_heads
        self.scale = self.head_dim ** -0.5

        inner_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, inner_dim * 3, bias=bias)
        self.proj = nn.Linear(inner_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """
        Apply self-attention.

        Args:
            x: Input tensor of shape (batch, seq_len, dim).
            mask: Optional attention mask of shape (batch, seq_len) or (batch, seq_len, seq_len).

        Returns:
            Output tensor of shape (batch, seq_len, dim).
        """
        B, N, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # Each: (B, num_heads, N, head_dim)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Compute output
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)

        return x


class AttentionBlock(nn.Module):
    """
    Transformer-style attention block with feed-forward network.

    Architecture:
        x -> LayerNorm -> SelfAttention -> Dropout -> (+x) ->
          -> LayerNorm -> FFN -> Dropout -> (+x)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        """
        Initialize attention block.

        Args:
            dim: Input/output dimension.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of MLP hidden dimension to input dimension.
            dropout: Dropout rate.
            attention_dropout: Dropout rate for attention weights.
            drop_path: Drop path rate for stochastic depth.
        """
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads, dropout=attention_dropout)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Forward pass."""
        # Self-attention with residual
        x = x + self.drop_path(self.attn(self.norm1(x), mask))

        # FFN with residual
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class DropPath(nn.Module):
    """Drop path (stochastic depth) for residual connections."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()

        return x.div(keep_prob) * random_tensor


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for sequences.
    """

    def __init__(self, dim: int, max_len: int = 5000, dropout: float = 0.0):
        """
        Initialize positional encoding.

        Args:
            dim: Embedding dimension.
            max_len: Maximum sequence length.
            dropout: Dropout rate.
        """
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        # Create positional encodings
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, dim)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch, seq_len, dim).

        Returns:
            Output with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding.
    """

    def __init__(self, dim: int, max_len: int = 5000, dropout: float = 0.0):
        """
        Initialize learnable positional encoding.

        Args:
            dim: Embedding dimension.
            max_len: Maximum sequence length.
            dropout: Dropout rate.
        """
        super().__init__()

        self.pe = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Add learnable positional encoding."""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ConvAttention(nn.Module):
    """
    Convolutional attention - combines CNN feature extraction with attention.

    Uses convolutions for local feature extraction and attention for
    global context aggregation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Initialize convolutional attention.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            kernel_size: Kernel size for convolution.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
        """
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.norm1 = nn.BatchNorm1d(out_channels)

        self.attention = SelfAttention(out_channels, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(out_channels)

        self.ffn = nn.Sequential(
            nn.Linear(out_channels, out_channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels * 4, out_channels),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input of shape (batch, channels, seq_len) or (batch, seq_len, channels).

        Returns:
            Output of shape (batch, seq_len, out_channels).
        """
        # Ensure correct format for conv
        if x.dim() == 3 and x.shape[1] != self.conv.in_channels:
            x = x.transpose(1, 2)

        # Conv processing
        x = self.conv(x)
        x = self.norm1(x)
        x = F.gelu(x)

        # Transpose for attention: (B, C, L) -> (B, L, C)
        x = x.transpose(1, 2)

        # Self-attention with residual
        x = x + self.attention(self.norm2(x))

        # FFN with residual
        x = x + self.ffn(self.norm3(x))

        return x


class GlobalAttentionPooling(nn.Module):
    """
    Global attention pooling for sequence classification.

    Learns to weight different positions based on their importance
    for the classification task.
    """

    def __init__(self, dim: int, num_heads: int = 1):
        """
        Initialize global attention pooling.

        Args:
            dim: Input dimension.
            num_heads: Number of attention heads (usually 1 for pooling).
        """
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Learnable query for pooling
        self.query = nn.Parameter(torch.randn(1, num_heads, 1, self.head_dim))

        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

        self.scale = self.head_dim ** -0.5

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """
        Pool sequence to single vector using attention.

        Args:
            x: Input of shape (batch, seq_len, dim).
            mask: Optional mask of shape (batch, seq_len).

        Returns:
            Pooled output of shape (batch, dim).
        """
        B, N, C = x.shape

        # Compute keys and values
        k = self.key(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.value(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Expand query for batch
        q = self.query.expand(B, -1, -1, -1)  # (B, num_heads, 1, head_dim)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, 1, N)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, N)
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)

        # Compute output
        out = (attn @ v).transpose(1, 2).reshape(B, -1)  # (B, dim)

        return out
