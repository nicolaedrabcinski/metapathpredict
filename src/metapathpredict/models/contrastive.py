"""
Contrastive Learning for DNA sequence representation.

Implements SimCLR-style contrastive learning for learning
robust sequence embeddings.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base import BaseModel
from .configurable_cnn import ConfigurableCNN


class ProjectionHead(nn.Module):
    """
    MLP projection head for contrastive learning.
    
    Maps embeddings to a space where contrastive loss is applied.
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        out_dim: int = 128,
    ):
        """
        Initialize projection head.
        
        Args:
            in_dim: Input dimension.
            hidden_dim: Hidden layer dimension.
            out_dim: Output dimension.
        """
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x)


class ContrastiveEncoder(BaseModel):
    """
    Contrastive learning encoder for DNA sequences.
    
    Uses a CNN backbone with a projection head for
    SimCLR-style contrastive learning.
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        backbone: str = "medium",
        projection_dim: int = 128,
        hidden_dim: int = 256,
        base_channels: int = 64,
    ):
        """
        Initialize contrastive encoder.
        
        Args:
            in_channels: Input channels (4 for DNA).
            backbone: CNN backbone preset.
            projection_dim: Projection head output dimension.
            hidden_dim: Projection head hidden dimension.
            base_channels: Base channels for backbone.
        """
        super().__init__()
        
        # Backbone encoder (without classifier)
        self.encoder = ConfigurableCNN(
            in_channels=in_channels,
            num_classes=3,  # Dummy, we use embeddings
            kernel_preset=backbone,
            base_channels=base_channels,
        )
        
        # Get embedding dimension
        embed_dim = self.encoder._final_channels
        
        # Projection head
        self.projection = ProjectionHead(
            in_dim=embed_dim,
            hidden_dim=hidden_dim,
            out_dim=projection_dim,
        )
        
        self.embed_dim = embed_dim
        self.projection_dim = projection_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning projected embeddings.
        
        Args:
            x: Input tensor [batch, channels, length].
        
        Returns:
            Projected embeddings [batch, projection_dim].
        """
        embeddings = self.encoder.get_embeddings(x)
        projections = self.projection(embeddings)
        return F.normalize(projections, dim=1)
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings without projection (for downstream tasks)."""
        return self.encoder.get_embeddings(x)


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).
    
    The contrastive loss used in SimCLR.
    """
    
    def __init__(self, temperature: float = 0.5):
        """
        Initialize NT-Xent loss.
        
        Args:
            temperature: Temperature scaling factor.
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute NT-Xent loss.
        
        Args:
            z_i: Embeddings of first augmented view [batch, dim].
            z_j: Embeddings of second augmented view [batch, dim].
        
        Returns:
            Scalar loss value.
        """
        batch_size = z_i.size(0)
        device = z_i.device
        
        # Concatenate embeddings
        z = torch.cat([z_i, z_j], dim=0)  # [2*batch, dim]
        
        # Compute similarity matrix
        sim = torch.mm(z, z.t()) / self.temperature  # [2*batch, 2*batch]
        
        # Create mask for positive pairs
        # Positive pairs: (i, i+batch) and (i+batch, i)
        mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
        sim = sim.masked_fill(mask, float("-inf"))
        
        # Labels: positive pair indices
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=device),
            torch.arange(batch_size, device=device),
        ])
        
        # Cross entropy loss
        loss = F.cross_entropy(sim, labels)
        
        return loss


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.
    
    Extends contrastive learning to use label information,
    pulling together samples from the same class.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
    ):
        """
        Initialize SupCon loss.
        
        Args:
            temperature: Temperature for scaling.
            base_temperature: Base temperature for normalization.
        """
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.
        
        Args:
            features: Projected features [batch, n_views, dim] or [batch, dim].
            labels: Ground truth labels [batch].
        
        Returns:
            Scalar loss value.
        """
        device = features.device
        
        if features.dim() == 2:
            features = features.unsqueeze(1)
        
        batch_size = features.size(0)
        n_views = features.size(1)
        
        # Flatten views
        features = features.view(batch_size * n_views, -1)  # [batch*views, dim]
        labels = labels.repeat(n_views)  # [batch*views]
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity
        sim = torch.mm(features, features.t()) / self.temperature
        
        # Mask for same instance
        mask_self = torch.eye(batch_size * n_views, device=device, dtype=torch.bool)
        
        # Mask for same class (positive pairs)
        labels = labels.view(-1, 1)
        mask_pos = torch.eq(labels, labels.t()).float()
        mask_pos = mask_pos.masked_fill(mask_self, 0)
        
        # Compute loss
        exp_sim = torch.exp(sim)
        exp_sim = exp_sim.masked_fill(mask_self, 0)
        
        # Log-sum-exp for denominator
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        
        # Mean of positive pairs
        mask_pos_sum = mask_pos.sum(dim=1)
        mask_pos_sum = torch.clamp(mask_pos_sum, min=1)
        
        mean_log_prob = (mask_pos * log_prob).sum(dim=1) / mask_pos_sum
        
        # Loss
        loss = -mean_log_prob.mean() * (self.temperature / self.base_temperature)
        
        return loss


class ContrastiveAugmentation(nn.Module):
    """
    DNA sequence augmentations for contrastive learning.
    """
    
    def __init__(
        self,
        mutation_rate: float = 0.1,
        mask_rate: float = 0.15,
        crop_ratio: tuple[float, float] = (0.8, 1.0),
    ):
        """
        Initialize augmentations.
        
        Args:
            mutation_rate: Random mutation probability.
            mask_rate: Masking probability.
            crop_ratio: Random crop ratio range.
        """
        super().__init__()
        self.mutation_rate = mutation_rate
        self.mask_rate = mask_rate
        self.crop_ratio = crop_ratio
    
    def random_mutation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random mutations."""
        mask = torch.rand_like(x[:, 0:1, :]) < self.mutation_rate
        mask = mask.expand_as(x)
        
        # Random one-hot
        random_onehot = F.one_hot(
            torch.randint(0, 4, (x.size(0), x.size(2)), device=x.device),
            num_classes=4,
        ).permute(0, 2, 1).float()
        
        return torch.where(mask, random_onehot, x)
    
    def random_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Mask random positions."""
        mask = torch.rand(x.size(0), 1, x.size(2), device=x.device) < self.mask_rate
        return x * (~mask).float()
    
    def reverse_complement(self, x: torch.Tensor) -> torch.Tensor:
        """Apply reverse complement."""
        # Reverse sequence
        x = torch.flip(x, dims=[-1])
        # Complement: swap A<->T (0<->3), G<->C (1<->2)
        idx = torch.tensor([3, 2, 1, 0], device=x.device)
        x = x.index_select(dim=1, index=idx)
        return x
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate two augmented views.
        
        Args:
            x: Input tensor [batch, channels, length].
        
        Returns:
            Tuple of two augmented views.
        """
        # View 1: mutation + optional RC
        view1 = self.random_mutation(x)
        if torch.rand(1).item() > 0.5:
            view1 = self.reverse_complement(view1)
        
        # View 2: mask + mutation
        view2 = self.random_mask(x)
        view2 = self.random_mutation(view2)
        
        return view1, view2


class ContrastiveTrainer:
    """
    Trainer for contrastive learning.
    """
    
    def __init__(
        self,
        encoder: ContrastiveEncoder,
        optimizer: torch.optim.Optimizer,
        augmentation: ContrastiveAugmentation | None = None,
        temperature: float = 0.5,
        use_supervised: bool = False,
        device: torch.device | str = "cuda",
    ):
        """
        Initialize trainer.
        
        Args:
            encoder: Contrastive encoder model.
            optimizer: Optimizer.
            augmentation: Augmentation module.
            temperature: Temperature for loss.
            use_supervised: Whether to use supervised contrastive loss.
            device: Device to train on.
        """
        self.encoder = encoder.to(device)
        self.optimizer = optimizer
        self.augmentation = augmentation or ContrastiveAugmentation()
        self.device = device
        
        if use_supervised:
            self.criterion = SupConLoss(temperature=temperature)
        else:
            self.criterion = NTXentLoss(temperature=temperature)
        
        self.use_supervised = use_supervised
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.encoder.train()
        total_loss = 0.0
        
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(self.device)
                labels = batch[1].to(self.device) if len(batch) > 1 else None
            else:
                x = batch.to(self.device)
                labels = None
            
            # Generate augmented views
            view1, view2 = self.augmentation(x)
            
            # Get projections
            z1 = self.encoder(view1)
            z2 = self.encoder(view2)
            
            # Compute loss
            if self.use_supervised and labels is not None:
                features = torch.stack([z1, z2], dim=1)
                loss = self.criterion(features, labels)
            else:
                loss = self.criterion(z1, z2)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def get_embeddings(self, dataloader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract embeddings for downstream tasks."""
        self.encoder.eval()
        
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(self.device)
                    labels = batch[1] if len(batch) > 1 else None
                else:
                    x = batch.to(self.device)
                    labels = None
                
                embeddings = self.encoder.get_embeddings(x)
                all_embeddings.append(embeddings.cpu())
                
                if labels is not None:
                    all_labels.append(labels)
        
        embeddings = torch.cat(all_embeddings, dim=0)
        labels = torch.cat(all_labels, dim=0) if all_labels else None
        
        return embeddings, labels
