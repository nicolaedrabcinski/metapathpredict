"""
Contrastive Learning for DNA sequence representation.

Implements SimCLR-style contrastive learning for learning
robust sequence embeddings.
"""

from __future__ import annotations

import logging
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base import BaseModel
from .configurable_cnn import ConfigurableCNN

logger = logging.getLogger(__name__)


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

        num_params = sum(p.numel() for p in self.parameters())
        logger.debug(
            f"ProjectionHead created: {in_dim} -> {hidden_dim} -> {hidden_dim} -> {out_dim} "
            f"({num_params:,} params)"
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
        num_classes: int = 3,
    ):
        """
        Initialize contrastive encoder.
        
        Args:
            in_channels: Input channels (4 for DNA).
            backbone: CNN backbone preset.
            projection_dim: Projection head output dimension.
            hidden_dim: Projection head hidden dimension.
            base_channels: Base channels for backbone.
            num_classes: Outputs of the backbone's classifier head (fit as a linear
                probe after pretraining; the contrastive loss itself never uses it).
        """
        super().__init__()
        self.num_classes = num_classes

        logger.info(
            f"Building ContrastiveEncoder: backbone={backbone}, "
            f"in_channels={in_channels}, base_channels={base_channels}, "
            f"projection_dim={projection_dim}, hidden_dim={hidden_dim}"
        )

        # Backbone encoder (without classifier)
        self.encoder = ConfigurableCNN(
            in_channels=in_channels,
            num_classes=num_classes,
            kernel_preset=backbone,
            base_channels=base_channels,
        )

        # Get embedding dimension
        embed_dim = self.encoder._final_channels
        logger.info(f"  CNN backbone: embed_dim={embed_dim}")

        # Projection head
        self.projection = ProjectionHead(
            in_dim=embed_dim,
            hidden_dim=hidden_dim,
            out_dim=projection_dim,
        )

        self.embed_dim = embed_dim
        self.projection_dim = projection_dim

        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        proj_params = sum(p.numel() for p in self.projection.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"  Params: encoder={encoder_params:,}, projection={proj_params:,}, "
            f"total={total_params:,}"
        )
    
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

    def __init__(self, temperature: float = 0.5, tau_plus: float = 0.0, beta: float = 0.0):
        """
        Initialize NT-Xent loss.

        Args:
            temperature: Temperature scaling factor.
            tau_plus: Class prior for debiasing (Chuang et al. 2020): the chance that a
                random "negative" actually belongs to the anchor's class. 0 disables it.
            beta: Hard-negative concentration (Robinson et al. 2021): negatives are
                reweighted by exp(beta * similarity). 0 disables it; beta=0 with
                tau_plus>0 is the plain debiased loss.
        """
        super().__init__()
        self.temperature = temperature
        self.tau_plus = tau_plus
        self.beta = beta
        self._call_count = 0
        logger.info(
            f"NTXentLoss initialized: temperature={temperature}, tau_plus={tau_plus}, beta={beta}"
        )
    
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
        
        if self.tau_plus > 0 or self.beta > 0:
            loss = self._debiased_hard_negative_loss(sim, labels)
        else:
            loss = F.cross_entropy(sim, labels)

        self._call_count += 1
        if self._call_count == 1:
            # Log detailed diagnostics on the very first batch
            with torch.no_grad():
                pos_sim = torch.sum(z_i * z_j, dim=1)  # cosine sim of positive pairs
                logger.info(
                    f"  [NTXent first batch] batch_size={batch_size}, "
                    f"sim_matrix=[{2*batch_size}x{2*batch_size}], "
                    f"pos_cosine_sim: mean={pos_sim.mean():.4f}, "
                    f"min={pos_sim.min():.4f}, max={pos_sim.max():.4f}, "
                    f"loss={loss.item():.4f}"
                )

        return loss


    def _debiased_hard_negative_loss(self, sim: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Debiased (tau_plus) and/or hard-negative-weighted (beta) contrastive loss, following the
        reference implementations of Chuang et al. and Robinson et al.

        `sim` is the (2B, 2B) similarity matrix already divided by the temperature, with -inf on
        the diagonal; `labels[i]` is the index of anchor i's positive.
        """
        n_total = sim.size(0)
        n_neg = n_total - 2  # everything except the anchor itself and its positive
        exp_sim = torch.exp(sim)  # diagonal becomes 0
        pos = exp_sim.gather(1, labels.unsqueeze(1)).squeeze(1)

        neg = exp_sim.clone()
        neg.scatter_(1, labels.unsqueeze(1), 0.0)

        if self.beta > 0:
            # weights proportional to exp(beta * sim) on true negatives, normalised to mean 1
            weights = torch.where(neg > 0, neg.clamp_min(1e-30) ** self.beta, torch.zeros_like(neg))
            neg_sum = (weights * neg).sum(dim=1) / (weights.sum(dim=1) / n_neg)
        else:
            neg_sum = neg.sum(dim=1)

        # estimate of the true-negative term, kept above its theoretical minimum
        est = (neg_sum - self.tau_plus * n_neg * pos) / (1.0 - self.tau_plus)
        est = est.clamp_min(n_neg * math.exp(-1.0 / self.temperature))
        return (-torch.log(pos / (pos + est))).mean()


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
        self._call_count = 0
        logger.info(
            f"SupConLoss initialized: temperature={temperature}, "
            f"base_temperature={base_temperature}"
        )
    
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

        self._call_count += 1
        if self._call_count == 1:
            # Detailed diagnostics on the first batch
            unique_labels = torch.unique(labels.squeeze())
            avg_pos_pairs = mask_pos.sum(dim=1).mean().item()
            sim_diag = sim.diag()
            logger.info(
                f"  [SupCon first batch] batch_size={batch_size}, n_views={n_views}, "
                f"classes={len(unique_labels)}, "
                f"avg_pos_pairs={avg_pos_pairs:.1f}, "
                f"sim: mean={sim.mean():.4f}, max={sim.max():.4f}, "
                f"loss={loss.item():.4f}"
            )

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
        per_sample: bool = True,
    ):
        """
        Initialize augmentations.
        
        Args:
            mutation_rate: Random mutation probability.
            mask_rate: Masking probability.
            crop_ratio: Random crop ratio range.
            per_sample: Draw the crop window and the reverse-complement decision
                independently for every sequence. False reproduces the earlier
                behaviour where one crop and one decision were drawn per batch,
                so every sequence in the batch was cut at the same place.
        """
        super().__init__()
        self.mutation_rate = mutation_rate
        self.mask_rate = mask_rate
        self.crop_ratio = crop_ratio
        self.per_sample = per_sample
        self._call_count = 0
        logger.info(
            f"ContrastiveAugmentation: mutation_rate={mutation_rate}, "
            f"mask_rate={mask_rate}, crop_ratio={crop_ratio}"
        )
    
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
        # View 1: crop -> mutation -> optional reverse complement
        view1 = self.random_crop(x)
        view1 = self.random_mutation(view1)
        view1 = self._maybe_reverse_complement(view1)

        # View 2: crop -> mask -> mutation
        view2 = self.random_crop(x)
        view2 = self.random_mask(view2)
        view2 = self.random_mutation(view2)

        # Логирование (оставить как есть)
        self._call_count += 1
        if self._call_count == 1:
            diff1 = (view1 - x).abs().sum().item() / x.numel()
            diff2 = (view2 - x).abs().sum().item() / x.numel()
            masked_frac = (view2.sum(dim=1) == 0).float().mean().item()
            logger.info(
                f"  [Augmentation first batch] input={list(x.shape)}, "
                f"view1_diff={diff1:.4f}, view2_diff={diff2:.4f}, "
                f"view2_masked_frac={masked_frac:.4f}"
            )

        return view1, view2

    def _maybe_reverse_complement(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse-complement with probability 0.5: per sequence, or once for the whole batch."""
        if not self.per_sample:
            return self.reverse_complement(x) if torch.rand(1).item() > 0.5 else x
        flip = torch.rand(x.size(0), 1, 1, device=x.device) > 0.5
        return torch.where(flip, self.reverse_complement(x), x)

    def _random_crop_per_sample(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sequence crop: each sequence gets its own window length, source and destination."""
        batch_size, _, seq_len = x.shape
        lo, hi = self.crop_ratio
        crop_len = (seq_len * (lo + (hi - lo) * torch.rand(batch_size, device=x.device))).long()
        crop_len = crop_len.clamp(1, seq_len)
        slack = seq_len - crop_len
        src = (torch.rand(batch_size, device=x.device) * (slack + 1)).long().clamp_max(slack)
        dst = (torch.rand(batch_size, device=x.device) * (slack + 1)).long().clamp_max(slack)

        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, L)
        offset = pos - dst.unsqueeze(1)  # position inside the window
        valid = (offset >= 0) & (offset < crop_len.unsqueeze(1))
        index = (src.unsqueeze(1) + offset).clamp(0, seq_len - 1)  # (B, L)
        gathered = torch.gather(x, 2, index.unsqueeze(1).expand(-1, x.size(1), -1))
        return gathered * valid.unsqueeze(1).to(x.dtype)

    def random_crop(self, x: torch.Tensor) -> torch.Tensor:
        """Случайное вырезание подпоследовательности с дополнением нулями."""
        if self.per_sample:
            return self._random_crop_per_sample(x)
        batch_size, channels, seq_len = x.shape
        # Случайная длина обрезки в пределах crop_ratio
        crop_len = int(seq_len * torch.empty(1).uniform_(*self.crop_ratio).item())
        # Случайная позиция начала окна
        start = torch.randint(0, seq_len - crop_len + 1, (1,)).item()
        # Вырезаем
        cropped = x[:, :, start:start + crop_len]
        # Дополняем нулями до исходной длины (случайное расположение окна внутри)
        pad_left = torch.randint(0, seq_len - crop_len + 1, (1,)).item()
        pad_right = seq_len - crop_len - pad_left
        padded = F.pad(cropped, (pad_left, pad_right), mode='constant', value=0)
        return padded

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
        tau_plus: float = 0.0,
        beta: float = 0.0,
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
            tau_plus: Debiasing class prior for NT-Xent (ignored with SupCon).
            beta: Hard-negative concentration for NT-Xent (ignored with SupCon).
        """
        self.encoder = encoder.to(device)
        self.optimizer = optimizer
        self.augmentation = augmentation or ContrastiveAugmentation()
        self.device = device

        if use_supervised:
            self.criterion = SupConLoss(temperature=temperature)
        else:
            self.criterion = NTXentLoss(temperature=temperature, tau_plus=tau_plus, beta=beta)

        self.use_supervised = use_supervised
        self._epoch_count = 0

        loss_name = "SupConLoss" if use_supervised else "NTXentLoss"
        logger.info(
            f"ContrastiveTrainer initialized: loss={loss_name}, "
            f"temperature={temperature}, device={device}"
        )
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self._epoch_count += 1
        self.encoder.train()
        total_loss = 0.0
        num_batches = len(dataloader)

        # Track stats across epoch
        all_grad_norms = []
        all_losses = []
        all_pos_sims = []
        all_neg_sims = []
        all_emb_stds = []
        min_loss = float("inf")
        max_loss = float("-inf")

        # Current LR
        current_lr = self.optimizer.param_groups[0]["lr"]
        logger.info(
            f"  Epoch {self._epoch_count} start: lr={current_lr:.2e}, "
            f"{num_batches} batches"
        )

        t0 = time.time()
        pbar = tqdm(
            enumerate(dataloader),
            total=num_batches,
            desc=f"Contrastive Epoch {self._epoch_count}",
            unit="batch",
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}] "
                "{postfix}"
            ),
        )

        for batch_idx, batch in pbar:
            t_batch = time.time()

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

            # Compute similarity stats (before loss, for monitoring)
            with torch.no_grad():
                # Positive pair cosine similarity
                pos_sim = F.cosine_similarity(z1, z2, dim=1)
                pos_sim_mean = pos_sim.mean().item()
                all_pos_sims.append(pos_sim_mean)

                # Negative pair similarity (mean of off-diagonal)
                sim_matrix = torch.mm(z1, z2.t())
                eye_mask = ~torch.eye(z1.size(0), device=z1.device, dtype=torch.bool)
                neg_sim = sim_matrix[eye_mask].mean().item()
                all_neg_sims.append(neg_sim)

                # Embedding std (collapse detection: std -> 0 means collapse)
                emb_std = z1.std(dim=0).mean().item()
                all_emb_stds.append(emb_std)

            # Compute loss
            if self.use_supervised and labels is not None:
                features = torch.stack([z1, z2], dim=1)
                loss = self.criterion(features, labels)
            else:
                loss = self.criterion(z1, z2)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient norm for monitoring
            grad_norm = 0.0
            max_grad = 0.0
            for p in self.encoder.parameters():
                if p.grad is not None:
                    pnorm = p.grad.data.norm(2).item()
                    grad_norm += pnorm ** 2
                    max_grad = max(max_grad, p.grad.data.abs().max().item())
            grad_norm = grad_norm ** 0.5
            all_grad_norms.append(grad_norm)

            self.optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss
            all_losses.append(batch_loss)
            min_loss = min(min_loss, batch_loss)
            max_loss = max(max_loss, batch_loss)

            # Update tqdm postfix
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix(
                loss=f"{batch_loss:.4f}",
                avg=f"{avg_loss:.4f}",
                pos=f"{pos_sim_mean:.3f}",
                neg=f"{neg_sim:.3f}",
                grad=f"{grad_norm:.2f}",
                std=f"{emb_std:.4f}",
            )

        pbar.close()

        # Epoch summary
        avg_loss = total_loss / num_batches
        epoch_time = time.time() - t0
        throughput = num_batches * x.size(0) / epoch_time

        avg_grad = sum(all_grad_norms) / len(all_grad_norms)
        avg_pos_sim = sum(all_pos_sims) / len(all_pos_sims)
        avg_neg_sim = sum(all_neg_sims) / len(all_neg_sims)
        avg_emb_std = sum(all_emb_stds) / len(all_emb_stds)

        logger.info(
            f"  Epoch {self._epoch_count} summary: "
            f"avg_loss={avg_loss:.6f} (min={min_loss:.4f}, max={max_loss:.4f})"
        )
        logger.info(
            f"    Similarity: pos={avg_pos_sim:.4f}, neg={avg_neg_sim:.4f}, "
            f"gap={avg_pos_sim - avg_neg_sim:.4f}"
        )
        logger.info(
            f"    Gradients: avg_norm={avg_grad:.4f}, "
            f"max_norm={max(all_grad_norms):.4f}"
        )
        logger.info(
            f"    Embedding std={avg_emb_std:.4f} "
            f"{'(WARNING: possible collapse!)' if avg_emb_std < 0.01 else '(healthy)'}"
        )
        logger.info(
            f"    Throughput: {throughput:.0f} samples/s, {epoch_time:.1f}s total"
        )

        self.last_train_stats = {
            "pos_sim": avg_pos_sim,
            "neg_sim": avg_neg_sim,
            "sim_gap": avg_pos_sim - avg_neg_sim,
            "grad_norm": avg_grad,
            "embedding_std": avg_emb_std,
            "samples_per_sec": throughput,
        }
        return avg_loss

    def validate_epoch(self, dataloader: DataLoader, metric_samples: int = 4096) -> float:
        """
        Compute the same contrastive loss on held-out data, no gradient/optimizer
        step. Lets training pick the checkpoint that generalizes instead of the
        one with the lowest train loss, and gives a train-vs-val curve to spot
        overfitting (val loss flattening or rising while train loss keeps falling).

        Also fills `last_val_metrics` from the first `metric_samples` validation samples:
        alignment (mean squared distance between the two views of a sample), uniformity
        (Wang & Isola: log mean exp(-2 d^2) over pairs) and the effective rank of the
        projection and of the backbone embedding. Unlike the NT-Xent value these do not
        depend on the batch size, and the rank shows dimensional collapse that a healthy
        per-dimension std can hide.
        """
        self.encoder.eval()
        total_loss = 0.0
        num_batches = len(dataloader)
        z1_all, z2_all, h_all = [], [], []
        collected = 0

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(self.device)
                    labels = batch[1].to(self.device) if len(batch) > 1 else None
                else:
                    x = batch.to(self.device)
                    labels = None

                view1, view2 = self.augmentation(x)
                z1 = self.encoder(view1)
                z2 = self.encoder(view2)

                if self.use_supervised and labels is not None:
                    features = torch.stack([z1, z2], dim=1)
                    loss = self.criterion(features, labels)
                else:
                    loss = self.criterion(z1, z2)

                total_loss += loss.item()

                if collected < metric_samples:
                    z1_all.append(z1)
                    z2_all.append(z2)
                    h_all.append(self.encoder.get_embeddings(view1))
                    collected += z1.size(0)

        self.last_val_metrics = self._representation_metrics(
            torch.cat(z1_all)[:metric_samples],
            torch.cat(z2_all)[:metric_samples],
            torch.cat(h_all)[:metric_samples],
        )
        self.encoder.train()
        return total_loss / num_batches

    @staticmethod
    def _effective_rank(x: torch.Tensor) -> float:
        """exp(entropy of the normalised singular values) of the centred matrix (Roy & Vetterli)."""
        s = torch.linalg.svdvals((x - x.mean(dim=0, keepdim=True)).float())
        p = s / s.sum().clamp_min(1e-12)
        return torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()).item()

    @classmethod
    def _representation_metrics(cls, z1: torch.Tensor, z2: torch.Tensor, h: torch.Tensor) -> dict[str, float]:
        alignment = (z1 - z2).pow(2).sum(dim=1).mean().item()
        uniformity = torch.log(torch.exp(-2.0 * torch.pdist(z1.float()).pow(2)).mean()).item()
        return {
            "alignment": alignment,
            "uniformity": uniformity,
            "erank_projection": cls._effective_rank(z1),
            "erank_backbone": cls._effective_rank(h),
        }

    def get_embeddings(self, dataloader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract embeddings for downstream tasks."""
        self.encoder.eval()
        logger.info("Extracting embeddings...")

        all_embeddings = []
        all_labels = []
        num_batches = len(dataloader)

        t0 = time.time()
        with torch.no_grad():
            for batch in tqdm(
                dataloader,
                total=num_batches,
                desc="Extracting embeddings",
                unit="batch",
            ):
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

        elapsed = time.time() - t0
        logger.info(
            f"Embeddings extracted: {embeddings.shape[0]} samples, "
            f"dim={embeddings.shape[1]}, {elapsed:.1f}s"
        )

        return embeddings, labels
