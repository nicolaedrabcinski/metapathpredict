"""
Unit tests for contrastive learning module.
"""

import pytest
import torch
import torch.nn as nn

from metapathpredict.models.contrastive import (
    ContrastiveEncoder,
    ProjectionHead,
    NTXentLoss,
    SupConLoss,
    ContrastiveAugmentation,
)


class TestProjectionHead:
    """Tests for ProjectionHead."""

    def test_output_shape(self):
        """Test projection head output shape."""
        head = ProjectionHead(
            in_dim=256,
            hidden_dim=256,
            out_dim=128,
        )
        x = torch.randn(8, 256)
        
        out = head(x)
        
        assert out.shape == (8, 128)

    def test_normalization(self):
        """Test output can be L2 normalized."""
        head = ProjectionHead(in_dim=256, out_dim=128)
        x = torch.randn(8, 256)
        
        out = head(x)
        out_normalized = nn.functional.normalize(out, dim=1)
        
        # L2 norm of each vector should be 1
        norms = torch.norm(out_normalized, dim=1)
        torch.testing.assert_close(norms, torch.ones(8), rtol=1e-5, atol=1e-5)

    def test_gradient_flow(self):
        """Test gradients flow through projection head."""
        head = ProjectionHead(in_dim=256, out_dim=128)
        x = torch.randn(8, 256, requires_grad=True)
        
        out = head(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None


class TestContrastiveEncoder:
    """Tests for ContrastiveEncoder."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        encoder = ContrastiveEncoder(
            in_channels=4,
            backbone="medium",
            projection_dim=128,
            hidden_dim=256,
        )
        x = torch.randn(8, 4, 1000)
        
        projections = encoder(x)
        
        assert projections.shape == (8, 128)

    def test_embedding_only(self):
        """Test getting only embeddings."""
        encoder = ContrastiveEncoder(
            in_channels=4,
            projection_dim=128,
        )
        x = torch.randn(8, 4, 1000)
        
        embeddings = encoder.get_embeddings(x)
        
        # Embedding dim depends on backbone
        assert embeddings.shape[0] == 8
        assert embeddings.dim() == 2

    def test_embeddings_normalized(self):
        """Test that projections are L2 normalized."""
        encoder = ContrastiveEncoder()
        x = torch.randn(8, 4, 500)
        
        projections = encoder(x)
        
        norms = torch.norm(projections, dim=1)
        torch.testing.assert_close(norms, torch.ones(8), rtol=1e-5, atol=1e-5)


class TestNTXentLoss:
    """Tests for NT-Xent (Normalized Temperature-scaled Cross Entropy) loss."""

    def test_basic_loss_computation(self):
        """Test basic loss computation."""
        loss_fn = NTXentLoss(temperature=0.5)
        
        # Simulated projections from two augmented views
        z_i = torch.randn(8, 128)
        z_j = torch.randn(8, 128)
        
        # Normalize
        z_i = nn.functional.normalize(z_i, dim=1)
        z_j = nn.functional.normalize(z_j, dim=1)
        
        loss = loss_fn(z_i, z_j)
        
        assert loss.ndim == 0  # Scalar
        assert loss.item() > 0  # Loss should be positive

    def test_loss_with_same_views(self):
        """Test loss when views are identical (should be low)."""
        loss_fn = NTXentLoss(temperature=0.5)
        
        z = torch.randn(8, 128)
        z = nn.functional.normalize(z, dim=1)
        
        # Same view should result in lower loss
        loss_same = loss_fn(z, z)
        
        # Different random views
        z2 = torch.randn(8, 128)
        z2 = nn.functional.normalize(z2, dim=1)
        loss_diff = loss_fn(z, z2)
        
        # Same views should have lower loss
        assert loss_same < loss_diff

    def test_temperature_effect(self):
        """Test that temperature affects loss magnitude."""
        z_i = torch.randn(8, 128)
        z_j = torch.randn(8, 128)
        z_i = nn.functional.normalize(z_i, dim=1)
        z_j = nn.functional.normalize(z_j, dim=1)
        
        loss_high_temp = NTXentLoss(temperature=1.0)(z_i, z_j)
        loss_low_temp = NTXentLoss(temperature=0.1)(z_i, z_j)
        
        # Lower temperature should generally give higher loss
        # (more confident, sharper distribution)
        assert loss_low_temp != loss_high_temp

    def test_gradient_flow(self):
        """Test gradients flow through loss."""
        loss_fn = NTXentLoss(temperature=0.5)
        
        z_i = torch.randn(8, 128, requires_grad=True)
        z_j = torch.randn(8, 128, requires_grad=True)
        
        loss = loss_fn(
            nn.functional.normalize(z_i, dim=1),
            nn.functional.normalize(z_j, dim=1),
        )
        loss.backward()
        
        assert z_i.grad is not None
        assert z_j.grad is not None


class TestSupConLoss:
    """Tests for Supervised Contrastive Loss."""

    def test_basic_loss_computation(self):
        """Test basic supervised contrastive loss."""
        loss_fn = SupConLoss(temperature=0.5)
        
        features = torch.randn(16, 128)
        features = nn.functional.normalize(features, dim=1)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 1, 2, 0])
        
        loss = loss_fn(features, labels)
        
        assert loss.ndim == 0  # Scalar
        assert loss.item() > 0  # Loss should be positive

    def test_loss_decreases_with_clustering(self):
        """Test that loss is lower when same-class samples are similar."""
        loss_fn = SupConLoss(temperature=0.5)
        
        # Well-clustered features (same class = similar)
        features_good = torch.zeros(8, 128)
        features_good[:4, :64] = 1  # Class 0 features
        features_good[4:, 64:] = 1  # Class 1 features
        features_good = nn.functional.normalize(features_good, dim=1)
        
        # Random features
        features_bad = torch.randn(8, 128)
        features_bad = nn.functional.normalize(features_bad, dim=1)
        
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        
        loss_good = loss_fn(features_good, labels)
        loss_bad = loss_fn(features_bad, labels)
        
        # Well-clustered should have lower loss
        assert loss_good < loss_bad

    def test_single_class(self):
        """Test loss with single class."""
        loss_fn = SupConLoss(temperature=0.5)
        
        features = torch.randn(8, 128)
        features = nn.functional.normalize(features, dim=1)
        labels = torch.zeros(8, dtype=torch.long)  # All same class
        
        loss = loss_fn(features, labels)
        
        assert loss.ndim == 0
        assert not torch.isnan(loss)


class TestContrastiveAugmentation:
    """Tests for DNA sequence augmentation."""

    def test_reverse_complement(self):
        """Test reverse complement augmentation method."""
        aug = ContrastiveAugmentation(
            mutation_rate=0.0,
            mask_rate=0.0,
        )
        
        # Simple one-hot encoded sequence: ACGT
        x = torch.tensor([
            [1, 0, 0, 0],  # A
            [0, 1, 0, 0],  # C
            [0, 0, 1, 0],  # G
            [0, 0, 0, 1],  # T
        ], dtype=torch.float32).T.unsqueeze(0)  # Shape: (1, 4, 4)
        
        # Call reverse_complement directly
        rc = aug.reverse_complement(x)
        
        assert rc.shape == x.shape

    def test_random_mutation(self):
        """Test random mutation augmentation."""
        aug = ContrastiveAugmentation(
            mutation_rate=0.5,  # High rate for testing
            mask_rate=0.0,
        )
        
        x = torch.randn(8, 4, 100)
        
        # Run multiple times to ensure mutations happen
        different_count = 0
        for _ in range(10):
            augmented = aug.random_mutation(x)
            if not torch.allclose(augmented, x):
                different_count += 1
        
        # At least some augmentations should differ
        assert different_count > 0

    def test_random_mask(self):
        """Test random masking augmentation."""
        aug = ContrastiveAugmentation(
            mutation_rate=0.0,
            mask_rate=0.5,  # High rate for testing
        )
        
        x = torch.ones(8, 4, 100)  # All ones
        
        augmented = aug.random_mask(x)
        
        # Some positions should be zeroed out
        assert augmented.sum() < x.sum()

    def test_no_augmentation(self):
        """Test with all augmentations disabled."""
        aug = ContrastiveAugmentation(
            mutation_rate=0.0,
            mask_rate=0.0,
        )
        
        x = torch.randn(8, 4, 100)
        # Forward returns tuple of two views
        view1, view2 = aug(x)
        
        # With zero mutation rate, view1 should be close to x or RC of x
        assert view1.shape == x.shape
        assert view2.shape == x.shape

    def test_batch_processing(self):
        """Test augmentation processes batches correctly."""
        aug = ContrastiveAugmentation()
        
        x = torch.randn(32, 4, 1000)
        view1, view2 = aug(x)
        
        assert view1.shape == x.shape
        assert view2.shape == x.shape


class TestContrastiveTrainingFlow:
    """Integration tests for contrastive training flow."""

    def test_full_forward_pass(self):
        """Test complete forward pass with loss computation."""
        encoder = ContrastiveEncoder(
            in_channels=4,
            backbone="small",
            projection_dim=128,
        )
        augmentation = ContrastiveAugmentation()
        loss_fn = NTXentLoss(temperature=0.5)
        
        # Original batch
        x = torch.randn(8, 4, 500)
        
        # Create two augmented views (forward returns tuple)
        x_i, x_j = augmentation(x)
        
        # Forward pass - encoder returns projections directly
        z_i = encoder(x_i)
        z_j = encoder(x_j)
        
        # Compute loss
        loss = loss_fn(z_i, z_j)
        
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_training_step(self):
        """Test complete training step with backward pass."""
        encoder = ContrastiveEncoder(
            in_channels=4,
            backbone="small",
            projection_dim=64,
        )
        optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
        augmentation = ContrastiveAugmentation()
        loss_fn = NTXentLoss(temperature=0.5)
        
        x = torch.randn(8, 4, 500)
        
        # Training step
        encoder.train()
        optimizer.zero_grad()
        
        x_i, x_j = augmentation(x)
        
        z_i = encoder(x_i)
        z_j = encoder(x_j)
        
        loss = loss_fn(z_i, z_j)
        loss.backward()
        optimizer.step()
        
        # Should complete without error
        assert True

    def test_supervised_training_step(self):
        """Test supervised contrastive training step."""
        encoder = ContrastiveEncoder(
            in_channels=4,
            backbone="small",
            projection_dim=64,
        )
        optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
        loss_fn = SupConLoss(temperature=0.5)
        
        x = torch.randn(16, 4, 500)
        labels = torch.randint(0, 3, (16,))
        
        encoder.train()
        optimizer.zero_grad()
        
        projections = encoder(x)
        loss = loss_fn(projections, labels)
        loss.backward()
        optimizer.step()
        
        assert True


class TestEmbeddingQuality:
    """Tests for embedding quality."""

    def test_embeddings_vary_with_input(self):
        """Test that different inputs produce different embeddings."""
        encoder = ContrastiveEncoder()
        encoder.eval()
        
        # Create very different inputs
        x1 = torch.zeros(4, 4, 500)
        x1[:, 0, :] = 1  # All A's
        
        x2 = torch.zeros(4, 4, 500)
        x2[:, 3, :] = 1  # All T's
        
        with torch.no_grad():
            emb1 = encoder.get_embeddings(x1)
            emb2 = encoder.get_embeddings(x2)
        
        # Very different inputs should give different embeddings
        # Use higher tolerance or check that at least some are different
        diff = (emb1 - emb2).abs().mean()
        assert diff > 0.001  # Some difference expected

    def test_similar_inputs_similar_embeddings(self):
        """Test that similar inputs produce similar embeddings."""
        encoder = ContrastiveEncoder()
        encoder.eval()
        
        x = torch.randn(4, 4, 500)
        x_noisy = x + 0.01 * torch.randn_like(x)  # Slight noise
        
        with torch.no_grad():
            emb1 = encoder.get_embeddings(x)
            emb2 = encoder.get_embeddings(x_noisy)
        
        # Similar inputs should have similar embeddings
        similarity = nn.functional.cosine_similarity(emb1, emb2, dim=1)
        assert similarity.mean() > 0.5  # Should be reasonably similar
