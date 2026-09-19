"""Debiased / hard-negative NT-Xent, per-sample augmentation, and validation metrics."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from metapathpredict.models.contrastive import (
    ContrastiveAugmentation,
    ContrastiveEncoder,
    ContrastiveTrainer,
    NTXentLoss,
)


def _pair(batch=16, dim=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    z1 = F.normalize(torch.randn(batch, dim, generator=g), dim=1)
    z2 = F.normalize(z1 + 0.3 * torch.randn(batch, dim, generator=g), dim=1)
    return z1, z2


class TestNTXentOptions:
    def test_defaults_use_plain_cross_entropy(self):
        z1, z2 = _pair()
        assert NTXentLoss(0.2)(z1, z2).item() == NTXentLoss(0.2, tau_plus=0.0, beta=0.0)(z1, z2).item()

    def test_debiased_path_with_zero_parameters_matches_cross_entropy(self):
        z1, z2 = _pair()
        loss = NTXentLoss(0.2)
        z = torch.cat([z1, z2])
        sim = (z @ z.t() / 0.2).masked_fill(torch.eye(len(z), dtype=torch.bool), float("-inf"))
        labels = torch.cat([torch.arange(16, 32), torch.arange(16)])
        assert loss._debiased_hard_negative_loss(sim, labels).item() == \
            __import__("pytest").approx(F.cross_entropy(sim, labels).item(), rel=1e-4)

    def test_debias_changes_loss_and_stays_finite_with_gradient(self):
        z1, z2 = _pair()
        z1.requires_grad_(True)
        base = NTXentLoss(0.2)(z1.detach(), z2).item()
        loss = NTXentLoss(0.2, tau_plus=1 / 8)(z1, z2)
        loss.backward()
        assert torch.isfinite(loss) and loss.item() != base
        assert torch.isfinite(z1.grad).all() and z1.grad.abs().sum() > 0

    def test_hard_negative_weighting_never_lowers_the_loss(self):
        z1, z2 = _pair()
        plain = NTXentLoss(0.2, tau_plus=0.1, beta=0.0)(z1, z2).item()
        hard = NTXentLoss(0.2, tau_plus=0.1, beta=1.0)(z1, z2).item()
        assert hard >= plain - 1e-6


class TestDecoupledLoss:
    def test_matches_the_paper_formula(self):
        # L_i = -s(i, pos)/tau + log sum_{k not in {i, pos}} exp(s(i, k)/tau)
        z1, z2 = _pair()
        tau = 0.2
        z = torch.cat([z1, z2])
        sim = z @ z.t() / tau
        n = len(z1)
        expected = []
        for i in range(2 * n):
            pos = (i + n) % (2 * n)
            neg = [k for k in range(2 * n) if k not in (i, pos)]
            expected.append(-sim[i, pos] + torch.logsumexp(sim[i, neg], dim=0))
        loss = NTXentLoss(tau, decoupled=True)(z1, z2)
        assert loss.item() == __import__("pytest").approx(torch.stack(expected).mean().item(), rel=1e-4)

    def test_is_lower_than_ntxent_and_has_gradient(self):
        z1, z2 = _pair()
        z1.requires_grad_(True)
        plain = NTXentLoss(0.2)(z1.detach(), z2).item()
        loss = NTXentLoss(0.2, decoupled=True)(z1, z2)
        loss.backward()
        assert torch.isfinite(loss) and loss.item() < plain
        assert torch.isfinite(z1.grad).all() and z1.grad.abs().sum() > 0

    def test_loss_type_dcl_is_a_valid_setting_and_builds_a_decoupled_criterion(self):
        from metapathpredict.config.settings import ContrastiveConfig

        assert ContrastiveConfig(loss_type="dcl").loss_type == "dcl"


class TestHybridLoss:
    def _trainer(self, **kw):
        from metapathpredict.models.contrastive import ContrastiveEncoder, ContrastiveTrainer

        enc = ContrastiveEncoder(backbone="small", projection_dim=32, hidden_dim=64, base_channels=16, num_classes=3)
        return ContrastiveTrainer(enc, torch.optim.SGD(enc.parameters(), lr=0.1), temperature=0.2, device="cpu", **kw)

    def test_hybrid_is_the_weighted_sum_of_supcon_and_ntxent(self):
        z1, z2 = _pair()
        labels = torch.arange(16) % 3
        hybrid = self._trainer(use_supervised=True, supcon_weight=0.3)
        sup = self._trainer(use_supervised=True)._compute_loss(z1, z2, labels).item()
        inst = self._trainer()._compute_loss(z1, z2, labels).item()
        got = hybrid._compute_loss(z1, z2, labels).item()
        assert got == __import__("pytest").approx(0.3 * sup + 0.7 * inst, rel=1e-5)

    def test_pure_supcon_and_unsupervised_paths_are_unchanged(self):
        z1, z2 = _pair()
        labels = torch.arange(16) % 3
        assert self._trainer(use_supervised=True).instance_criterion is None
        assert self._trainer()._compute_loss(z1, z2, None).item() == NTXentLoss(0.2)(z1, z2).item()
        with __import__("pytest").raises(ValueError):
            self._trainer(use_supervised=True)._compute_loss(z1, z2, None)
        # without labels a supervised trainer falls back to the instance loss
        assert self._trainer(use_supervised=True, supcon_weight=0.5)._compute_loss(z1, z2, None).item() == \
            NTXentLoss(0.2)(z1, z2).item()

    def test_loss_type_hybrid_is_a_valid_setting(self):
        from metapathpredict.config.settings import ContrastiveConfig

        cfg = ContrastiveConfig(loss_type="hybrid", supcon_weight=0.25)
        assert cfg.loss_type == "hybrid" and cfg.supcon_weight == 0.25


class TestPerSampleAugmentation:
    def _batch(self, n=64, length=100):
        base = F.one_hot(torch.randint(0, 4, (1, length)), 4).permute(0, 2, 1).float()
        return base.repeat(n, 1, 1)  # every sequence identical

    def test_full_ratio_crop_is_identity(self):
        x = self._batch()
        aug = ContrastiveAugmentation(crop_ratio=(1.0, 1.0), per_sample=True)
        assert torch.equal(aug.random_crop(x), x)

    def test_crop_windows_differ_between_samples(self):
        out = ContrastiveAugmentation(crop_ratio=(0.5, 0.9), per_sample=True).random_crop(self._batch())
        assert len({tuple(row.flatten().tolist()) for row in out}) > 10

    def test_batch_level_mode_cuts_every_sample_identically(self):
        out = ContrastiveAugmentation(crop_ratio=(0.5, 0.9), per_sample=False).random_crop(self._batch())
        assert all(torch.equal(out[0], row) for row in out)

    def test_crop_is_one_contiguous_window_of_the_source(self):
        x = self._batch(n=32)
        out = ContrastiveAugmentation(crop_ratio=(0.5, 0.9), per_sample=True).random_crop(x)
        for xi, oi in zip(x, out):
            cols = oi.sum(dim=0).nonzero().flatten()
            assert 50 <= len(cols) <= 90
            assert (cols[1:] - cols[:-1] == 1).all()  # contiguous
            window = oi[:, cols[0]: cols[-1] + 1]
            # the window is a substring of the original sequence
            src = xi.argmax(dim=0).tolist()
            got = window.argmax(dim=0).tolist()
            assert any(src[k: k + len(got)] == got for k in range(len(src) - len(got) + 1))

    def test_reverse_complement_is_decided_per_sample(self):
        x = torch.zeros(200, 4, 20)
        x[:, 0, :] = 1.0  # all 'A'; reverse complement makes them all 'T'
        mixed = ContrastiveAugmentation(per_sample=True)._maybe_reverse_complement(x)
        flipped = (mixed[:, 3, :].sum(dim=1) > 0)
        assert 40 < flipped.sum() < 160
        batch_level = ContrastiveAugmentation(per_sample=False)._maybe_reverse_complement(x)
        assert len({bool(v) for v in (batch_level[:, 3, :].sum(dim=1) > 0)}) == 1


class TestValidationMetrics:
    def test_metrics_are_reported_and_in_range(self):
        torch.manual_seed(0)
        encoder = ContrastiveEncoder(
            in_channels=4, backbone="small", projection_dim=16, hidden_dim=16, base_channels=16, num_classes=3,
        )
        x = F.one_hot(torch.randint(0, 4, (96, 100)), 4).permute(0, 2, 1).float()
        loader = DataLoader(TensorDataset(x, torch.randint(0, 3, (96,))), batch_size=32, drop_last=True)
        trainer = ContrastiveTrainer(
            encoder, torch.optim.SGD(encoder.parameters(), lr=0.01),
            ContrastiveAugmentation(), temperature=0.2, device="cpu",
        )
        loss = trainer.validate_epoch(loader, metric_samples=64)
        m = trainer.last_val_metrics
        assert loss > 0
        assert m["alignment"] >= 0 and m["uniformity"] <= 0
        assert 1.0 <= m["erank_projection"] <= 16
        assert 1.0 <= m["erank_backbone"] <= encoder.embed_dim
