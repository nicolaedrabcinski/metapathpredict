"""Frozen-backbone linear probe: BatchNorm recalibration and head fitting."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from metapathpredict.config import Settings
from metapathpredict.probe import cache_embeddings, fit_linear_probe, recalibrate_batchnorm


class TinyBackbone(nn.Module):
    """Stand-in with the two things the probe needs: get_embeddings and a classifier head."""

    def __init__(self, in_dim=8, feat_dim=16, classes=3):
        super().__init__()
        self.linear = nn.Linear(in_dim, feat_dim)
        self.bn = nn.BatchNorm1d(feat_dim)
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(feat_dim, feat_dim // 2), nn.ReLU(),
                                        nn.Dropout(0.1), nn.Linear(feat_dim // 2, classes))

    def get_embeddings(self, x):
        return self.bn(self.linear(x))


def _clusters(n=600, in_dim=8, classes=3, seed=0):
    """Same class centres for every call; only the noise depends on `seed`."""
    centers = torch.randn(classes, in_dim, generator=torch.Generator().manual_seed(123)) * 3
    y = torch.arange(n) % classes
    noise = torch.randn(n, in_dim, generator=torch.Generator().manual_seed(seed))
    return TensorDataset(centers[y] + 0.3 * noise, y)


def test_recalibration_matches_clean_data_statistics_and_leaves_eval_mode():
    torch.manual_seed(0)
    net = TinyBackbone()
    net.bn.running_mean.fill_(100.0)  # statistics from some other (augmented) distribution
    ds = _clusters()
    used = recalibrate_batchnorm(net, DataLoader(ds, batch_size=50), "cpu", max_batches=100)
    assert used == 12
    with torch.no_grad():
        clean_mean = net.linear(ds.tensors[0]).mean(dim=0)
    assert torch.allclose(net.bn.running_mean, clean_mean, atol=0.05)
    assert net.bn.momentum == 0.1 and not net.training


def test_cached_embeddings_use_eval_statistics_and_keep_labels():
    torch.manual_seed(0)
    net = TinyBackbone()
    ds = _clusters(n=100)
    net.train()
    feats, labels = cache_embeddings(net, DataLoader(ds, batch_size=32), "cpu")
    assert not net.training and feats.shape == (100, 16) and torch.equal(labels, ds.tensors[1])


def test_probe_fits_separable_data_without_touching_backbone_parameters():
    torch.manual_seed(0)
    net = TinyBackbone()
    before = {k: v.clone() for k, v in net.named_parameters() if not k.startswith("classifier")}
    train, val = _clusters(600, seed=1), _clusters(300, seed=2)
    result = fit_linear_probe(
        net, DataLoader(train, batch_size=64), DataLoader(val, batch_size=64), "cpu",
        epochs=60, patience=10, recalibrate_batches=10,
    )
    assert result["best_val_acc"] > 0.95 and 1 <= result["best_epoch"] <= result["epochs_run"]
    for k, v in net.named_parameters():
        if not k.startswith("classifier"):
            assert torch.equal(v, before[k]), k
    assert not net.training


def test_probe_stops_early_when_validation_stops_improving():
    torch.manual_seed(0)
    net = TinyBackbone()
    # labels unrelated to the inputs: validation accuracy cannot keep improving
    x = torch.randn(300, 8)
    noise = TensorDataset(x, torch.randint(0, 3, (300,)))
    result = fit_linear_probe(
        net, DataLoader(noise, batch_size=64), DataLoader(noise, batch_size=64), "cpu",
        epochs=200, patience=5, recalibrate_batches=5,
    )
    assert result["epochs_run"] < 200


def test_default_probe_mode_stays_legacy():
    # Flipped only after the runs that were started with the legacy probe have finished.
    assert Settings().contrastive.probe_mode == "legacy"
