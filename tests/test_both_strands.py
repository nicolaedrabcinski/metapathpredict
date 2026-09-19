"""Reverse-complement averaging at prediction time."""

import pytest
import torch

from metapathpredict.cli import _predict_single
from metapathpredict.models.contrastive import reverse_complement


def _x(n: int = 1, length: int = 30) -> torch.Tensor:
    return torch.nn.functional.one_hot(torch.randint(0, 4, (n, length)), 4).permute(0, 2, 1).float()


class _Strand2Way:
    """Stub with a strand-dependent head: reads the mean of channel A over the first half."""

    def __init__(self):
        self.encoder = self._logits

    @staticmethod
    def _logits(x):
        a = x[:, 0, : x.shape[-1] // 2].mean(dim=1)
        return torch.stack([a * 5, -a * 5, torch.zeros_like(a)], dim=1)


def test_reverse_complement_is_an_involution_and_swaps_bases():
    x = _x(4)
    assert torch.equal(reverse_complement(reverse_complement(x)), x)
    a_first = torch.zeros(1, 4, 3)
    a_first[0, 0, 0] = 1  # A at position 0, C/G/T elsewhere left as zeros
    rc = reverse_complement(a_first)
    assert rc[0, 3, 2] == 1  # becomes T at the last position


def test_both_strands_averages_the_two_probability_vectors():
    x = _x()
    model = _Strand2Way()
    _, _, forward = _predict_single(model, "contrastive", x)
    _, _, reverse = _predict_single(model, "contrastive", reverse_complement(x))
    _, conf, both = _predict_single(model, "contrastive", x, both_strands=True)
    expected = [(f + r) / 2 for f, r in zip(forward, reverse)]
    assert both == pytest.approx(expected, abs=1e-6)
    assert conf == pytest.approx(max(expected), abs=1e-6)
    assert sum(both) == pytest.approx(1.0, abs=1e-5)


def test_default_is_the_single_strand_prediction():
    x = _x()
    model = _Strand2Way()
    assert _predict_single(model, "contrastive", x)[2] == _predict_single(model, "contrastive", x, both_strands=False)[2]
