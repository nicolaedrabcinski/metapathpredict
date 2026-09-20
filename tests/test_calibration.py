"""Temperature scaling and selective prediction (abstention)."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from metapathpredict import calibration


def _overconfident_logits(n=2000, seed=0, true_accuracy=0.7, scale=8.0):
    """Logits that pick the right class `true_accuracy` of the time but always with high confidence."""
    rng = np.random.default_rng(seed)
    targets = rng.integers(0, 3, size=n)
    correct = rng.random(n) < true_accuracy
    predicted = np.where(correct, targets, (targets + rng.integers(1, 3, size=n)) % 3)
    logits = np.full((n, 3), -scale, dtype=np.float32)
    logits[np.arange(n), predicted] = scale
    return torch.from_numpy(logits), torch.from_numpy(targets).long()


def test_collect_logits_matches_targets_and_a_manual_pass():
    torch.manual_seed(0)
    x = torch.randn(20, 4, 50)
    y = torch.randint(0, 3, (20,))
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(200, 3)).eval()
    loader = DataLoader(TensorDataset(x, y), batch_size=7)
    logits, targets = calibration.collect_logits(model, loader)
    with torch.no_grad():
        expected = model(x)
    assert torch.allclose(logits, expected, atol=1e-5) and torch.equal(targets, y)


def test_temperature_scaling_does_not_change_accuracy_but_softens_overconfidence():
    logits, targets = _overconfident_logits()
    raw = calibration.calibrated_probabilities(logits, 1.0)
    t = calibration.fit_temperature(logits, targets)
    calibrated = calibration.calibrated_probabilities(logits, t)
    assert t > 1.0  # this model is overconfident (70% right, ~100% confident), T should soften it
    assert torch.equal(raw.argmax(dim=1), calibrated.argmax(dim=1))  # same predictions
    assert calibrated.max(dim=1).values.mean() < raw.max(dim=1).values.mean()  # less extreme confidence
    assert calibration.expected_calibration_error(calibrated, targets) < calibration.expected_calibration_error(raw, targets)


def test_a_well_calibrated_model_gets_temperature_close_to_one():
    rng = np.random.default_rng(1)
    n, targets = 4000, None
    targets = torch.from_numpy(rng.integers(0, 2, size=n)).long()
    # logits whose softmax already equals the true accuracy, on average, at each confidence level
    p_correct = torch.from_numpy(rng.uniform(0.5, 0.95, size=n)).float()
    correct = torch.from_numpy(rng.random(n)) < p_correct
    predicted = torch.where(correct, targets, 1 - targets)
    logits = torch.zeros(n, 2)
    odds = torch.log(p_correct / (1 - p_correct))
    logits[torch.arange(n), predicted] = odds
    t = calibration.fit_temperature(logits, targets)
    assert 0.85 < t < 1.15


def test_calibrated_probabilities_reject_non_positive_temperature():
    logits, _ = _overconfident_logits(n=4)
    with pytest.raises(ValueError):
        calibration.calibrated_probabilities(logits, 0.0)


def test_expected_calibration_error_is_zero_for_a_perfectly_calibrated_set():
    # every example predicted with probability exactly equal to the bucket's true accuracy
    probs = torch.tensor([[0.9, 0.1]] * 90 + [[0.1, 0.9]] * 10)
    targets = torch.tensor([0] * 81 + [1] * 9 + [0] * 1 + [1] * 9)
    assert calibration.expected_calibration_error(probs, targets, n_bins=2) == pytest.approx(0.0, abs=1e-6)


def test_risk_coverage_curve_is_monotonic_coverage_and_higher_threshold_never_hurts_accuracy():
    logits, targets = _overconfident_logits(true_accuracy=0.6, scale=2.0)
    probs = calibration.calibrated_probabilities(logits, 1.0)
    curve = calibration.risk_coverage_curve(probs, targets, thresholds=np.linspace(0.3, 0.9, 7))
    assert curve["coverage"] == sorted(curve["coverage"], reverse=True)  # coverage falls as threshold rises
    valid = [a for a in curve["accuracy"] if not np.isnan(a)]
    assert valid[-1] >= valid[0] - 0.05  # the most selective end should not be worse than the least


def test_risk_coverage_threshold_above_every_confidence_gives_zero_coverage():
    logits, targets = _overconfident_logits(n=50)
    probs = calibration.calibrated_probabilities(logits, 1.0)
    curve = calibration.risk_coverage_curve(probs, targets, thresholds=[2.0])
    assert curve["coverage"] == [0.0] and np.isnan(curve["accuracy"][0])


def test_genome_risk_coverage_votes_per_genome_not_per_fragment():
    # two genomes, 10 fragments each; genome A: 6 confident-correct + 4 low-confidence-wrong,
    # genome B: 10 fragments, split 5/5 with the wrong side more confident (majority still swings it)
    probs = torch.tensor(
        [[0.9, 0.1]] * 6 + [[0.55, 0.45]] * 4 +      # genome A: majority correct, all kept at t=0.5
        [[0.6, 0.4]] * 5 + [[0.95, 0.05]] * 5          # genome B: majority (10/10 predict 0) wrong for B's true=1
    )
    targets = torch.tensor([0] * 10 + [1] * 10)
    genomes = np.array(["A"] * 10 + ["B"] * 10)
    curve = calibration.genome_risk_coverage(probs, targets, genomes, thresholds=[0.5])
    assert curve["coverage"] == [1.0]  # both genomes have at least one fragment >= 0.5
    assert curve["accuracy"] == [0.5]  # A correct, B wrong: 1 of 2 genomes


def test_genome_risk_coverage_drops_a_genome_with_no_confident_fragments():
    probs = torch.tensor([[0.9, 0.1]] * 5 + [[0.6, 0.4]] * 5)
    targets = torch.tensor([0] * 5 + [0] * 5)
    genomes = np.array(["confident"] * 5 + ["unsure"] * 5)
    curve = calibration.genome_risk_coverage(probs, targets, genomes, thresholds=[0.8])
    assert curve["coverage"] == [0.5] and curve["accuracy"] == [1.0]  # only "confident" survives, and it's right
