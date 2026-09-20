"""
Post-hoc calibration and selective prediction (abstention) for a trained classifier.

A classifier's own softmax probability is not necessarily its true confidence: a network is often
overconfident, especially on inputs unlike its training data (novelty_distance.py: genomes with no
close relative in training score ~33-40% while the model's own top probability there is usually much
higher). Two independent fixes:

* Temperature scaling (Guo et al. 2017): a single scalar T fit on held-out logits, applied as
  softmax(logits / T). This does not change which class wins (accuracy is unchanged), only how
  trustworthy the winning probability looks - which the next step depends on.
* Selective prediction / abstention: refuse to answer below a confidence threshold. The
  risk-coverage curve says what accuracy you get for what fraction of inputs answered.

Both work on whatever a model's `forward` returns treated as logits - including
metapathpredict.baselines.EnsembleClassifier, whose forward already returns log(mean probability):
softmax(log(p) / T) is temperature scaling of an ensemble's averaged probabilities, not a special case.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def collect_logits(model, loader, device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Raw model outputs (pre-softmax) and targets, over a whole loader. Model stays in eval mode."""
    model.eval()
    logits, targets = [], []
    with torch.no_grad():
        for batch in loader:
            logits.append(model(batch[0].to(device)).cpu())
            targets.append(batch[1])
    return torch.cat(logits), torch.cat(targets)


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor, lr: float = 0.05, max_iter: int = 100) -> float:
    """
    The scalar T > 0 minimizing cross-entropy of softmax(logits / T) against `targets` (Guo et al.
    2017, via L-BFGS on log T so T stays positive). T > 1 softens an overconfident model, T < 1
    sharpens an underconfident one; T = 1 is a no-op. Fit on validation data, not on the test data
    being reported on, and not on training data (a model is overconfident on its own training set by
    construction).
    """
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())


def calibrated_probabilities(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """softmax(logits / temperature); temperature=1.0 is plain softmax."""
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    return F.softmax(logits / temperature, dim=1)


def expected_calibration_error(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> float:
    """
    ECE (Guo et al. 2017): examples are bucketed by their top predicted probability into `n_bins`
    equal-width bins; ECE is the size-weighted mean gap between each bin's accuracy and its mean
    confidence. 0 is perfectly calibrated; there is no upper bound in general, but with 2+ classes it
    cannot exceed 1.
    """
    confidences, predictions = probs.max(dim=1)
    correct = (predictions == targets).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        if in_bin.any():
            ece += in_bin.float().mean().item() * abs(correct[in_bin].mean().item() - confidences[in_bin].mean().item())
    return ece


def risk_coverage_curve(probs: torch.Tensor, targets: torch.Tensor, thresholds=None) -> dict:
    """
    Selective prediction: for each confidence threshold, the fraction of examples whose top
    probability clears it ("coverage") and the accuracy among those ("accuracy"; 1 - risk). A
    threshold with no examples above it gets accuracy NaN. `thresholds` defaults to 30 points from 0
    to the 99th percentile of the observed confidences (so the curve does not end in mostly-empty bins).
    """
    confidences, predictions = probs.max(dim=1)
    confidences_np = confidences.numpy()
    correct = (predictions == targets).numpy()
    if thresholds is None:
        thresholds = np.linspace(0.0, float(np.quantile(confidences_np, 0.99)), 30)
    coverage, accuracy = [], []
    for t in thresholds:
        keep = confidences_np >= t
        coverage.append(float(keep.mean()))
        accuracy.append(float(correct[keep].mean()) if keep.any() else float("nan"))
    return {"threshold": [float(t) for t in thresholds], "coverage": coverage, "accuracy": accuracy}


def genome_risk_coverage(probs: torch.Tensor, targets: torch.Tensor, genomes: np.ndarray, thresholds=None) -> dict:
    """
    Like risk_coverage_curve, but a genome only counts once it has at least one fragment above the
    threshold (majority vote of the kept fragments), and coverage/accuracy are over genomes, not
    fragments - the level results are actually used at (novelty_distance.py, prophage_check.py).
    """
    confidences, predictions = probs.max(dim=1)
    confidences_np, predictions_np = confidences.numpy(), predictions.numpy()
    targets_np = targets.numpy()
    unique = np.unique(genomes)
    genome_true = {g: targets_np[genomes == g][0] for g in unique}
    if thresholds is None:
        thresholds = np.linspace(0.0, float(np.quantile(confidences_np, 0.99)), 30)
    coverage, accuracy = [], []
    for t in thresholds:
        kept_correct, kept_total = 0, 0
        for g in unique:
            sel = (genomes == g) & (confidences_np >= t)
            if not sel.any():
                continue
            votes = np.bincount(predictions_np[sel])
            kept_total += 1
            kept_correct += int(votes.argmax() == genome_true[g])
        coverage.append(kept_total / len(unique))
        accuracy.append(kept_correct / kept_total if kept_total else float("nan"))
    return {"threshold": [float(t) for t in thresholds], "coverage": coverage, "accuracy": accuracy}
