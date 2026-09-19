"""
Explaining a DNA CNN classifier: attributions, Grad-CAM, checks that the attributions mean something,
and questions about what the network uses at all.

Everything works on one-hot batches [batch, 4, length] (channels A, C, G, T; an unknown base is all
zeros, as in the training data) and on models mapping them to class logits, such as ConfigurableCNN.

A heat map of a classifier that averages local motifs over the whole fragment is easy to draw and easy
to over-read, so the module also has the tests that say whether a method deserves trust:

* `deletion_curve` - remove the positions a method calls important; the class probability must drop
  faster than when random positions are removed.
* `randomization_check` - attributions of the trained model must differ from those of the same
  architecture with random weights.
* `shuffle_accuracy` - accuracy on fragments shuffled while keeping the base (or dinucleotide)
  composition says how much of the prediction the composition alone explains.
"""

from __future__ import annotations

import copy
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from metapathpredict.models.base import ConvBlock

BASES = "ACGT"


def _target(logits: torch.Tensor, target: torch.Tensor | int | None) -> torch.Tensor:
    if target is None:
        return logits.argmax(dim=1)
    if isinstance(target, int):
        return torch.full((logits.size(0),), target, dtype=torch.long, device=logits.device)
    return target.to(logits.device)


def _selected_logit(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return logits.gather(1, target[:, None]).sum()


def position_scores(attribution: torch.Tensor) -> torch.Tensor:
    """[batch, 4, length] attribution -> [batch, length]: what each position contributed as a whole."""
    return attribution.sum(dim=1)


# --------------------------------------------------------------------------------------- attributions
def gradient_x_input(model: nn.Module, x: torch.Tensor, target=None) -> torch.Tensor:
    """Gradient of the target logit with respect to the input, times the input. [batch, 4, length]."""
    model.eval()
    with torch.enable_grad():
        x = x.detach().requires_grad_(True)
        logits = model(x)
        (grad,) = torch.autograd.grad(_selected_logit(logits, _target(logits, target)), x)
    return (grad * x).detach()


def integrated_gradients(model: nn.Module, x: torch.Tensor, target=None, baseline: str | torch.Tensor = "zeros",
                         steps: int = 32) -> torch.Tensor:
    """
    Integrated Gradients (Sundararajan et al. 2017) with the midpoint rule. `baseline` is "zeros" (an
    all-N sequence, what the network saw as missing data), "uniform" (0.25 everywhere) or a tensor.
    Attributions sum to logit(x) - logit(baseline), see `completeness_gap`. [batch, 4, length].
    """
    model.eval()
    if isinstance(baseline, str):
        if baseline not in ("zeros", "uniform"):
            raise ValueError(f"baseline must be zeros, uniform or a tensor, got {baseline!r}")
        base = torch.full_like(x, 0.25) if baseline == "uniform" else torch.zeros_like(x)
    else:
        base = baseline.to(x)
    with torch.no_grad():
        chosen = _target(model(x), target)
    total = torch.zeros_like(x)
    for step in range(steps):
        alpha = (step + 0.5) / steps
        with torch.enable_grad():
            point = (base + alpha * (x - base)).detach().requires_grad_(True)
            logits = model(point)
            (grad,) = torch.autograd.grad(_selected_logit(logits, chosen), point)
        total += grad
    return ((x - base) * total / steps).detach()


def completeness_gap(model: nn.Module, x: torch.Tensor, attribution: torch.Tensor, baseline="zeros", target=None) -> float:
    """Relative gap between sum(attribution) and logit(x) - logit(baseline); ~0 for Integrated Gradients."""
    model.eval()
    base = torch.full_like(x, 0.25) if isinstance(baseline, str) and baseline == "uniform" else (
        torch.zeros_like(x) if isinstance(baseline, str) else baseline.to(x))
    with torch.no_grad():
        logits, base_logits = model(x), model(base)
        chosen = _target(logits, target)
        expected = logits.gather(1, chosen[:, None])[:, 0] - base_logits.gather(1, chosen[:, None])[:, 0]
    got = attribution.sum(dim=(1, 2))
    return float(((got - expected).abs() / expected.abs().clamp_min(1e-6)).mean())


def last_conv_block(model: nn.Module) -> nn.Module:
    blocks = [m for m in model.modules() if isinstance(m, ConvBlock)]
    if not blocks:
        raise ValueError("the model has no ConvBlock to explain; pass `layer` explicitly")
    return blocks[-1]


def grad_cam(model: nn.Module, x: torch.Tensor, target=None, layer: nn.Module | None = None) -> torch.Tensor:
    """
    Grad-CAM (Selvaraju et al. 2017) for a 1-D CNN: the activation maps of `layer` (default: the last
    ConvBlock), weighted by the mean gradient of the target logit over positions, summed over channels,
    passed through a ReLU, stretched to the input length and scaled to a maximum of 1. [batch, length].

    The resolution is that of the layer (a third of the sequence is one feature-map cell after three
    poolings of 2), so this is a coarse localisation, not a per-base attribution.
    """
    model.eval()
    layer = layer if layer is not None else last_conv_block(model)
    store: dict[str, torch.Tensor] = {}
    handle = layer.register_forward_hook(lambda module, inputs, output: store.__setitem__("activation", output))
    try:
        with torch.enable_grad():
            logits = model(x.detach().requires_grad_(True))
            activation = store["activation"]
            (grad,) = torch.autograd.grad(_selected_logit(logits, _target(logits, target)), activation)
    finally:
        handle.remove()
    weights = grad.mean(dim=-1, keepdim=True)
    cam = F.relu((weights * activation).sum(dim=1, keepdim=True)).detach()
    cam = F.interpolate(cam, size=x.size(-1), mode="linear", align_corners=False)[:, 0]
    return cam / cam.amax(dim=1, keepdim=True).clamp_min(1e-12)


# ------------------------------------------------------------------------------------ trust checks
def deletion_curve(model: nn.Module, x: torch.Tensor, scores: torch.Tensor, fractions, target=None) -> np.ndarray:
    """
    Mean probability of the target class after removing (setting to N) the highest-scoring fraction of
    positions in every sequence. `target` defaults to the class predicted on the intact sequence.
    A method that finds the positions the prediction relies on makes this fall faster than random scores.
    """
    model.eval()
    with torch.no_grad():
        chosen = _target(model(x), target)
        order = scores.argsort(dim=1, descending=True)
        curve = []
        for fraction in fractions:
            removed = torch.zeros_like(scores, dtype=torch.bool)
            removed.scatter_(1, order[:, : int(round(fraction * x.size(-1)))], True)
            probs = F.softmax(model(x.masked_fill(removed[:, None, :], 0.0)), dim=1)
            curve.append(probs.gather(1, chosen[:, None]).mean().item())
    return np.array(curve)


def area_under(curve: np.ndarray, fractions) -> float:
    """Trapezoid area of a deletion curve; smaller means the removed positions mattered more."""
    fractions = np.asarray(fractions, dtype=float)
    return float(np.sum((curve[1:] + curve[:-1]) / 2 * np.diff(fractions)))


def randomized_copy(model: nn.Module, seed: int = 0) -> nn.Module:
    """The same architecture with re-initialised weights (and reset BatchNorm statistics)."""
    clone = copy.deepcopy(model)
    torch.manual_seed(seed)
    for module in clone.modules():
        if hasattr(module, "reset_parameters") and module is not clone:
            module.reset_parameters()
    return clone.eval()


def _ranks(a: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(float)


def randomization_check(model: nn.Module, x: torch.Tensor, method: Callable[[nn.Module, torch.Tensor], torch.Tensor],
                        seed: int = 0) -> float:
    """
    Mean Spearman correlation between the position scores of `method` for the trained model and for a
    randomly initialised one (Adebayo et al. 2018). Near 0 is good: a high value means the maps are
    driven by the input, not by what the model learned.
    """
    trained = method(model, x).detach().cpu().numpy()
    random_ = method(randomized_copy(model, seed), x).detach().cpu().numpy()
    corr = [np.corrcoef(_ranks(t), _ranks(r))[0, 1] for t, r in zip(trained, random_)]
    return float(np.nanmean(corr))


# --------------------------------------------------------------------------- what does the model use
def to_indices(x: np.ndarray) -> np.ndarray:
    """One-hot [n, 4, length] -> base indices [n, length]; an all-zero column (N) becomes 4."""
    idx = x.argmax(axis=1).astype(np.int8)
    idx[x.sum(axis=1) == 0] = 4
    return idx


def to_onehot(idx: np.ndarray) -> np.ndarray:
    out = np.zeros((idx.shape[0], 4, idx.shape[1]), dtype=np.float32)
    for base in range(4):
        out[:, base][idx == base] = 1.0
    return out


def dinucleotide_shuffle(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Random sequence with exactly the same dinucleotide counts, first and last base as `seq`
    (Altschul & Erikson 1985: a random Eulerian path through the dinucleotide graph).
    """
    seq = [int(s) for s in seq]
    if len(seq) < 4:
        return np.array(seq, dtype=np.int8)
    last = seq[-1]
    edges: dict[int, list[int]] = {}
    for a, b in zip(seq[:-1], seq[1:]):
        edges.setdefault(a, []).append(b)
    while True:  # pick each vertex's final exit so that following them from anywhere reaches `last`
        exits = {v: edges[v][int(rng.integers(len(edges[v])))] for v in edges if v != last}
        reaches = True
        for start in exits:
            seen, node = set(), start
            while node != last:
                if node in seen or node not in exits:
                    reaches = False
                    break
                seen.add(node)
                node = exits[node]
            if not reaches:
                break
        if reaches:
            break
    shuffled = {}
    for vertex, outs in edges.items():
        outs = list(outs)
        if vertex != last:
            outs.remove(exits[vertex])
        rng.shuffle(outs)
        shuffled[vertex] = outs + ([exits[vertex]] if vertex != last else [])
    out, cursor, node = [seq[0]], {v: 0 for v in shuffled}, seq[0]
    for _ in range(len(seq) - 1):
        node_next = shuffled[node][cursor[node]]
        cursor[node] += 1
        out.append(node_next)
        node = node_next
    return np.array(out, dtype=np.int8)


def shuffle_sequences(x: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    """One-hot [n, 4, length] with every sequence shuffled: none | mono (any order) | dinuc (keeps dinucleotides)."""
    if mode == "none":
        return x
    idx = to_indices(x)
    if mode == "mono":
        shuffled = np.stack([rng.permutation(row) for row in idx])
    elif mode == "dinuc":
        shuffled = np.stack([dinucleotide_shuffle(row, rng) for row in idx])
    else:
        raise ValueError(f"mode must be none, mono or dinuc, got {mode!r}")
    return to_onehot(shuffled)


def predict(model: nn.Module, x: np.ndarray, device="cpu", batch_size: int = 256) -> np.ndarray:
    """Predicted class of every one-hot sequence in `x` [n, 4, length]."""
    model.eval().to(device)
    out = []
    with torch.no_grad():
        for lo in range(0, len(x), batch_size):
            out.append(model(torch.from_numpy(x[lo:lo + batch_size]).to(device)).argmax(dim=1).cpu())
    return torch.cat(out).numpy()


def shuffle_accuracy(model: nn.Module, x: np.ndarray, y: np.ndarray, mode: str, seed: int = 0, device="cpu") -> dict:
    """Accuracy overall and per class on the sequences shuffled as `mode` says."""
    preds = predict(model, shuffle_sequences(x, mode, np.random.default_rng(seed)), device)
    return {"accuracy": float((preds == y).mean()),
            "per_class": {int(c): float((preds[y == c] == c).mean()) for c in np.unique(y)}}


def first_layer_motifs(model: nn.Module, x: np.ndarray, y: np.ndarray, num_classes: int, top_windows: int = 200,
                       device="cpu", batch_size: int = 128) -> dict:
    """
    Motifs of the first convolution. For every filter the window that excites it most is taken from every
    sequence; the `top_windows` strongest of those give a position weight matrix. Returns
    {"pwm": [F, 4, k], "information": [F] (bits), "class_mean": [classes, F] (mean best activation per
    class), "specificity": [F], "best_class": [F]}. Specificity is an effect size: the mean best activation
    in the class that excites the filter most, minus the mean over the other classes, in units of the
    standard deviation of the best activation over all sequences.
    """
    conv = next(m for m in model.modules() if isinstance(m, ConvBlock)).conv
    width, pad = conv.kernel_size[0], conv.padding[0]
    conv = conv.eval().to(device)
    best_value, best_window = [], []
    with torch.no_grad():
        for lo in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[lo:lo + batch_size]).to(device)
            activation = F.conv1d(batch, conv.weight, padding=pad)                 # [b, F, positions]
            windows = F.pad(batch, (pad, pad)).unfold(2, width, 1)                  # [b, 4, positions, k]
            value, where = activation.max(dim=2)                                    # [b, F]
            index = where[:, None, :, None].expand(-1, 4, -1, width)               # [b, 4, F, k]
            best_value.append(value.cpu())
            best_window.append(windows.gather(2, index).permute(0, 2, 1, 3).cpu())  # [b, F, 4, k]
    value, window = torch.cat(best_value).numpy(), torch.cat(best_window).numpy()  # [n, F], [n, F, 4, k]
    n, filters = value.shape
    pwm = np.zeros((filters, 4, width), dtype=np.float32)
    for f in range(filters):
        top = np.argsort(-value[:, f])[:top_windows]
        pwm[f] = (window[top, f].mean(axis=0) + 1e-3)
        pwm[f] /= pwm[f].sum(axis=0, keepdims=True)
    information = (2 + (pwm * np.log2(pwm)).sum(axis=1)).sum(axis=1)
    class_mean = np.stack([value[y == c].mean(axis=0) if (y == c).any() else np.zeros(filters) for c in range(num_classes)])
    others = (class_mean.sum(axis=0) - class_mean.max(axis=0)) / max(num_classes - 1, 1)
    specificity = (class_mean.max(axis=0) - others) / value.std(axis=0).clip(1e-6)
    return {"pwm": pwm, "information": information, "class_mean": class_mean,
            "specificity": specificity, "best_class": class_mean.argmax(axis=0)}
