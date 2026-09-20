"""
Classify a contig - a sequence longer than the model's fixed fragment size - by sliding a window
across it and aggregating the per-window predictions, instead of truncating to the first window
(what `metapathpredict predict` did before this module existed, throwing away the rest of the contig).

Mean log-probability aggregation over independent windows was estimated earlier on this project by
sampling random same-genome fragment groups as a stand-in for a contig: 67.8% -> 80.9% fragment
accuracy at 8 windows, on the same species-disjoint test genomes (see BACKLOG.md, Q-2). This module
is the real version of that, run over an actual sliding window of one input sequence.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

BASES = {"A": 0, "C": 1, "G": 2, "T": 3}


def sliding_windows(length: int, window: int, step: int) -> list[tuple[int, int]]:
    """(start, end) pairs covering `length`, each `window` bases long and `step` bases apart. A
    trailing partial window (shorter than `window`) is dropped - it is not what the model was trained on."""
    if window <= 0 or step <= 0:
        raise ValueError(f"window and step must be positive, got window={window}, step={step}")
    if length < window:
        return []
    return [(s, s + window) for s in range(0, length - window + 1, step)]


def encode_window(sequence: str) -> np.ndarray:
    """One-hot [4, len(sequence)]; a base outside ACGT is left all-zero (matches how `prepare` encodes
    an unknown base, not the [0.25]*4 "uniform" convention some other parts of this codebase use)."""
    encoded = np.zeros((4, len(sequence)), dtype=np.float32)
    for i, base in enumerate(sequence.upper()):
        j = BASES.get(base)
        if j is not None:
            encoded[j, i] = 1.0
    return encoded


def classify_contig(probs_fn: Callable[[str], list[float] | np.ndarray], sequence: str, window: int,
                    step: int | None = None) -> dict:
    """
    Slide a `window`-base window (`step` bases apart, default: `window` itself, i.e. non-overlapping)
    across `sequence`, call `probs_fn` on each window's raw text to get its per-class probabilities,
    and aggregate by mean log-probability (equivalent to the geometric mean of the windows'
    probabilities, renormalised) - the same rule an ensemble's members are combined with
    (metapathpredict.baselines.EnsembleClassifier), just over positions instead of models.

    Returns {"num_windows", "predicted_class" (index), "probs" (aggregated, sums to 1), "windows":
    [{"start", "end", "probs"} for every window, for inspecting disagreement along the contig]}.
    Raises if the contig is shorter than one window - `metapathpredict predict` falls back to the
    single-fragment path in that case, it is not this function's job to pad.
    """
    spans = sliding_windows(len(sequence), window, step or window)
    if not spans:
        raise ValueError(f"sequence is {len(sequence)} bases, shorter than one window ({window})")
    per_window = [np.asarray(probs_fn(sequence[s:e]), dtype=np.float64) for s, e in spans]
    log_probs = np.log(np.clip(np.stack(per_window), 1e-12, 1.0))
    mean_log_prob = log_probs.mean(axis=0)
    probs = np.exp(mean_log_prob - mean_log_prob.max())
    probs /= probs.sum()
    return {
        "num_windows": len(spans),
        "predicted_class": int(mean_log_prob.argmax()),
        "probs": probs.tolist(),
        "windows": [{"start": s, "end": e, "probs": p.tolist()} for (s, e), p in zip(spans, per_window)],
    }
