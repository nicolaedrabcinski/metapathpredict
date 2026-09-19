"""
Stacking: a small classifier trained on the predictions of several models.

The base models (CNNs, k-mer boosting) are fit on the training genomes. A meta-classifier is then fit on the
*validation* split, whose families were never seen in training, so it learns how far to trust each base model
on organisms that are new to them - which averaging cannot do. VirHunter (Sukhorukov et al. 2022) stacks three
CNNs with a deliberately weak random forest (depth 5, 10 trees) trained on a set where hard fragments are
over-represented; both ideas are here (`kind="forest"`, `balance_hard=True`).

Scoring the meta-classifier on a split whose genomes and families it has not seen (the test split) is what
tells whether it helps.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

EPS = 1e-4


def stack_features(probabilities: list[np.ndarray]) -> np.ndarray:
    """[models][n, classes] probabilities -> [n, models * classes] log-probabilities."""
    return np.concatenate([np.log(np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0)) for p in probabilities], axis=1)


def average_probabilities(probabilities: list[np.ndarray]) -> np.ndarray:
    return np.mean([np.asarray(p, dtype=np.float64) for p in probabilities], axis=0)


def hard_balance_weights(probabilities: list[np.ndarray], y: np.ndarray, threshold: float = 0.8) -> np.ndarray:
    """
    Sample weights giving the well-predicted fragments (mean probability of the true class >= threshold) and the
    poorly predicted ones the same total weight, so the meta-classifier is not dominated by the easy majority.
    """
    true_prob = average_probabilities(probabilities)[np.arange(len(y)), y]
    easy = true_prob >= threshold
    weights = np.ones(len(y))
    if easy.any() and (~easy).any():
        weights[easy] = 0.5 / easy.sum()
        weights[~easy] = 0.5 / (~easy).sum()
        weights *= len(y)
    return weights


def fit_meta(probabilities: list[np.ndarray], y: np.ndarray, kind: str = "logreg", balance_hard: bool = False,
             C: float = 0.1, seed: int = 0):
    """
    Fit the meta-classifier on `probabilities` (one [n, classes] array per base model) and the labels `y`.
    kind: "logreg" (multinomial logistic regression on log-probabilities, strongly regularised by `C`) or
    "forest" (random forest, depth 5, 10 trees, as in VirHunter).
    """
    features = stack_features(probabilities)
    weights = hard_balance_weights(probabilities, y) if balance_hard else None
    if kind == "logreg":
        model = LogisticRegression(C=C, max_iter=500)
    elif kind == "forest":
        model = RandomForestClassifier(n_estimators=10, max_depth=5, max_features="sqrt", max_samples=0.2, random_state=seed, n_jobs=4)
    else:
        raise ValueError(f"kind must be logreg or forest, got {kind!r}")
    return model.fit(features, y, sample_weight=weights)


def predict_meta(model, probabilities: list[np.ndarray], num_classes: int) -> np.ndarray:
    """Class probabilities [n, num_classes] of a fitted meta-classifier (classes missing from its training data get 0)."""
    proba = model.predict_proba(stack_features(probabilities))
    out = np.zeros((len(proba), num_classes))
    out[:, model.classes_] = proba
    return out
