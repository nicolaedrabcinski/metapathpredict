"""Meta-classifier over the predictions of several models."""

import numpy as np
import pytest

from metapathpredict import stacking

C = 3


def _base(y, accuracy, rng):
    """A model that puts most of its probability on the right class with the given accuracy."""
    n = len(y)
    right = rng.random(n) < accuracy
    guess = np.where(right, y, (y + rng.integers(1, C, n)) % C)
    p = np.full((n, C), 0.1)
    p[np.arange(n), guess] = 0.8
    return p / p.sum(axis=1, keepdims=True)


def test_features_are_log_probabilities_of_all_models_side_by_side():
    a = np.array([[0.7, 0.2, 0.1]])
    b = np.array([[0.1, 0.1, 0.8]])
    f = stacking.stack_features([a, b])
    assert f.shape == (1, 6) and np.allclose(f[0, :3], np.log(a[0])) and np.allclose(f[0, 3:], np.log(b[0]))
    assert np.isfinite(stacking.stack_features([np.array([[1.0, 0.0, 0.0]])])).all()  # zeros are clipped


def test_average_probabilities():
    p = stacking.average_probabilities([np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])])
    assert p.tolist() == [[0.5, 0.5]]


def test_hard_balance_gives_easy_and_hard_fragments_equal_total_weight():
    y = np.array([0, 0, 0, 0, 1])
    p = np.array([[0.9, 0.1]] * 3 + [[0.2, 0.8]] + [[0.1, 0.9]])   # first 3 easy, 4th wrong, last easy
    w = stacking.hard_balance_weights([p], y)
    easy = p[np.arange(5), y] >= 0.8
    assert np.isclose(w[easy].sum(), w[~easy].sum()) and np.isclose(w.sum(), 5)
    assert np.all(stacking.hard_balance_weights([np.array([[0.9, 0.1]] * 2)], np.array([0, 0])) == 1)  # nothing hard: no reweighting


@pytest.mark.parametrize("kind", ["logreg", "forest"])
@pytest.mark.parametrize("balance", [False, True])
def test_a_meta_classifier_learns_which_base_model_to_trust(kind, balance):
    rng = np.random.default_rng(0)
    y_val, y_test = rng.integers(0, C, 3000), rng.integers(0, C, 3000)
    good = lambda y: _base(y, 0.9, rng)
    noise = lambda y: _base(y, 0.34, rng)                       # chance level
    meta = stacking.fit_meta([good(y_val), noise(y_val), noise(y_val)], y_val, kind=kind, balance_hard=balance)
    proba = stacking.predict_meta(meta, [good(y_test), noise(y_test), noise(y_test)], C)
    averaged = stacking.average_probabilities([good(y_test), noise(y_test), noise(y_test)])
    assert proba.shape == (3000, C) and np.allclose(proba.sum(axis=1), 1.0)
    assert (proba.argmax(1) == y_test).mean() > 0.85            # it found the informative model
    assert (proba.argmax(1) == y_test).mean() > (averaged.argmax(1) == y_test).mean()  # averaging is dragged down by noise


def test_classes_missing_from_the_meta_training_data_get_probability_zero():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)                                  # class 2 never occurs
    p = _base(y, 0.9, rng)[:, :2]
    p = np.concatenate([p, np.full((400, 1), 0.05)], axis=1)
    meta = stacking.fit_meta([p], y)
    assert (stacking.predict_meta(meta, [p], 3)[:, 2] == 0).all()


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError):
        stacking.fit_meta([np.full((4, 2), 0.5)], np.array([0, 1, 0, 1]), kind="svm")
