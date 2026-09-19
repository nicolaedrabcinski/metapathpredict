"""Explanation methods on a network that has learned a known motif."""

from collections import Counter

import numpy as np
import pytest
import torch
import torch.nn as nn

from metapathpredict import explain
from metapathpredict.models.configurable_cnn import ConfigurableCNN

MOTIF, LENGTH = "ACGTTGCA", 100


def _onehot(idx):
    return explain.to_onehot(np.asarray(idx, dtype=np.int8))


def _data(n, seed):
    """Random sequences; those of class 1 carry MOTIF at a random position."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, 4, size=(n, LENGTH)).astype(np.int8)
    y = (np.arange(n) % 2).astype(np.int64)
    starts = rng.integers(10, LENGTH - len(MOTIF) - 10, size=n)
    for i in np.flatnonzero(y == 1):
        idx[i, starts[i]:starts[i] + len(MOTIF)] = ["ACGT".index(b) for b in MOTIF]
    return _onehot(idx), y, starts


@pytest.fixture(scope="module")
def trained():
    torch.manual_seed(0)
    x, y, _ = _data(1000, 0)
    model = ConfigurableCNN(num_classes=2, kernel_preset="small", base_channels=16, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    for _ in range(60):
        model.train()
        for lo in range(0, len(xt), 64):
            loss = nn.functional.cross_entropy(model(xt[lo:lo + 64]), yt[lo:lo + 64])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    xv, yv, starts = _data(200, 1)
    assert explain.predict(model, xv).__eq__(yv).mean() > 0.95  # the fixture is only useful if it learned the task
    return model, xv, yv, starts


def _hit_rate(scores, starts, y, tolerance):
    hits = [abs(int(np.argmax(s)) - (st + len(MOTIF) // 2)) <= len(MOTIF) // 2 + tolerance
            for s, st, label in zip(scores, starts, y) if label == 1]
    return float(np.mean(hits))


def test_integrated_gradients_satisfy_completeness_and_point_at_the_motif(trained):
    model, x, y, starts = trained
    xt = torch.from_numpy(x[y == 1][:40])
    attribution = explain.integrated_gradients(model, xt, target=1, steps=64)
    assert attribution.shape == xt.shape
    assert explain.completeness_gap(model, xt, attribution, target=1) < 0.05
    scores = explain.position_scores(attribution).numpy()
    assert _hit_rate(scores, starts[y == 1][:40], np.ones(40), tolerance=3) > 0.7


def test_gradient_x_input_and_grad_cam_localise_the_motif(trained):
    model, x, y, starts = trained
    xt = torch.from_numpy(x[y == 1][:40])
    grad = explain.position_scores(explain.gradient_x_input(model, xt, target=1)).numpy()
    assert _hit_rate(grad, starts[y == 1][:40], np.ones(40), tolerance=3) > 0.6
    cam = explain.grad_cam(model, xt, target=1).numpy()
    assert cam.shape == (40, LENGTH) and cam.min() >= 0 and np.allclose(cam.max(axis=1), 1.0)
    assert _hit_rate(cam, starts[y == 1][:40], np.ones(40), tolerance=10) > 0.7  # coarse: one cell = 8 bases


def test_grad_cam_needs_a_conv_block_or_an_explicit_layer():
    with pytest.raises(ValueError):
        explain.grad_cam(nn.Sequential(nn.Flatten(), nn.Linear(4 * LENGTH, 2)), torch.zeros(1, 4, LENGTH))


def test_deleting_the_important_positions_beats_deleting_random_ones(trained):
    model, x, y, _ = trained
    xt = torch.from_numpy(x[y == 1][:60])
    fractions = np.linspace(0, 0.3, 7)
    ig = explain.position_scores(explain.integrated_gradients(model, xt, target=1, steps=32))
    curve = explain.deletion_curve(model, xt, ig, fractions, target=1)
    random_curve = np.mean([explain.deletion_curve(model, xt, torch.rand_like(ig), fractions, target=1) for _ in range(5)], axis=0)
    assert curve[0] == pytest.approx(random_curve[0])  # nothing removed yet
    assert explain.area_under(curve, fractions) < explain.area_under(random_curve, fractions) - 0.05


def test_randomization_check_separates_learned_from_input_driven_maps(trained):
    model, x, y, _ = trained
    xt = torch.from_numpy(x[y == 1][:30])
    constant = lambda m, batch: batch.sum(dim=1)  # ignores the weights entirely
    assert explain.randomization_check(model, xt, constant) == pytest.approx(1.0)
    ig = lambda m, batch: explain.position_scores(explain.integrated_gradients(m, batch, target=1, steps=16))
    assert explain.randomization_check(model, xt, ig) < 0.9


def test_dinucleotide_shuffle_keeps_dinucleotides_and_ends_but_changes_the_sequence():
    rng = np.random.default_rng(0)
    for _ in range(30):
        seq = rng.integers(0, 4, size=120).astype(np.int8)
        out = explain.dinucleotide_shuffle(seq, rng)
        assert len(out) == len(seq) and out[0] == seq[0] and out[-1] == seq[-1]
        assert Counter(zip(out[:-1], out[1:])) == Counter(zip(seq[:-1], seq[1:]))
    assert (explain.dinucleotide_shuffle(seq, rng) != seq).any()
    assert list(explain.dinucleotide_shuffle(np.array([1, 2, 3], dtype=np.int8), rng)) == [1, 2, 3]  # too short to move


def test_shuffle_sequences_modes():
    x, _, _ = _data(10, 3)
    rng = np.random.default_rng(0)
    assert explain.shuffle_sequences(x, "none", rng) is x
    mono = explain.shuffle_sequences(x, "mono", rng)
    assert np.array_equal(mono.sum(axis=2), x.sum(axis=2)) and not np.array_equal(mono, x)  # same base counts
    dinuc = explain.shuffle_sequences(x, "dinuc", rng)
    assert np.array_equal(dinuc.sum(axis=2), x.sum(axis=2))
    with pytest.raises(ValueError):
        explain.shuffle_sequences(x, "reverse", rng)


def test_shuffling_destroys_a_motif_but_not_a_composition_signal(trained):
    model, x, y, _ = trained
    original = explain.shuffle_accuracy(model, x, y, "none")["accuracy"]
    shuffled = explain.shuffle_accuracy(model, x, y, "dinuc")
    assert original > 0.95 and shuffled["accuracy"] < 0.75  # the motif is order information
    assert set(shuffled["per_class"]) == {0, 1}


def test_first_layer_motifs_have_valid_pwms_and_recover_part_of_the_motif(trained):
    model, x, y, _ = trained
    result = explain.first_layer_motifs(model, x, y, num_classes=2, top_windows=40)
    filters, width = result["pwm"].shape[0], result["pwm"].shape[2]
    assert result["pwm"].shape == (filters, 4, width) == (16, 4, 5)
    assert np.allclose(result["pwm"].sum(axis=1), 1.0, atol=1e-4) and (result["information"] >= -1e-6).all()
    assert np.isfinite(result["specificity"]).all() and (result["specificity"] >= 0).all()
    # the filter that separates the motif class best has learned a piece of the motif
    best = int(np.argmax(result["class_mean"][1] - result["class_mean"][0]))
    consensus = "".join(explain.BASES[i] for i in result["pwm"][best].argmax(axis=0))
    assert consensus in MOTIF and result["best_class"][best] == 1
