"""Sliding-window contig classification."""

import numpy as np
import pytest

from metapathpredict import contigs

CLASSES = 3


def test_sliding_windows_non_overlapping_and_drops_the_partial_tail():
    assert contigs.sliding_windows(1000, 500, 500) == [(0, 500), (500, 1000)]
    assert contigs.sliding_windows(1250, 500, 500) == [(0, 500), (500, 1000)]  # last 250bp dropped


def test_sliding_windows_overlapping_step():
    assert contigs.sliding_windows(1000, 500, 250) == [(0, 500), (250, 750), (500, 1000)]


def test_sliding_windows_shorter_than_window_gives_nothing():
    assert contigs.sliding_windows(300, 500, 500) == []


def test_sliding_windows_rejects_non_positive_window_or_step():
    with pytest.raises(ValueError):
        contigs.sliding_windows(1000, 0, 500)
    with pytest.raises(ValueError):
        contigs.sliding_windows(1000, 500, 0)


def test_encode_window_one_hot_and_unknown_base_is_all_zero():
    encoded = contigs.encode_window("ACGTN")
    assert encoded.shape == (4, 5)
    assert encoded[:, 0].tolist() == [1, 0, 0, 0]
    assert encoded[:, 1].tolist() == [0, 1, 0, 0]
    assert encoded[:, 2].tolist() == [0, 0, 1, 0]
    assert encoded[:, 3].tolist() == [0, 0, 0, 1]
    assert encoded[:, 4].tolist() == [0, 0, 0, 0]  # N: no uniform-0.25 guess, matches the training encoding
    assert contigs.encode_window("acgt")[:, 0].tolist() == [1, 0, 0, 0]  # case-insensitive


def test_classify_contig_matches_a_single_window_prediction_when_the_contig_is_one_window():
    calls = []

    def probs_fn(window):
        calls.append(window)
        return [0.7, 0.2, 0.1]

    result = contigs.classify_contig(probs_fn, "A" * 500, window=500)
    assert calls == ["A" * 500]
    assert result["num_windows"] == 1 and result["predicted_class"] == 0
    assert result["probs"] == pytest.approx([0.7, 0.2, 0.1], abs=1e-6)


def test_classify_contig_aggregates_by_geometric_mean_not_arithmetic_mean():
    # two distinguishable windows: one screams class 0, the other is neutral - geometric mean should
    # still favour 0 but be pulled down relative to the confident window's own probability
    seq = "A" * 500 + "C" * 500

    def probs_fn(window):
        return [0.99, 0.005, 0.005] if window == "A" * 500 else [0.34, 0.33, 0.33]

    result = contigs.classify_contig(probs_fn, seq, window=500)
    assert result["num_windows"] == 2 and result["predicted_class"] == 0
    manual = np.exp(np.mean(np.log([[0.99, 0.005, 0.005], [0.34, 0.33, 0.33]]), axis=0))
    manual /= manual.sum()
    assert result["probs"] == pytest.approx(manual.tolist(), abs=1e-6)


def test_classify_contig_disagreement_between_windows_can_flip_the_majority():
    # three distinguishable windows; two lean class 1, one screams class 0 - class 1 should still win
    seqs = ["A" * 500, "C" * 500, "G" * 500]
    votes = {seqs[0]: [0.9, 0.05, 0.05], seqs[1]: [0.1, 0.8, 0.1], seqs[2]: [0.1, 0.75, 0.15]}
    full = "".join(seqs)

    result = contigs.classify_contig(lambda w: votes[w], full, window=500)
    assert result["predicted_class"] == 1
    assert [w["start"] for w in result["windows"]] == [0, 500, 1000]


def test_classify_contig_rejects_a_sequence_shorter_than_one_window():
    with pytest.raises(ValueError):
        contigs.classify_contig(lambda w: [1, 0, 0], "A" * 100, window=500)


def test_classify_contig_windows_report_matches_the_raw_per_window_probabilities():
    def probs_fn(window):
        return [0.5, 0.3, 0.2]

    result = contigs.classify_contig(probs_fn, "A" * 1000, window=500)
    assert len(result["windows"]) == 2
    for w in result["windows"]:
        assert w["probs"] == pytest.approx([0.5, 0.3, 0.2], abs=1e-6)
