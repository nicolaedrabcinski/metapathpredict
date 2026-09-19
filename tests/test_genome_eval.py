"""Genome-level accuracy with a bootstrap over genomes."""

import numpy as np
import pytest

from metapathpredict.genome_eval import (
    evaluate_by_genome,
    paired_difference,
    format_genome_report,
    genome_report,
    genome_report_to_metrics,
    genome_table,
    read_fragment_accessions,
)

CLASSES = ["bacteria", "archaea", "fungi", "protozoa", "plant", "invertebrate", "vertebrate", "virus"]


def _split(per_genome):
    """per_genome: [(accession, class_id, n_fragments, n_correct)] -> targets, preds, genomes."""
    targets, preds, genomes = [], [], []
    for acc, cls, n, correct in per_genome:
        targets += [cls] * n
        preds += [cls] * correct + [(cls + 1) % 8] * (n - correct)
        genomes += [acc] * n
    return np.array(targets), np.array(preds), np.array(genomes)


def test_reads_accessions_in_file_order(tmp_path):
    fasta = tmp_path / "f.fasta"
    fasta.write_text(">bacteria_0|label=0|acc=GCF_1\nACGT\n>virus_1|label=7|acc=GCF_9\nACGT\n")
    assert read_fragment_accessions(fasta).tolist() == ["GCF_1", "GCF_9"]
    fasta.write_text(">bacteria_0|label=0\nACGT\n")
    with pytest.raises(ValueError):
        read_fragment_accessions(fasta)


def test_genome_table_counts_fragments_and_correct_predictions():
    t, p, g = _split([("a", 0, 10, 7), ("b", 0, 5, 5), ("c", 7, 4, 0)])
    table = genome_table(t, p, g)
    assert table["accession"].tolist() == ["a", "b", "c"]
    assert table["fragments"].tolist() == [10, 5, 4]
    assert table["correct"].tolist() == [7, 5, 0]
    assert table["label"].tolist() == [0, 0, 7]


def test_a_genome_with_two_classes_or_misaligned_inputs_are_rejected():
    with pytest.raises(ValueError):
        genome_table(np.array([0, 1]), np.array([0, 1]), np.array(["a", "a"]))
    with pytest.raises(ValueError):
        genome_table(np.array([0, 1]), np.array([0]), np.array(["a", "b"]))


def test_point_estimates_are_the_plain_accuracies():
    t, p, g = _split([("a", 0, 10, 7), ("b", 0, 5, 5), ("c", 7, 4, 0)])
    report = genome_report(t, p, g, CLASSES, n_boot=200)
    assert report["accuracy"]["value"] == pytest.approx((t == p).mean())
    assert report["recall"]["bacteria"]["value"] == pytest.approx(12 / 15)
    assert report["recall"]["virus"]["value"] == 0.0
    assert report["n_genomes"] == {"bacteria": 2, "virus": 1}
    assert report["balanced_accuracy"]["value"] == pytest.approx((12 / 15 + 0.0) / 2)


def test_perfect_predictions_give_a_degenerate_interval():
    t, p, g = _split([("a", 0, 6, 6), ("b", 0, 6, 6), ("c", 2, 6, 6), ("d", 2, 6, 6)])
    report = genome_report(t, p, g, CLASSES, n_boot=100)
    assert report["accuracy"] == {"value": 1.0, "ci_low": 1.0, "ci_high": 1.0}


def test_two_genomes_that_disagree_give_a_wide_interval():
    # same fragment-level accuracy, but the genome-level uncertainty is what the interval must show
    t, p, g = _split([("good", 0, 100, 100), ("bad", 0, 100, 0)])
    report = genome_report(t, p, g, CLASSES, n_boot=2000)
    assert report["accuracy"]["value"] == pytest.approx(0.5)
    assert report["accuracy"]["ci_low"] == 0.0 and report["accuracy"]["ci_high"] == 1.0


def test_many_consistent_genomes_give_a_narrow_interval_and_the_seed_is_deterministic():
    t, p, g = _split([(f"g{i}", 0, 20, 14) for i in range(30)])
    a = genome_report(t, p, g, CLASSES, n_boot=300, seed=1)
    b = genome_report(t, p, g, CLASSES, n_boot=300, seed=1)
    assert a["accuracy"] == b["accuracy"]
    assert a["accuracy"]["ci_high"] - a["accuracy"]["ci_low"] < 0.01


def test_three_class_accuracy_counts_prokaryote_confusions_as_correct():
    # bacteria (0) predicted as archaea (1): wrong in 8 classes, right in prokaryote/eukaryote/virus
    t, p, g = np.zeros(10, int), np.ones(10, int), np.array(["a"] * 10)
    report = genome_report(t, p, g, CLASSES, n_boot=50)
    assert report["accuracy"]["value"] == 0.0
    assert report["accuracy_3class"]["value"] == 1.0


def test_per_genome_listing_and_metrics_are_exposed():
    t, p, g = _split([("a", 0, 10, 7), ("b", 7, 4, 4)])
    report = genome_report(t, p, g, CLASSES, n_boot=50)
    assert {row["accession"]: row["accuracy"] for row in report["per_genome"]} == {"a": 0.7, "b": 1.0}
    metrics = genome_report_to_metrics(report, "test/x")
    assert {"test/x/genome_accuracy", "test/x/genome_accuracy_ci_low", "test/x/genome_accuracy_ci_high",
            "test/x/genome_recall_virus_ci_low"} <= set(metrics)
    assert "accuracy" in format_genome_report(report)


def test_evaluate_by_genome_skips_a_missing_or_misaligned_fasta(tmp_path):
    t, p, _ = _split([("a", 0, 3, 3)])
    assert evaluate_by_genome(tmp_path, t, p, CLASSES) is None
    (tmp_path / "test_fragments.fasta").write_text(">x|label=0|acc=a\nAC\n")
    assert evaluate_by_genome(tmp_path, t, p, CLASSES) is None  # 1 fragment in the file, 3 predictions
    (tmp_path / "test_fragments.fasta").write_text("".join(f">x|label=0|acc=a\nAC\n" for _ in range(3)))
    assert evaluate_by_genome(tmp_path, t, p, CLASSES, n_boot=20)["accuracy"]["value"] == 1.0


def _paired(per_genome_a, per_genome_b):
    t, pa, g = _split(per_genome_a)
    _, pb, _ = _split(per_genome_b)
    return t, pa, pb, g


def test_paired_difference_is_narrower_than_two_independent_intervals():
    # genomes differ a lot in difficulty (30%..90%) but B is uniformly 5 points better than A
    a = [(f"g{i}", 0, 100, 30 + 3 * i) for i in range(20)]
    b = [(acc, c, n, k + 5) for acc, c, n, k in a]
    t, pa, pb, g = _paired(a, b)
    diff = paired_difference(t, pa, pb, g, CLASSES, n_boot=500)
    assert diff["accuracy"]["value"] == pytest.approx(0.05)
    assert diff["accuracy"]["ci_low"] == pytest.approx(0.05) and diff["accuracy"]["ci_high"] == pytest.approx(0.05)
    assert diff["p_b_better"] == 1.0
    independent = genome_report(t, pa, g, CLASSES, n_boot=500)["accuracy"]
    assert independent["ci_high"] - independent["ci_low"] > 0.1  # the unpaired view would call 5 points noise


def test_paired_difference_of_identical_models_is_zero_and_sign_follows_the_better_model():
    a = [(f"g{i}", 0, 20, 10) for i in range(10)]
    t, pa, pb, g = _paired(a, a)
    same = paired_difference(t, pa, pb, g, CLASSES, n_boot=100)
    assert same["accuracy"] == {"value": 0.0, "ci_low": 0.0, "ci_high": 0.0} and same["p_b_better"] == 0.0
    t, pa, pb, g = _paired(a, [(acc, c, n, 5) for acc, c, n, _ in a])
    worse = paired_difference(t, pa, pb, g, CLASSES, n_boot=100)
    assert worse["accuracy"]["value"] == pytest.approx(-0.25) and worse["p_b_better"] == 0.0


def test_paired_difference_is_uncertain_when_only_two_genomes_disagree():
    t, pa, pb, g = _paired([("x", 0, 50, 50), ("y", 0, 50, 0)], [("x", 0, 50, 0), ("y", 0, 50, 50)])
    diff = paired_difference(t, pa, pb, g, CLASSES, n_boot=2000)
    assert diff["accuracy"]["value"] == 0.0
    assert diff["accuracy"]["ci_low"] < 0 < diff["accuracy"]["ci_high"]
