"""Near/far relatedness of test genomes to the training set."""

import csv

import numpy as np
import pytest

from metapathpredict.genome_eval import evaluate_by_genome, format_genome_report, genome_report, genome_report_to_metrics
from metapathpredict.relatedness import annotate_assignments, relatedness_by_accession

FIELDS = ["accession", "class", "split", "species_taxid", "organism", "fragments", "phylum", "class_", "order", "family", "genus"]
CLASSES = ["bacteria", "archaea", "fungi", "protozoa", "plant", "invertebrate", "vertebrate", "virus"]


def _row(acc, cls, split, phylum="", order="", family="", genus=""):
    return {"accession": acc, "class": cls, "split": split, "species_taxid": acc, "organism": acc, "fragments": 1,
            "phylum": phylum, "class_": "", "order": order, "family": family, "genus": genus}


def _write(path, rows):
    header = ["accession", "class", "split", "species_taxid", "organism", "fragments",
              "lineage_phylum", "lineage_class", "lineage_order", "lineage_family", "lineage_genus"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)
        for r in rows:
            w.writerow([r["accession"], r["class"], r["split"], r["species_taxid"], r["organism"], r["fragments"],
                        r["phylum"], r["class_"], r["order"], r["family"], r["genus"]])


@pytest.fixture
def assignments(tmp_path):
    path = tmp_path / "split_assignments.tsv"
    _write(path, [
        _row("t1", "bacteria", "train", "P1", "O1", "F1", "G1"),
        _row("t2", "bacteria", "train", "P2", "O2", "F2", "G2"),
        _row("t3", "fungi", "train", "P3", "O3", "F3", "G3"),
        _row("x_genus", "bacteria", "test", "P1", "O1", "F1", "G1"),      # same genus as t1
        _row("x_family", "bacteria", "test", "P1", "O1", "F1", "GX"),     # same family only
        _row("x_order", "bacteria", "test", "P1", "O1", "FX", "GX"),
        _row("x_phylum", "bacteria", "test", "P2", "OX", "FX", "GX"),
        _row("x_none", "bacteria", "test", "PX", "OX", "FX", "GX"),
        _row("x_other_class", "archaea", "test", "P1", "O1", "F1", "G1"),  # relatives exist only in another class
        _row("v1", "bacteria", "val", "P1", "O1", "F1", "G1"),             # validation genomes are ignored
    ])
    return path


def test_closest_shared_rank_is_found_within_the_same_class(assignments):
    result = relatedness_by_accession(assignments)
    assert result == {"x_genus": "genus", "x_family": "family", "x_order": "order", "x_phylum": "phylum",
                      "x_none": "none", "x_other_class": "none"}


def test_missing_file_or_lineage_columns_give_none(tmp_path):
    assert relatedness_by_accession(tmp_path / "nope.tsv") is None
    plain = tmp_path / "plain.tsv"
    plain.write_text("accession\tclass\tsplit\tspecies_taxid\nA\tbacteria\ttest\t1\n")
    assert relatedness_by_accession(plain) is None


def test_annotate_adds_lineage_columns_once(tmp_path):
    tsv = tmp_path / "a.tsv"
    tsv.write_text("accession\tclass\tsplit\tspecies_taxid\torganism\tfragments\n"
                   "A\tbacteria\ttrain\t5\tx\t3\nB\tvirus\ttest\t6\ty\t2\n")
    lineages = {"5": {"phylum": "P", "class": "C", "order": "O", "family": "F", "genus": "G"}}
    assert annotate_assignments(tsv, lineages) == 2
    rows = list(csv.DictReader(open(tsv), delimiter="\t"))
    assert rows[0]["lineage_family"] == "F" and rows[0]["group"] == "family:F"
    assert rows[0]["class"] == "bacteria"  # the dataset class column is untouched
    assert rows[1]["lineage_family"] == "" and rows[1]["group"] == "genome:B"  # unknown lineage falls back to the genome
    assert annotate_assignments(tsv, lineages) == 0  # already annotated


def _split(per_genome):
    t, p, g = [], [], []
    for acc, cls, n, correct in per_genome:
        t += [cls] * n
        p += [cls] * correct + [(cls + 1) % 8] * (n - correct)
        g += [acc] * n
    return np.array(t), np.array(p), np.array(g)


def test_report_breaks_accuracy_down_by_relatedness():
    t, p, g = _split([("near1", 0, 10, 9), ("near2", 0, 10, 7), ("far1", 0, 10, 3), ("far2", 0, 10, 5)])
    relatedness = {"near1": "genus", "near2": "family", "far1": "order", "far2": "none"}
    report = genome_report(t, p, g, CLASSES, n_boot=50, relatedness=relatedness)
    assert report["by_relatedness"]["genus"] == {"genomes": 1, "fragments": 10, "accuracy": 0.9}
    assert report["near_far"]["near"] == {"genomes": 2, "fragments": 20, "accuracy": 0.8}
    assert report["near_far"]["far"] == {"genomes": 2, "fragments": 20, "accuracy": 0.4}
    metrics = genome_report_to_metrics(report, "test/x")
    assert metrics["test/x/relatedness_near_accuracy"] == 0.8 and metrics["test/x/relatedness_far_genomes"] == 2.0
    assert "near" in format_genome_report(report) and "far" in format_genome_report(report)


def test_report_without_relatedness_has_no_breakdown():
    t, p, g = _split([("a", 0, 10, 9)])
    assert "near_far" not in genome_report(t, p, g, CLASSES, n_boot=20)


def test_evaluate_by_genome_reads_split_assignments_next_to_the_fasta(assignments):
    tmp = assignments.parent
    t = np.zeros(4, dtype=int)
    (tmp / "test_fragments.fasta").write_text("".join(f">x|label=0|acc={a}\nAC\n" for a in ("x_genus", "x_genus", "x_none", "x_none")))
    report = evaluate_by_genome(tmp, t, np.array([0, 0, 1, 1]), CLASSES, n_boot=20)
    assert report["near_far"]["near"]["accuracy"] == 1.0 and report["near_far"]["far"]["accuracy"] == 0.0


def test_lineage_targets_label_every_fragment_and_ignore_unknown_names(tmp_path):
    from metapathpredict.relatedness import fragment_genomes, lineage_targets

    path = tmp_path / "split_assignments.tsv"
    _write(path, [
        {**_row("a", "bacteria", "train", phylum="P1"), "fragments": 3},
        {**_row("b", "bacteria", "train", phylum="P2"), "fragments": 2},
        {**_row("c", "fungi", "train", phylum=""), "fragments": 1},          # no phylum known
        {**_row("d", "fungi", "val", phylum="P9"), "fragments": 2},           # a phylum the training split never saw
    ])
    assert fragment_genomes(tmp_path, "train").tolist() == ["a", "a", "a", "b", "b", "c"]
    labels, names = lineage_targets(tmp_path, "train", "phylum")
    assert names == ["P1", "P2"] and labels.tolist() == [0, 0, 0, 1, 1, -100]
    val_labels, _ = lineage_targets(tmp_path, "val", "phylum", names=names)
    assert val_labels.tolist() == [-100, -100]
    with pytest.raises(ValueError):
        lineage_targets(tmp_path, "train", "kingdom")
    (tmp_path / "plain.tsv").write_text("accession\tclass\tsplit\tspecies_taxid\tfragments\nA\tbacteria\ttrain\t1\t2\n")
    (tmp_path / "plain").mkdir()
    (tmp_path / "plain" / "split_assignments.tsv").write_text((tmp_path / "plain.tsv").read_text())
    with pytest.raises(ValueError):
        lineage_targets(tmp_path / "plain", "train", "phylum")


def test_genome_balanced_weights_are_inverse_fragment_count_per_genome(tmp_path):
    from metapathpredict.relatedness import genome_balanced_weights

    _write(tmp_path / "split_assignments.tsv", [
        {**_row("big", "bacteria", "train"), "fragments": 100},
        {**_row("small", "bacteria", "train"), "fragments": 10},
        {**_row("other_split", "bacteria", "test"), "fragments": 5},
    ])
    import numpy as np

    weights = genome_balanced_weights(tmp_path, "train")
    assert len(weights) == 110
    assert np.allclose(weights[:100], 0.01) and np.allclose(weights[100:], 0.1)
    # every genome's fragments sum to the same total weight regardless of how many fragments it has
    assert weights[:100].sum() == pytest.approx(weights[100:].sum())


def test_genome_balanced_weights_build_a_valid_weighted_sampler(tmp_path):
    import torch

    from metapathpredict.relatedness import genome_balanced_weights

    _write(tmp_path / "split_assignments.tsv", [
        {**_row("big", "bacteria", "train"), "fragments": 100},
        {**_row("small", "bacteria", "train"), "fragments": 10},
    ])
    weights = genome_balanced_weights(tmp_path, "train")
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=100000, replacement=True)
    drawn = torch.tensor(list(sampler))
    from_small = (drawn >= 100).float().mean().item()
    assert 0.4 < from_small < 0.6  # each genome ~half the draws despite the 10x size difference
