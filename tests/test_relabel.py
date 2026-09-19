"""Cutting the protozoa class of a prepared dataset into eukaryotic supergroups."""

import csv
import json

import h5py
import numpy as np
import pytest

from metapathpredict.cli import _SplitWriter
from metapathpredict.relabel import SCHEMES, new_class_of, relabel_dataset, relabeled_class_names
from metapathpredict.relatedness import fragment_genomes

L = 20
OLD = ["bacteria", "protozoa", "virus"]
LINEAGES = {
    "1": "cellular organisms; Bacteria; Pseudomonadota",
    "10": "cellular organisms; Eukaryota; Sar; Alveolata; Apicomplexa",
    "11": "cellular organisms; Eukaryota; Sar; Stramenopiles; Oomycota",
    "12": "cellular organisms; Eukaryota; Discoba; Euglenozoa",
    "13": "cellular organisms; Eukaryota; Rhodophyta; Bangiophyceae",
    "14": "cellular organisms; Eukaryota; Metamonada; Fornicata",
}


def test_lineage_decides_the_supergroup_and_other_classes_are_untouched():
    scheme = SCHEMES["protist4"]
    assert new_class_of("protozoa", LINEAGES["10"], scheme) == "protist_alveolata"
    assert new_class_of("protozoa", LINEAGES["11"], scheme) == "protist_stramenopiles"
    assert new_class_of("protozoa", LINEAGES["12"], scheme) == "protist_excavata"
    assert new_class_of("protozoa", LINEAGES["14"], scheme) == "protist_excavata"
    assert new_class_of("protozoa", LINEAGES["13"], scheme) == "protist_other"
    assert new_class_of("protozoa", "", scheme) == "protist_other"            # unknown lineage
    assert new_class_of("bacteria", LINEAGES["10"], scheme) == "bacteria"     # only protozoa is split
    assert relabeled_class_names(OLD, scheme) == ["bacteria", "protist_alveolata", "protist_stramenopiles",
                                                  "protist_excavata", "protist_other", "virus"]


@pytest.fixture
def dataset(tmp_path):
    """bacteria/protozoa/virus genomes with 2-3 fragments each, spread over train and test."""
    src = tmp_path / "src"
    src.mkdir()
    genomes = [  # accession, class, taxid, split, fragments
        ("B1", "bacteria", "1", "train", 3), ("P1", "protozoa", "10", "train", 2), ("P2", "protozoa", "11", "train", 3),
        ("P3", "protozoa", "12", "train", 2), ("V1", "virus", "2", "train", 2),
        ("B2", "bacteria", "1", "test", 2), ("P4", "protozoa", "13", "test", 2), ("P5", "protozoa", "14", "test", 3), ("V2", "virus", "2", "test", 2),
    ]
    rng = np.random.default_rng(0)
    for split in ("train", "val", "test"):
        writer = _SplitWriter(src / f"encoded_{split}_{L}.hdf5", L, 8, {"class_names": json.dumps(OLD), "num_classes": 3, "sequence_length": L})
        for acc, cls, _, sp, n in genomes:
            if sp == split:
                for _ in range(n):
                    writer.append(rng.integers(0, 2, size=(4, L)).astype(np.float32), OLD.index(cls))
        writer.close()
    with open(src / "split_assignments.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["accession", "class", "split", "species_taxid", "organism", "fragments"])
        for acc, cls, taxid, sp, n in genomes:
            w.writerow([acc, cls, sp, taxid, acc, n])
    with open(src / "test_fragments.fasta", "w") as f:
        counts = {}
        for acc, cls, _, sp, n in genomes:
            for _ in range(n) if sp == "test" else []:
                counts[cls] = counts.get(cls, 0) + 1
                f.write(f">{cls}_{counts[cls] - 1}|label={OLD.index(cls)}|acc={acc}\nACGT\n")
    (src / "metadata.json").write_text(json.dumps({"sequence_length": L, "class_names": OLD, "num_classes": 3,
                                                   "train_size": 12, "val_size": 0, "test_size": 9}))
    return src


def test_relabel_changes_only_labels_and_merges_back(dataset, tmp_path):
    dst = tmp_path / "dst"
    meta = relabel_dataset(dataset, dst, "protist4", LINEAGES)
    names = meta["class_names"]
    assert names == relabeled_class_names(OLD, SCHEMES["protist4"]) and meta["num_classes"] == 6
    assert meta["merge_to_original"] == [0, 1, 1, 1, 1, 2] and meta["original_class_names"] == OLD
    for split in ("train", "val", "test"):
        with h5py.File(dataset / f"encoded_{split}_{L}.hdf5") as old, h5py.File(dst / f"encoded_{split}_{L}.hdf5") as new:
            assert np.array_equal(old["sequences"][:], new["sequences"][:])              # fragments untouched
            assert np.array_equal(np.array(meta["merge_to_original"])[new["labels"][:]], old["labels"][:])
            assert json.loads(new.attrs["class_names"]) == names
    # per-genome labels: P1 alveolata, P2 stramenopiles, P3 excavata, in the training order
    with h5py.File(dst / f"encoded_train_{L}.hdf5") as f:
        assert f["labels"][:].tolist() == [0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 5, 5]
    assert fragment_genomes(dst, "test").tolist() == fragment_genomes(dataset, "test").tolist()  # genome order kept


def test_tsv_fasta_and_superclasses_follow_the_new_classes(dataset, tmp_path):
    dst = tmp_path / "dst"
    meta = relabel_dataset(dataset, dst, "protist4", LINEAGES)
    rows = list(csv.DictReader(open(dst / "split_assignments.tsv"), delimiter="\t"))
    assert {r["accession"]: r["class"] for r in rows}["P4"] == "protist_other"
    assert {r["accession"]: r["original_class"] for r in rows}["P4"] == "protozoa"
    fasta = [l.strip() for l in open(dst / "test_fragments.fasta") if l.startswith(">")]
    names = meta["class_names"]
    for header in fasta:
        label = int(header.split("label=")[1].split("|")[0])
        assert header[1:].startswith(names[label])
    assert meta["superclass_of_class"] == [0, 1, 1, 1, 1, 2]  # protists stay eukaryotes in the 3-class roll-up


def test_relabelling_fails_loudly_if_the_labels_cannot_be_reproduced(dataset, tmp_path):
    (dataset / "split_assignments.tsv").write_text(
        (dataset / "split_assignments.tsv").read_text().replace("B1\tbacteria", "B1\tvirus"))  # wrong class for a genome
    with pytest.raises(ValueError):
        relabel_dataset(dataset, tmp_path / "dst", "protist4", LINEAGES)
