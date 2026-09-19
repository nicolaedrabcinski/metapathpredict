"""Tests for genome-level splitting, manifest-based dataset preparation, and early stopping."""

import argparse
import csv
import gzip
import json

import h5py
import numpy as np
import pytest

from metapathpredict.cli import (
    EarlyStopping,
    _allocate_quotas,
    _assign_sequence_splits,
    prepare_command,
)
from metapathpredict.config.settings import (
    NCBI_GROUP_TO_TAXON,
    SUPERCLASSES,
    TAXON_CLASSES,
    superclass_index_map,
)


class TestAssignSequenceSplits:
    def test_every_split_gets_a_genome_when_there_are_enough(self):
        splits = _assign_sequence_splits(10, 0.7, 0.15, 0.15, np.random.RandomState(0))
        assert len(splits) == 10
        assert {"train", "val", "test"} <= set(splits)

    def test_minimum_three_genomes_covers_all_splits(self):
        splits = _assign_sequence_splits(3, 0.7, 0.15, 0.15, np.random.RandomState(0))
        assert sorted(splits) == ["test", "train", "val"]

    def test_deterministic_for_same_seed(self):
        a = _assign_sequence_splits(50, 0.8, 0.1, 0.1, np.random.RandomState(42))
        b = _assign_sequence_splits(50, 0.8, 0.1, 0.1, np.random.RandomState(42))
        assert a == b

    def test_ratios_roughly_respected(self):
        splits = _assign_sequence_splits(100, 0.7, 0.15, 0.15, np.random.RandomState(1))
        assert splits.count("train") == 70
        assert splits.count("val") == 15 and splits.count("test") == 15

    def test_tiny_inputs(self):
        assert _assign_sequence_splits(1, 0.8, 0.1, 0.1, np.random.RandomState(0)) == ["train"]
        assert sorted(_assign_sequence_splits(2, 0.8, 0.1, 0.1, np.random.RandomState(0))) == ["train", "val"]


class TestAllocateQuotas:
    def test_even_split_when_capacity_is_ample(self):
        assert _allocate_quotas([100, 100, 100], 30) == pytest.approx([10, 10, 10])

    def test_small_genomes_capped_and_remainder_redistributed(self):
        quotas = _allocate_quotas([2, 100, 100], 30)
        assert quotas[0] == 2
        assert quotas[1] == pytest.approx(14) and quotas[2] == pytest.approx(14)
        assert sum(quotas) == pytest.approx(30)

    def test_never_exceeds_capacity_when_budget_is_too_large(self):
        caps = [3, 4, 5]
        assert _allocate_quotas(caps, 1000) == pytest.approx(caps)


class TestSuperclasses:
    def test_taxon_classes_roll_up_to_three(self):
        mapping = superclass_index_map(TAXON_CLASSES)
        names = [SUPERCLASSES[i] for i in mapping]
        assert names == [
            "prokaryote", "prokaryote", "eukaryote", "eukaryote",
            "eukaryote", "eukaryote", "eukaryote", "virus",
        ]

    def test_legacy_three_class_names(self):
        assert superclass_index_map(["bacteria", "eukaryotic", "virus"]) == [0, 1, 2]

    def test_unknown_class_has_no_mapping(self):
        assert superclass_index_map(["bacteria", "mystery"]) is None

    def test_every_ncbi_group_maps_to_a_known_class(self):
        assert set(NCBI_GROUP_TO_TAXON.values()) == set(TAXON_CLASSES)


class TestEarlyStopping:
    def test_stops_after_patience_epochs_without_improvement(self):
        stopper = EarlyStopping(patience=3, mode="min")
        assert [stopper.step(v) for v in (5.0, 4.0, 4.1, 4.2)] == [False, False, False, False]
        assert stopper.step(4.3) is True

    def test_improvement_resets_counter(self):
        stopper = EarlyStopping(patience=2, mode="max")
        assert [stopper.step(v) for v in (0.5, 0.4, 0.6, 0.5)] == [False, False, False, False]
        assert stopper.step(0.5) is True

    def test_patience_zero_disables(self):
        stopper = EarlyStopping(patience=0)
        assert not any(stopper.step(1.0) for _ in range(20))


@pytest.fixture
def synthetic_manifest(tmp_path):
    """3 genomes (one species each) for every NCBI group, each with a class-specific base bias."""
    rng = np.random.RandomState(0)
    groups = list(NCBI_GROUP_TO_TAXON)
    rows = []
    genomes = tmp_path / "genomes"
    genomes.mkdir()
    for gi, group in enumerate(groups):
        for k in range(3):
            acc = f"GCF_{gi:03d}{k:03d}.1"
            probs = np.full(4, 0.2)
            probs[gi % 4] += 0.4
            seq = "".join(rng.choice(list("ACGT"), size=6000, p=probs / probs.sum()))
            with gzip.open(genomes / f"{acc}.fna.gz", "wt") as f:
                f.write(f">{acc}_chr1 test\n{seq[:3000]}\n>{acc}_chr2 test\n{seq[3000:]}\n")
            rows.append({
                "accession": acc, "group": group, "species_taxid": str(1000 * gi + k),
                "organism": f"Testus {group}{k}", "genome_size": "6000",
                "path": f"genomes/{acc}.fna.gz",
            })
    manifest = tmp_path / "manifest.tsv"
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    return manifest


def _run_prepare(manifest, out_dir, per_class=60):
    args = argparse.Namespace(
        inputs=[], manifest=str(manifest), output=str(out_dir), config=None,
        length=100, max_fragments=None, chunk_size=50, fragments_per_class=per_class,
    )
    return prepare_command(args)


class TestPrepareFromManifest:
    def test_builds_eight_class_dataset_split_by_genome(self, synthetic_manifest, tmp_path):
        out = tmp_path / "out"
        assert _run_prepare(synthetic_manifest, out) == 0

        meta = json.loads((out / "metadata.json").read_text())
        assert meta["num_classes"] == 8
        assert meta["class_names"] == TAXON_CLASSES

        for split in ("train", "val", "test"):
            with h5py.File(out / f"encoded_{split}_100.hdf5") as f:
                assert f["sequences"].shape[1:] == (4, 100)
                assert int(f.attrs["num_classes"]) == 8
                assert json.loads(f.attrs["class_names"]) == TAXON_CLASSES
                labels = f["labels"][:]
                assert len(labels) == len(f["sequences"]) > 0
                assert labels.max() < 8

    def test_no_species_shared_between_splits(self, synthetic_manifest, tmp_path):
        out = tmp_path / "out"
        assert _run_prepare(synthetic_manifest, out) == 0
        with open(out / "split_assignments.tsv") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        by_species = {}
        for r in rows:
            by_species.setdefault(r["species_taxid"], set()).add(r["split"])
        assert all(len(s) == 1 for s in by_species.values())
        assert {r["split"] for r in rows} == {"train", "val", "test"}

    def test_test_fasta_matches_test_split(self, synthetic_manifest, tmp_path):
        out = tmp_path / "out"
        assert _run_prepare(synthetic_manifest, out) == 0
        with h5py.File(out / "encoded_test_100.hdf5") as f:
            n_test = len(f["labels"])
        n_records = sum(1 for line in open(out / "test_fragments.fasta") if line.startswith(">"))
        assert n_records == n_test

    def test_fewer_than_three_genomes_per_class_is_rejected(self, tmp_path):
        manifest = tmp_path / "manifest.tsv"
        with open(manifest, "w", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=["accession", "group", "species_taxid", "organism", "genome_size", "path"],
                delimiter="\t",
            )
            w.writeheader()
            w.writerow({"accession": "GCF_1", "group": "bacteria", "species_taxid": "1",
                        "organism": "x", "genome_size": "1000", "path": "genomes/none.fna.gz"})
        assert _run_prepare(manifest, tmp_path / "out") == 1


class TestFamilySplit:
    RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}

    def test_groups_never_straddle_splits_and_every_split_is_used(self):
        from metapathpredict.cli import _assign_group_splits

        groups = [f"f{i}" for i in range(12) for _ in range(1 + i % 4)]  # 12 families of 1-4 genomes
        splits = _assign_group_splits(groups, self.RATIOS, np.random.RandomState(0))
        by_group = {}
        for g, s in zip(groups, splits):
            by_group.setdefault(g, set()).add(s)
        assert all(len(v) == 1 for v in by_group.values())
        assert set(splits) == {"train", "val", "test"}
        assert splits.count("train") > splits.count("val") and splits.count("train") > splits.count("test")

    def test_is_deterministic_per_seed_and_the_seed_changes_the_split(self):
        from metapathpredict.cli import _assign_group_splits

        groups = [f"f{i}" for i in range(30) for _ in range(2)]
        a = _assign_group_splits(groups, self.RATIOS, np.random.RandomState(1))
        b = _assign_group_splits(groups, self.RATIOS, np.random.RandomState(1))
        c = _assign_group_splits(groups, self.RATIOS, np.random.RandomState(2))
        assert a == b and a != c

    def test_three_groups_give_one_group_per_split(self):
        from metapathpredict.cli import _assign_group_splits

        splits = _assign_group_splits(["a", "a", "a", "b", "b", "c"], self.RATIOS, np.random.RandomState(0))
        assert sorted(set(splits)) == ["test", "train", "val"]

    def test_prepare_keeps_families_together_and_records_lineage(self, tmp_path):
        import json

        rng = np.random.RandomState(0)
        rows, lineages = [], {}
        (tmp_path / "genomes").mkdir()
        for gi, group in enumerate(NCBI_GROUP_TO_TAXON):
            for k in range(6):
                acc, taxid = f"GCF_{gi:03d}{k:03d}.1", str(1000 * gi + k)
                seq = "".join(rng.choice(list("ACGT"), size=4000))
                with gzip.open(tmp_path / "genomes" / f"{acc}.fna.gz", "wt") as f:
                    f.write(f">{acc}\n{seq}\n")
                rows.append({"accession": acc, "group": group, "species_taxid": taxid, "organism": f"Testus {group}{k}",
                             "genome_size": "4000", "path": f"genomes/{acc}.fna.gz"})
                lineages[taxid] = {"phylum": "P", "class": "C", "order": f"O{gi}", "family": f"F{gi}_{k // 2}",
                                   "genus": f"G{gi}_{k}"}
        manifest = tmp_path / "manifest.tsv"
        with open(manifest, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        (tmp_path / "lineages.json").write_text(json.dumps(lineages))  # complete cache: no network

        args = argparse.Namespace(
            inputs=[], manifest=str(manifest), output=str(tmp_path / "out"), config=None, length=100,
            max_fragments=None, chunk_size=50, fragments_per_class=120, split_by="family", split_seed=7,
        )
        assert prepare_command(args) == 0
        with open(tmp_path / "out" / "split_assignments.tsv") as f:
            assignments = list(csv.DictReader(f, delimiter="\t"))
        assert {"phylum", "class", "order", "family", "genus", "group"} <= set(assignments[0])
        family_splits = {}
        for a in assignments:
            family_splits.setdefault(a["family"], set()).add(a["split"])
        assert len(family_splits) == len(NCBI_GROUP_TO_TAXON) * 3 and all(len(v) == 1 for v in family_splits.values())
        assert {a["split"] for a in assignments} == {"train", "val", "test"}
        meta = json.loads((tmp_path / "out" / "metadata.json").read_text())
        assert meta["split_by"] == "family" and meta["split_seed"] == 7 and "family-disjoint" in meta["split_method"]
