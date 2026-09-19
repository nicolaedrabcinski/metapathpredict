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
        assert {"lineage_phylum", "lineage_class", "lineage_order", "lineage_family", "lineage_genus", "group"} <= set(assignments[0])
        assert {a["class"] for a in assignments} <= set(TAXON_CLASSES)  # the 8-class label must survive
        family_splits = {}
        for a in assignments:
            family_splits.setdefault(a["lineage_family"], set()).add(a["split"])
        assert len(family_splits) == len(NCBI_GROUP_TO_TAXON) * 3 and all(len(v) == 1 for v in family_splits.values())
        assert {a["split"] for a in assignments} == {"train", "val", "test"}
        # the genome of every fragment can be rebuilt from the assignments alone
        import h5py

        from metapathpredict.relatedness import fragment_genomes, lineage_targets

        out = tmp_path / "out"
        fasta_accessions = [l.strip().split("|acc=")[1] for l in open(out / "test_fragments.fasta") if l.startswith(">")]
        assert fragment_genomes(out, "test").tolist() == fasta_accessions
        names = json.loads((out / "metadata.json").read_text())["class_names"]
        by_accession = {a["accession"]: a["class"] for a in assignments}
        for split in ("train", "val", "test"):
            genomes = fragment_genomes(out, split)
            with h5py.File(out / f"encoded_{split}_100.hdf5") as f:
                labels = f["labels"][:]
            assert len(genomes) == len(labels)
            assert [names.index(by_accession[g]) for g in genomes] == labels.tolist()
        targets, phyla = lineage_targets(out, "train", "family")
        assert len(targets) == len(fragment_genomes(out, "train")) and (targets >= 0).all() and len(phyla) > 1
        meta = json.loads((tmp_path / "out" / "metadata.json").read_text())
        assert meta["split_by"] == "family" and meta["split_seed"] == 7 and "family-disjoint" in meta["split_method"]


class TestFixedSplits:
    RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}

    def test_fixed_genomes_keep_their_split_and_their_whole_family_follows(self):
        from metapathpredict.cli import _assign_group_splits

        groups = ["a", "a", "b", "c", "c", "d", "e", "f", "g", "h"]
        fixed = ["test", None, None, "val", None, None, None, None, None, None]      # "a" is anchored in test, "c" in val
        splits = _assign_group_splits(groups, self.RATIOS, np.random.RandomState(0), fixed=fixed)
        assert splits[0] == "test" and splits[1] == "test"                            # the new member of "a" follows
        assert splits[3] == "val" and splits[4] == "val"
        by_group = {}
        for g, s in zip(groups, splits):
            by_group.setdefault(g, set()).add(s)
        assert all(len(v) == 1 for v in by_group.values())                           # still family-disjoint

    def test_new_families_fill_the_splits_that_are_short(self):
        from metapathpredict.cli import _assign_group_splits

        groups = [f"old{i}" for i in range(6)] + [f"new{i}" for i in range(14)]
        fixed = ["train"] * 6 + [None] * 14                                          # six anchored train genomes, 20 in total
        splits = _assign_group_splits(groups, self.RATIOS, np.random.RandomState(0), fixed=fixed)
        assert set(splits[:6]) == {"train"}
        counts = {s: splits.count(s) for s in self.RATIOS}
        assert counts["val"] >= 2 and counts["test"] >= 2 and abs(counts["train"] - 14) <= 2

    def test_conflicting_anchors_are_an_error(self):
        from metapathpredict.cli import _assign_group_splits

        with pytest.raises(ValueError):
            _assign_group_splits(["a", "a", "b"], self.RATIOS, np.random.RandomState(0), fixed=["train", "test", None])

    def test_without_fixed_splits_nothing_changes(self):
        from metapathpredict.cli import _assign_group_splits

        groups = [f"f{i}" for i in range(20) for _ in range(2)]
        a = _assign_group_splits(groups, self.RATIOS, np.random.RandomState(3))
        b = _assign_group_splits(groups, self.RATIOS, np.random.RandomState(3), fixed=[None] * len(groups))
        assert a == b

    def test_prepare_with_fixed_splits_reproduces_the_earlier_assignment(self, tmp_path):
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
                lineages[taxid] = {"phylum": "P", "class": "C", "order": f"O{gi}", "family": f"F{gi}_{k // 2}", "genus": f"G{gi}_{k}"}
        (tmp_path / "lineages.json").write_text(json.dumps(lineages))

        def write_manifest(path, subset):
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
                w.writeheader()
                w.writerows(subset)

        def prepare(manifest, out, fixed=None, seed=5):
            return prepare_command(argparse.Namespace(
                inputs=[], manifest=str(manifest), output=str(out), config=None, length=100, max_fragments=None,
                chunk_size=50, fragments_per_class=120, split_by="family", split_seed=seed, fixed_splits=fixed))

        old_rows = [r for r in rows if not r["accession"].endswith(("4.1", "5.1"))]            # 4 of 6 genomes per class
        write_manifest(tmp_path / "old.tsv", old_rows)
        write_manifest(tmp_path / "all.tsv", rows)
        assert prepare(tmp_path / "old.tsv", tmp_path / "old") == 0
        assert prepare(tmp_path / "all.tsv", tmp_path / "new", fixed=tmp_path / "old" / "split_assignments.tsv", seed=99) == 0
        read = lambda d: {a["accession"]: a for a in csv.DictReader(open(tmp_path / d / "split_assignments.tsv"), delimiter="\t")}
        old, new = read("old"), read("new")
        assert len(new) == len(rows) > len(old)
        assert all(new[acc]["split"] == a["split"] for acc, a in old.items())                  # old genomes stayed put
        family_splits = {}
        for a in new.values():
            family_splits.setdefault(a["lineage_family"], set()).add(a["split"])
        assert all(len(v) == 1 for v in family_splits.values())                                # and families stay together
