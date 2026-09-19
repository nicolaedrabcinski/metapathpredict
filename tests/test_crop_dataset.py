"""Cropping a prepared dataset to shorter fragments."""

import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from metapathpredict.cli import _SplitWriter

L = 40


def _module():
    spec = importlib.util.spec_from_file_location("crop", Path(__file__).resolve().parents[1] / "scripts/crop_dataset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _onehot(seq):
    x = np.zeros((4, len(seq)), dtype=np.float32)
    for i, base in enumerate(seq):
        x["ACGT".index(base), i] = 1
    return x


@pytest.fixture
def dataset(tmp_path):
    rng = np.random.default_rng(0)
    src = tmp_path / "src"
    src.mkdir()
    sizes = {"train": 30, "val": 8, "test": 10}
    sequences = {}
    for split, n in sizes.items():
        seqs = ["".join(rng.choice(list("ACGT"), L)) for _ in range(n)]
        sequences[split] = seqs
        writer = _SplitWriter(src / f"encoded_{split}_{L}.hdf5", L, 16, {"num_classes": 2, "class_names": "[]", "sequence_length": L})
        for i, s in enumerate(seqs):
            writer.append(_onehot(s), i % 2)
        writer.close()
    with open(src / "test_fragments.fasta", "w") as f:
        for i, s in enumerate(sequences["test"]):
            f.write(f">c_{i}|label={i % 2}|acc=G{i // 5}\n{s}\n")
    (src / "split_assignments.tsv").write_text("accession\tclass\tsplit\nG0\tbacteria\ttest\n")
    (src / "metadata.json").write_text(json.dumps({"sequence_length": L, "train_size": 30, "val_size": 8, "test_size": 10}))
    return src, sequences


def test_every_fragment_becomes_one_of_its_halves_and_the_fasta_follows(dataset, tmp_path):
    src, sequences = dataset
    dst = tmp_path / "dst"
    meta = _module().crop_dataset(src, dst, L // 2, seed=3)
    assert meta["sequence_length"] == L // 2 and meta["cropped_from"] == L
    halves_seen = set()
    for split, seqs in sequences.items():
        with h5py.File(dst / f"encoded_{split}_{L // 2}.hdf5") as f:
            x, y = f["sequences"][:], f["labels"][:]
            assert x.shape == (len(seqs), 4, L // 2) and f.attrs["sequence_length"] == L // 2
        assert y.tolist() == [i % 2 for i in range(len(seqs))]
        for row, seq in zip(x, seqs):
            left, right = _onehot(seq[: L // 2]), _onehot(seq[L // 2:])
            is_left, is_right = np.array_equal(row, left), np.array_equal(row, right)
            assert is_left or is_right
            halves_seen.add("left" if is_left else "right")
    assert halves_seen == {"left", "right"}  # not always the same half

    records = [l.rstrip() for l in open(dst / "test_fragments.fasta")]
    headers, cropped = records[0::2], records[1::2]
    assert headers == [f">c_{i}|label={i % 2}|acc=G{i // 5}" for i in range(10)]
    with h5py.File(dst / f"encoded_test_{L // 2}.hdf5") as f:
        for row, text in zip(f["sequences"][:], cropped):
            assert np.array_equal(row, _onehot(text))  # the FASTA and the HDF5 hold the same window
    assert (dst / "split_assignments.tsv").read_text() == (src / "split_assignments.tsv").read_text()


def test_a_length_that_does_not_divide_the_source_is_rejected(dataset, tmp_path):
    with pytest.raises(ValueError):
        _module().crop_dataset(dataset[0], tmp_path / "x", 30)
