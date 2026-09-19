"""
Make a shorter-fragment copy of a prepared dataset by cropping every fragment to one random window.

Cheaper than running `prepare` again (which re-reads every genome): a 1000 bp fragment becomes one of
its two 500 bp halves, chosen at random per fragment, so the windows stay uniform along the genomes and
the number of fragments per genome is unchanged. split_assignments.tsv is copied as is; test_fragments.fasta
is cropped at the same positions, so genome-level evaluation keeps working.

    python scripts/crop_dataset.py data/datasets/taxa8fam_s1 data/datasets/taxa8fam500_s1 --length 500
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np

from metapathpredict.cli import _SplitWriter

SPLITS = ("train", "val", "test")


def crop_dataset(src: Path, dst: Path, length: int, seed: int = 0, chunk: int = 20000) -> dict:
    src, dst = Path(src), Path(dst)
    meta = json.loads((src / "metadata.json").read_text())
    source_length = meta["sequence_length"]
    if source_length % length:
        raise ValueError(f"{length} does not divide the source fragment length {source_length}")
    rng = np.random.default_rng(seed)
    dst.mkdir(parents=True, exist_ok=True)

    starts_by_split = {}
    for split in SPLITS:
        with h5py.File(src / f"encoded_{split}_{source_length}.hdf5") as f:
            attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
            attrs["sequence_length"] = length
            n = f["labels"].shape[0]
            starts = rng.integers(0, source_length // length, size=n) * length
            writer = _SplitWriter(dst / f"encoded_{split}_{length}.hdf5", length, 10000, attrs)
            for lo in range(0, n, chunk):
                block, labels = f["sequences"][lo:lo + chunk], f["labels"][lo:lo + chunk]
                for row, label, start in zip(block, labels, starts[lo:lo + chunk]):
                    writer.append(row[:, start:start + length], int(label))
            writer.close()
        starts_by_split[split] = starts

    # test_fragments.fasta, cropped at the same positions and in the same order
    starts, index = starts_by_split["test"], 0
    with open(src / "test_fragments.fasta") as fin, open(dst / "test_fragments.fasta", "w") as fout:
        header = None
        for line in fin:
            line = line.rstrip("\n")
            if line.startswith(">"):
                header = line
            else:
                fout.write(f"{header}\n{line[starts[index]:starts[index] + length]}\n")
                index += 1
    if index != len(starts):
        raise ValueError(f"test_fragments.fasta has {index} records, the test split {len(starts)}")

    shutil.copy(src / "split_assignments.tsv", dst / "split_assignments.tsv")
    meta.update({"sequence_length": length, "cropped_from": source_length, "crop_seed": seed})
    (dst / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--length", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    meta = crop_dataset(args.src, args.dst, args.length, args.seed)
    print(f"{args.dst}: {meta['train_size']} train / {meta['val_size']} val / {meta['test_size']} test fragments of {args.length} bp")


if __name__ == "__main__":
    main()
