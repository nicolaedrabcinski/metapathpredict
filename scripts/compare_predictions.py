"""
Is model B really better than model A on the species-disjoint test split?

Compares two saved prediction files (int class ids, one per test fragment, HDF5 order) with a paired
bootstrap over genomes. The predictions are written by scripts/refit_probe.py
(test_frozen_probe_predictions.npy) and scripts/baselines.py (test_predictions.npy).

    python scripts/compare_predictions.py experiments/abl_aug_sample/*/test_frozen_probe_predictions.npy \
        experiments/new_dcl/*/test_frozen_probe_predictions.npy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from metapathpredict.genome_eval import TEST_FASTA, paired_difference, read_fragment_accessions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="predictions of the reference model A")
    ap.add_argument("b", nargs="+", help="predictions of one or more models B, each compared with A")
    ap.add_argument("--data-dir", default="data/datasets/taxa8")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    class_names = json.loads((data_dir / "metadata.json").read_text())["class_names"]
    with h5py.File(data_dir / "encoded_test_500.hdf5") as f:
        targets = f["labels"][:]
    genomes = read_fragment_accessions(data_dir / TEST_FASTA)
    a = np.load(args.a).astype(int)

    def fmt(x: dict) -> str:
        return f"{x['value']:+.3f} [{x['ci_low']:+.3f}, {x['ci_high']:+.3f}]"

    print(f"A = {args.a}  (accuracy {(a == targets).mean():.3f})")
    print(f"{'B':70s} {'acc':>7s} {'diff accuracy (B-A)':>26s} {'diff 3-class':>26s} {'P(B>A)':>7s}")
    for path in args.b:
        b = np.load(path).astype(int)
        diff = paired_difference(targets, a, b, genomes, class_names, n_boot=args.n_boot)
        print(f"{path[-70:]:70s} {(b == targets).mean():7.3f} {fmt(diff['accuracy']):>26s} "
              f"{fmt(diff['accuracy_3class']) if 'accuracy_3class' in diff else '-':>26s} {diff['p_b_better']:7.2f}")


if __name__ == "__main__":
    main()
