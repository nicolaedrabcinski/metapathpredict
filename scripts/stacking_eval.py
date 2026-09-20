"""
Does a meta-classifier over several models beat plain averaging?

Base models: CNNs saved by `scripts/baselines.py supervised --save-checkpoint` (experiments/baselines/<name>/model.pt)
and, with --gbm, a gradient-boosted 4-mer model trained here. Probabilities on the validation and test splits are cached
in experiments/stacking/<dataset>/. The meta-classifier is fit on the validation split (its families are new to the
base models) and scored on the test split; every score has a genome-level bootstrap interval and the paired difference
to plain averaging of the CNNs.

    python scripts/stacking_eval.py --data-dir data/datasets/taxa8fam500_s1 --gbm \
        --models ce_rc_fam_s1_seed1 ce_rc_fam_s1_seed2 ce_mildaug_fam_s1 ce_rc_cos50_fam_s1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from metapathpredict import stacking
from metapathpredict.baselines import kmer_frequencies, load_supervised_checkpoint
from metapathpredict.genome_eval import (
    TEST_FASTA,
    genome_report,
    paired_difference,
    read_fragment_accessions,
)

ROOT = Path(__file__).resolve().parents[1]


def load_split(data_dir: Path, split: str, length: int):
    with h5py.File(data_dir / f"encoded_{split}_{length}.hdf5") as f:
        return f["sequences"][:], f["labels"][:]


def cnn_probabilities(model, x: np.ndarray, batch: int = 512) -> np.ndarray:
    out = []
    with torch.no_grad():
        for lo in range(0, len(x), batch):
            out.append(torch.softmax(model(torch.from_numpy(x[lo:lo + batch])), dim=1))
    return torch.cat(out).numpy()


def cached(path: Path, compute):
    if path.exists():
        return np.load(path).astype(np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = compute()
    np.save(path, value.astype(np.float16))
    return value.astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/datasets/taxa8fam500_s1")
    ap.add_argument("--fragment-size", type=int, default=500)
    ap.add_argument("--models", nargs="+", required=True, help="names in experiments/baselines with a model.pt")
    ap.add_argument("--gbm", action="store_true", help="add a gradient-boosted 4-mer model (trained here, ~3 min)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    data_dir = Path(args.data_dir)
    cache = ROOT / "experiments" / "stacking" / data_dir.name
    names = json.loads((data_dir / "metadata.json").read_text())["class_names"]
    x_val, y_val = load_split(data_dir, "val", args.fragment_size)
    x_test, y_test = load_split(data_dir, "test", args.fragment_size)
    genomes = read_fragment_accessions(data_dir / TEST_FASTA)

    val, test = {}, {}
    for name in args.models:
        model = load_supervised_checkpoint(ROOT / "experiments" / "baselines" / name / "model.pt")
        val[name] = cached(cache / f"{name}_val.npy", lambda: cnn_probabilities(model, x_val))
        test[name] = cached(cache / f"{name}_test.npy", lambda: cnn_probabilities(model, x_test))
        print(f"{name:34s} val {100 * (val[name].argmax(1) == y_val).mean():5.1f}%  test {100 * (test[name].argmax(1) == y_test).mean():5.1f}%", flush=True)
    cnn_names = list(args.models)
    if args.gbm:
        from sklearn.ensemble import HistGradientBoostingClassifier

        x_train, y_train = load_split(data_dir, "train", args.fragment_size)
        fitted = {}

        def gbm(split_x):
            if "model" not in fitted:
                fitted["model"] = HistGradientBoostingClassifier(max_iter=200, early_stopping=False, random_state=0).fit(kmer_frequencies(x_train, 4), y_train)
            return fitted["model"].predict_proba(kmer_frequencies(split_x, 4))

        val["gbm4"] = cached(cache / "gbm4_val.npy", lambda: gbm(x_val))
        test["gbm4"] = cached(cache / "gbm4_test.npy", lambda: gbm(x_test))
        print(f"{'gbm4':34s} val {100 * (val['gbm4'].argmax(1) == y_val).mean():5.1f}%  test {100 * (test['gbm4'].argmax(1) == y_test).mean():5.1f}%", flush=True)
    all_names = cnn_names + (["gbm4"] if args.gbm else [])

    def score(probs):
        report = genome_report(y_test, probs.argmax(1), genomes, names, n_boot=1000)
        return report["accuracy"], report["accuracy_3class"]

    candidates = {"average of CNNs": stacking.average_probabilities([test[n] for n in cnn_names])}
    if args.gbm:
        candidates["average of CNNs + gbm4"] = stacking.average_probabilities([test[n] for n in all_names])
    for kind, balance in (("logreg", False), ("logreg", True), ("forest", True)):
        meta = stacking.fit_meta([val[n] for n in all_names], y_val, kind=kind, balance_hard=balance)
        label = f"meta {kind}{' + hard balance' if balance else ''} (fit on val)"
        candidates[label] = stacking.predict_meta(meta, [test[n] for n in all_names], len(names))

    reference = candidates["average of CNNs"].argmax(1)
    def f(x):
        return f"{100 * x['value']:.1f}% [{100 * x['ci_low']:.1f}, {100 * x['ci_high']:.1f}]"
    print(f"\n{'test, 8 classes':44s} {'accuracy [95% by genomes]':>26s} {'3-class':>22s} {'vs average of CNNs (paired)':>30s}")
    results = {}
    for label, probs in candidates.items():
        a8, a3 = score(probs)
        diff = paired_difference(y_test, reference, probs.argmax(1), genomes, names, n_boot=1000)["accuracy"]
        d = f"{100 * diff['value']:+.1f} [{100 * diff['ci_low']:+.1f}, {100 * diff['ci_high']:+.1f}]" if label != "average of CNNs" else "-"
        print(f"{label:44s} {f(a8):>26s} {f(a3):>22s} {d:>30s}")
        results[label] = {"accuracy": a8, "accuracy_3class": a3, "diff_vs_average": None if d == "-" else diff}
    if args.out:
        Path(args.out).write_text(json.dumps({"models": all_names, "results": results}, indent=2))


if __name__ == "__main__":
    main()
