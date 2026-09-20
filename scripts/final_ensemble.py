"""
Evaluate an ensemble: average the test-set probabilities of several saved checkpoints (from
scripts/baselines.py supervised --save-checkpoint) and score the result with a genome-level bootstrap,
plus a paired comparison against one or more reference prediction files (e.g. the pre-improvement baseline).

    python scripts/final_ensemble.py --data-dir data/datasets/taxa8vir_fam500_s1 \
        --models final_s1_seed0 final_s1_seed1 final_s1_seed2 \
        --reference experiments/baselines/ce_rc_fam_s1_seed1/test_predictions.npy=baseline (no rc-share, no extra virus)

Writes experiments/baselines/<--out>/ with test.json, test_genomes.json and test_predictions.npy, like a
single baselines.py run, so it can be reused (e.g. as a --reference for another split).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from metapathpredict.baselines import load_supervised_checkpoint
from metapathpredict.cli import _evaluation_report
from metapathpredict.genome_eval import (
    TEST_FASTA,
    evaluate_by_genome,
    format_genome_report,
    paired_difference,
    read_fragment_accessions,
)

ROOT = Path(__file__).resolve().parents[1]


def cnn_probabilities(model, x: np.ndarray, batch: int = 512) -> np.ndarray:
    out = []
    with torch.no_grad():
        for lo in range(0, len(x), batch):
            out.append(torch.softmax(model(torch.from_numpy(x[lo:lo + batch])), dim=1))
    return torch.cat(out).numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--fragment-size", type=int, default=500)
    ap.add_argument("--models", nargs="+", required=True, help="names in experiments/baselines with a model.pt")
    ap.add_argument("--reference", nargs="*", default=[], metavar="LABEL=PATH",
                    help="prediction .npy files to compare against (paired, over the same test genomes)")
    ap.add_argument("--out", default=None, help="write experiments/baselines/<out>/ (default: ensemble_<data-dir name>)")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    data_dir = Path(args.data_dir)
    names = json.loads((data_dir / "metadata.json").read_text())["class_names"]
    with h5py.File(data_dir / f"encoded_test_{args.fragment_size}.hdf5") as f:
        x_test, y_test = f["sequences"][:], f["labels"][:]
    genomes = read_fragment_accessions(data_dir / TEST_FASTA)

    probs = []
    for name in args.models:
        model = load_supervised_checkpoint(ROOT / "experiments" / "baselines" / name / "model.pt")
        p = cnn_probabilities(model, x_test)
        print(f"{name:24s} single-model accuracy {100 * (p.argmax(1) == y_test).mean():.1f}%", flush=True)
        probs.append(p)
    ensemble = np.mean(probs, axis=0)
    preds = ensemble.argmax(1)

    report = _evaluation_report(names, y_test, preds)
    genome = evaluate_by_genome(data_dir, y_test, preds, names)
    print(f"\nEnsemble of {len(args.models)}: accuracy {report['accuracy']:.4f}"
          + (f", 3-class {report['aggregated_3class']['accuracy']:.4f}" if report['aggregated_3class'] else ""))
    if genome:
        print(format_genome_report(genome))

    out_name = args.out or f"ensemble_{data_dir.name}"
    out_dir = ROOT / "experiments" / "baselines" / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test.json").write_text(json.dumps(report, indent=2))
    np.save(out_dir / "test_predictions.npy", preds.astype(np.int8))
    np.save(out_dir / "test_probabilities.npy", ensemble.astype(np.float16))
    if genome:
        (out_dir / "test_genomes.json").write_text(json.dumps(genome, indent=2))

    for ref in args.reference:
        label, path = ref.split("=", 1)
        ref_preds = np.load(path).astype(int)
        if len(ref_preds) != len(y_test):
            print(f"\n{label}: skipped, {len(ref_preds)} predictions for a test split of {len(y_test)} fragments")
            continue
        diff = paired_difference(y_test, ref_preds, preds, genomes, names, n_boot=1000)
        def f(d):
            return f"{100 * d['value']:+.1f} [{100 * d['ci_low']:+.1f}, {100 * d['ci_high']:+.1f}]"
        print(f"\nensemble vs {label} (paired over genomes): accuracy {f(diff['accuracy'])}"
              + (f", 3-class {f(diff['accuracy_3class'])}" if "accuracy_3class" in diff else "")
              + f", P(ensemble better)={diff['p_b_better']:.2f}")


if __name__ == "__main__":
    main()
