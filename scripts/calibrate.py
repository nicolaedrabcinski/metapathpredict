"""
Fit a temperature on a checkpoint's validation logits and report calibration and selective-prediction
(abstention) numbers on the test split.

    python scripts/calibrate.py --model experiments/baselines/final_ensemble_s1/ensemble.pt \
        --data-dir data/datasets/taxa8vir_fam500_s1

Writes <checkpoint's directory>/calibration.json (temperature, ECE before/after, the risk-coverage
curve by fragment and by genome) and prints a short table: at a few coverage levels, what accuracy do
you get if the model is allowed to abstain below the matching confidence threshold?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from metapathpredict import calibration
from metapathpredict.cli import _load_model_from_checkpoint
from metapathpredict.genome_eval import TEST_FASTA, read_fragment_accessions


def _loader(x: np.ndarray, y: np.ndarray, batch_size: int = 512):
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y).long())
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="a scripts/baselines.py checkpoint (model.pt or an ensemble.pt)")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--fragment-size", type=int, default=500)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    model, model_type = _load_model_from_checkpoint(Path(args.model), torch.device(args.device))
    if model_type != "supervised":
        raise SystemExit(f"{args.model}: model_type={model_type!r}, calibrate.py expects a scripts/baselines.py "
                         "checkpoint (plain or ensemble)")
    with h5py.File(data_dir / f"encoded_val_{args.fragment_size}.hdf5") as f:
        x_val, y_val = f["sequences"][:], f["labels"][:]
    with h5py.File(data_dir / f"encoded_test_{args.fragment_size}.hdf5") as f:
        x_test, y_test = f["sequences"][:], f["labels"][:]
    genomes = read_fragment_accessions(data_dir / TEST_FASTA)

    val_logits, val_targets = calibration.collect_logits(model, _loader(x_val, y_val), args.device)
    temperature = calibration.fit_temperature(val_logits, val_targets)
    print(f"temperature (fit on val): {temperature:.3f}")

    test_logits, test_targets = calibration.collect_logits(model, _loader(x_test, y_test), args.device)
    raw = calibration.calibrated_probabilities(test_logits, 1.0)
    calibrated = calibration.calibrated_probabilities(test_logits, temperature)
    ece_raw = calibration.expected_calibration_error(raw, test_targets)
    ece_calibrated = calibration.expected_calibration_error(calibrated, test_targets)
    print(f"test ECE: raw {ece_raw:.4f} -> calibrated {ece_calibrated:.4f}")
    assert torch.equal(raw.argmax(dim=1), calibrated.argmax(dim=1)), "temperature scaling must not change predictions"

    fragment_curve = calibration.risk_coverage_curve(calibrated, test_targets)
    genome_curve = calibration.genome_risk_coverage(calibrated, test_targets, genomes)

    print(f"\n{'coverage':>10s} {'fragment accuracy':>18s} {'genome accuracy':>16s}")
    for cov_target in (1.0, 0.9, 0.75, 0.5, 0.25):
        i = min(range(len(fragment_curve["coverage"])), key=lambda i: abs(fragment_curve["coverage"][i] - cov_target))
        j = min(range(len(genome_curve["coverage"])), key=lambda i: abs(genome_curve["coverage"][i] - cov_target))
        print(f"{fragment_curve['coverage'][i]:10.2f} {fragment_curve['accuracy'][i]:18.3f} {genome_curve['accuracy'][j]:16.3f}"
              f"   (genome coverage {genome_curve['coverage'][j]:.2f})")

    out = Path(args.model).parent / "calibration.json"
    out.write_text(json.dumps({
        "temperature": temperature, "ece_raw": ece_raw, "ece_calibrated": ece_calibrated,
        "fragment_risk_coverage": fragment_curve, "genome_risk_coverage": genome_curve,
    }, indent=2))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
