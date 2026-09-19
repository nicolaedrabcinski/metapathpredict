"""
Refit the linear probe of contrastive checkpoints on frozen, eval-mode embeddings and score them on
the test split, so runs made with the earlier probe (backbone in train mode) can be compared fairly.

The original checkpoint is left untouched; the refit one is written next to it as
contrastive_frozen_probe.pt together with test_frozen_probe.json.

    python scripts/refit_probe.py data/weights/taxa8/contrastive_best.pt
    python scripts/refit_probe.py experiments/bs512/*/weights/contrastive_best.pt --mlflow
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from metapathpredict.cli import _evaluation_report, _load_model_from_checkpoint, _report_to_metrics
from metapathpredict.data.datamodule import _open_split
from metapathpredict.experiment_tracking import MLflowSink, NullSink
from metapathpredict.genome_eval import evaluate_by_genome, genome_report_to_metrics
from metapathpredict.probe import cache_embeddings, fit_linear_probe


def _previous_test_report(checkpoint: Path) -> dict | None:
    for name in ("test_contrastive.json", "test_eval.json"):
        candidate = checkpoint.parent.parent / name
        if candidate.exists():
            return json.loads(candidate.read_text())
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--data-dir", default="data/datasets/taxa8")
    ap.add_argument("--fragment-size", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--mlflow", action="store_true", help="log each refit as a run in experiment 'refit_probe'")
    args = ap.parse_args()

    device = torch.device(args.device)
    data_dir = Path(args.data_dir)
    splits = {s: _open_split(data_dir / f"encoded_{s}_{args.fragment_size}.hdf5") for s in ("train", "val", "test")}
    loaders = {s: DataLoader(ds, batch_size=512, shuffle=False, num_workers=0) for s, ds in splits.items()}

    print(f"{'checkpoint':58s} {'val':>6s} {'test 8c':>8s} {'95% CI (genomes)':>17s} {'(was)':>7s} {'test 3c':>8s} {'(was)':>7s} {'virus':>6s}")
    for path in map(Path, args.checkpoints):
        torch.manual_seed(0)
        model, _ = _load_model_from_checkpoint(path, device)
        probe = fit_linear_probe(model.encoder, loaders["train"], loaders["val"], device, epochs=args.epochs)

        x_test, y_test = cache_embeddings(model.encoder, loaders["test"], device)
        with torch.no_grad():
            preds = model.encoder.classifier(x_test).argmax(dim=1).cpu().numpy()
        report = _evaluation_report(list(model.class_names), y_test.cpu().numpy(), preds)
        genome = evaluate_by_genome(data_dir, y_test.cpu().numpy(), preds, list(model.class_names))

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint["encoder_state_dict"] = model.state_dict()
        checkpoint["probe_val_acc"] = probe["best_val_acc"]
        checkpoint["probe_mode"] = "frozen"
        torch.save(checkpoint, path.parent / "contrastive_frozen_probe.pt")
        (path.parent.parent / "test_frozen_probe.json").write_text(json.dumps(report, indent=2))
        if genome:
            (path.parent.parent / "test_frozen_probe_genomes.json").write_text(json.dumps(genome, indent=2))

        before = _previous_test_report(path)
        agg = report["aggregated_3class"]["accuracy"]
        was8 = f"{before['accuracy']:.3f}" if before else "-"
        was3 = f"{before['aggregated_3class']['accuracy']:.3f}" if before and before.get("aggregated_3class") else "-"
        virus = report["classification_report"]["virus"]["recall"]
        ci = f"[{genome['accuracy']['ci_low']:.3f}, {genome['accuracy']['ci_high']:.3f}]" if genome else "-"
        print(f"{str(path)[-58:]:58s} {probe['best_val_acc']:6.3f} {report['accuracy']:8.3f} {ci:>17s} {was8:>7s} {agg:8.3f} {was3:>7s} {virus:6.3f}"
              f"   (best epoch {probe['best_epoch']}/{probe['epochs_run']})")

        if args.mlflow:
            sink = MLflowSink("refit_probe", run_name=str(path.parent.parent.name), system_metrics=False,
                              tags={"source_checkpoint": str(path)})
        else:
            sink = NullSink()
        with sink:
            sink.log_params({"probe_epochs": args.epochs, "source_checkpoint": str(path)})
            sink.log_metrics({"probe/best_val_acc": probe["best_val_acc"], "probe/epochs_run": probe["epochs_run"]})
            sink.log_metrics(_report_to_metrics(report, "test/contrastive_frozen"))
            if genome:
                sink.log_metrics(genome_report_to_metrics(genome, "test/contrastive_frozen"))


if __name__ == "__main__":
    main()
