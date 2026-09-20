"""
Reference baselines, scored exactly like the contrastive runs: species-disjoint test split, genome-level
bootstrap interval, MLflow experiment "baselines".

    python scripts/baselines.py kmer --k 4                       # logistic regression + gradient boosting
    python scripts/baselines.py supervised --augment rc          # same CNN as the encoder, plain cross-entropy
    python scripts/baselines.py supervised --augment full --epochs 20 --name ce_full_aug

Results are also written to experiments/baselines/<name>/ (test.json, test_genomes.json).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from metapathpredict.baselines import (
    WithAuxTargets,
    build_classifier,
    kmer_frequencies,
    load_backbone_weights,
    predict_probabilities,
    save_supervised_checkpoint,
    train_supervised,
)
from metapathpredict.cli import _evaluation_report, _report_to_metrics
from metapathpredict.data.datamodule import _open_split
from metapathpredict.experiment_tracking import MLflowSink
from metapathpredict.genome_eval import (
    evaluate_by_genome,
    format_genome_report,
    genome_report_to_metrics,
)
from metapathpredict.models.contrastive import ContrastiveAugmentation

REPO_ROOT = Path(__file__).resolve().parents[1]
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("baselines")


def _finish(sink: MLflowSink, out_dir: Path, data_dir: Path, class_names: list[str], targets, preds, extra: dict,
            probs=None) -> None:
    report = _evaluation_report(class_names, targets, preds)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test.json").write_text(json.dumps(report, indent=2))
    np.save(out_dir / "test_predictions.npy", np.asarray(preds, dtype=np.int8))
    if probs is not None:  # class probabilities of every test fragment, for aggregating fragments of one genome
        np.save(out_dir / "test_probabilities.npy", np.asarray(probs, dtype=np.float16))
    sink.log_metrics(_report_to_metrics(report, "test/baseline"))
    genome = evaluate_by_genome(data_dir, targets, preds, class_names)
    if genome:
        (out_dir / "test_genomes.json").write_text(json.dumps(genome, indent=2))
        sink.log_metrics(genome_report_to_metrics(genome, "test/baseline"))
        logger.info("\n" + format_genome_report(genome))
    sink.log_metrics(extra)
    sink.log_artifact(out_dir)
    agg = report["aggregated_3class"]
    logger.info(f"TEST accuracy {report['accuracy']:.4f}" + (f", 3-class {agg['accuracy']:.4f}" if agg else ""))

    meta = json.loads((data_dir / "metadata.json").read_text())
    if "merge_to_original" in meta:  # a relabelled dataset: also score it in the classes it was made from
        merge = np.asarray(meta["merge_to_original"])
        original = meta["original_class_names"]
        merged = _evaluation_report(original, merge[targets], merge[preds])
        (out_dir / "test_merged.json").write_text(json.dumps(merged, indent=2))
        np.save(out_dir / "test_predictions_merged.npy", merge[preds].astype(np.int8))  # original class ids
        sink.log_metrics(_report_to_metrics(merged, "test/baseline_merged"))
        merged_genome = evaluate_by_genome(data_dir, merge[targets], merge[preds], original)
        if merged_genome:
            (out_dir / "test_merged_genomes.json").write_text(json.dumps(merged_genome, indent=2))
            sink.log_metrics(genome_report_to_metrics(merged_genome, "test/baseline_merged"))
        m3 = merged["aggregated_3class"]
        logger.info(f"TEST merged back to {len(original)} original classes: accuracy {merged['accuracy']:.4f}"
                    + (f", 3-class {m3['accuracy']:.4f}" if m3 else ""))


def run_kmer(args, data_dir: Path, class_names: list[str]) -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    arrays = {}
    for split in ("train", "val", "test"):
        with h5py.File(data_dir / f"encoded_{split}_{args.fragment_size}.hdf5") as f:
            arrays[split] = (kmer_frequencies(f["sequences"][:], args.k), f["labels"][:])
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = arrays["train"], arrays["val"], arrays["test"]

    for model_name in args.models:
        name = args.name or f"kmer{args.k}_{model_name}"
        start = time.time()
        if model_name == "logreg":
            scaler = StandardScaler().fit(x_train)
            model = LogisticRegression(max_iter=300).fit(scaler.transform(x_train), y_train)
            def predict(x):
                return model.predict(scaler.transform(x))
            def predict_proba(x):
                return model.predict_proba(scaler.transform(x))
        else:
            model = HistGradientBoostingClassifier(max_iter=args.iterations, early_stopping=False, random_state=args.seed)
            model.fit(x_train, y_train)
            predict, predict_proba = model.predict, model.predict_proba
        val_acc = float((predict(x_val) == y_val).mean())
        logger.info(f"{name}: val accuracy {val_acc:.4f} ({time.time() - start:.0f}s)")
        with MLflowSink("baselines", run_name=name, tracking_uri=f"sqlite:///{REPO_ROOT / 'mlflow.db'}",
                        tags={"baseline": "kmer"}, system_metrics=False) as sink:
            sink.log_params({"model": model_name, "k": args.k, "iterations": args.iterations, "seed": args.seed,
                             "features": int(4 ** args.k), "dataset": str(data_dir)})
            _finish(sink, REPO_ROOT / "experiments" / "baselines" / name, data_dir, class_names, y_test,
                    predict(x_test), {"baseline/val_accuracy": val_acc}, probs=predict_proba(x_test))


def run_supervised(args, data_dir: Path, class_names: list[str]) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    splits = {s: _open_split(data_dir / f"encoded_{s}_{args.fragment_size}.hdf5") for s in ("train", "val", "test")}
    aux_head = None
    if args.aux_rank:
        from metapathpredict.relatedness import lineage_targets

        aux, aux_names = lineage_targets(data_dir, "train", args.aux_rank)
        splits["train"] = WithAuxTargets(splits["train"], aux)
        logger.info(f"Auxiliary task: {args.aux_rank} of the genome, {len(aux_names)} classes, "
                    f"{int((aux == -100).sum())} of {len(aux)} training fragments unlabelled")
    train_loader = DataLoader(splits["train"], batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(splits["val"], batch_size=512, shuffle=False, num_workers=0)
    test_loader = DataLoader(splits["test"], batch_size=512, shuffle=False, num_workers=0)

    model = build_classifier(len(class_names), args.backbone, args.base_channels, args.norm, args.pool, args.rc_share,
                             args.dilation, args.gate)
    if args.aux_rank:
        aux_head = torch.nn.Linear(model._final_channels, len(aux_names))
    if args.init_from:
        loaded = load_backbone_weights(model, args.init_from)
        logger.info(f"Initialised the backbone from {args.init_from} ({loaded} tensors); classifier head is new")
    name = args.name or f"ce_{args.augment}_aug"
    augmentation = ContrastiveAugmentation(mutation_rate=args.mutation_rate, mask_rate=args.mask_rate)
    with MLflowSink("baselines", run_name=name, tracking_uri=f"sqlite:///{REPO_ROOT / 'mlflow.db'}",
                    tags={"baseline": "supervised"}) as sink:
        sink.log_params({"model": "cnn_cross_entropy", "backbone": args.backbone, "base_channels": args.base_channels,
                         "norm": args.norm, "pool": args.pool, "rc_share": args.rc_share, "dilation": args.dilation,
                         "gate": args.gate, "augment": args.augment, "lr": args.lr, "weight_decay": args.weight_decay,
                         "batch_size": args.batch_size, "epochs": args.epochs, "patience": args.patience,
                         "seed": args.seed, "mutation_rate": args.mutation_rate, "mask_rate": args.mask_rate,
                         "init_from": args.init_from or "scratch", "lr_schedule": args.lr_schedule,
                         "aux_rank": args.aux_rank or "none", "aux_weight": args.aux_weight,
                         "dataset": str(data_dir)})
        start = time.time()
        fit = train_supervised(model, train_loader, val_loader, device, epochs=args.epochs, patience=args.patience,
                               lr=args.lr, weight_decay=args.weight_decay, augment=args.augment,
                               augmentation=augmentation, sink=sink, lr_schedule=args.lr_schedule,
                               aux_head=aux_head, aux_weight=args.aux_weight)
        logger.info(f"best val accuracy {fit['best_val_acc']:.4f} at epoch {fit['best_epoch']} ({time.time() - start:.0f}s)")
        targets, probs = predict_probabilities(model, test_loader, device)
        out_dir = REPO_ROOT / "experiments" / "baselines" / name
        _finish(sink, out_dir, data_dir, class_names, targets, probs.argmax(axis=1),
                {"baseline/val_accuracy": fit["best_val_acc"], "baseline/best_epoch": fit["best_epoch"]}, probs=probs)
        if args.save_checkpoint:
            save_supervised_checkpoint(model, out_dir / "model.pt", {
                "backbone": args.backbone, "base_channels": args.base_channels, "norm": args.norm,
                "pool": args.pool, "rc_share": args.rc_share, "dilation": args.dilation, "gate": args.gate,
                "num_classes": len(class_names), "class_names": class_names, "best_epoch": fit["best_epoch"]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/datasets/taxa8")
    parser.add_argument("--fragment-size", type=int, default=500)
    parser.add_argument("--name", help="run name (default derived from the options)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-threads", type=int, default=4, help="CPU threads (shared machine; ask before >16)")
    sub = parser.add_subparsers(dest="kind", required=True)

    kmer = sub.add_parser("kmer")
    kmer.add_argument("--k", type=int, default=4)
    kmer.add_argument("--models", nargs="+", choices=["logreg", "gbm"], default=["logreg", "gbm"])
    kmer.add_argument("--iterations", type=int, default=200, help="boosting iterations")

    sup = sub.add_parser("supervised")
    sup.add_argument("--augment", choices=["none", "rc", "full"], default="rc")
    sup.add_argument("--epochs", type=int, default=20)
    sup.add_argument("--patience", type=int, default=7)
    sup.add_argument("--lr", type=float, default=1e-3)
    sup.add_argument("--weight-decay", type=float, default=1e-4)
    sup.add_argument("--batch-size", type=int, default=64)
    sup.add_argument("--backbone", default="large")
    sup.add_argument("--base-channels", type=int, default=128)
    sup.add_argument("--norm", choices=["batch", "group"], default="batch")
    sup.add_argument("--pool", choices=["avg", "max", "avgmax"], default="avg", help="reduction over positions")
    sup.add_argument("--rc-share", choices=["none", "mean", "max"], default="none",
                     help="same weights for both strands: combine the embeddings of a sequence and its reverse complement")
    sup.add_argument("--dilation", type=int, default=1,
                     help="conv dilation (same for every block): widens the receptive field without adding parameters")
    sup.add_argument("--gate", action="store_true",
                     help="Gated Linear Unit instead of ReLU in every block (roughly doubles conv parameters)")
    sup.add_argument("--init-from", help="contrastive checkpoint whose backbone initialises the CNN")
    sup.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant")
    sup.add_argument("--aux-rank", choices=["phylum", "class", "order", "family", "genus"],
                     help="auxiliary head predicting the taxonomic rank of the genome (needs lineage columns)")
    sup.add_argument("--aux-weight", type=float, default=0.3, help="weight of the auxiliary loss")
    sup.add_argument("--save-checkpoint", action="store_true", help="also save the weights as model.pt")
    sup.add_argument("--mutation-rate", type=float, default=0.3, help="augment=full")
    sup.add_argument("--mask-rate", type=float, default=0.3, help="augment=full")
    sup.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.set_num_threads(args.max_threads)
    from threadpoolctl import threadpool_limits

    data_dir = Path(args.data_dir)
    class_names = json.loads((data_dir / "metadata.json").read_text())["class_names"]
    with threadpool_limits(args.max_threads):
        (run_kmer if args.kind == "kmer" else run_supervised)(args, data_dir, class_names)


if __name__ == "__main__":
    main()
