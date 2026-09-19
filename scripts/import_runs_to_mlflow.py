"""
Import finished runs into MLflow from their logs.

Hydra runs (experiments/<name>/<timestamp>/) are read from run_experiment.log, .hydra/config.yaml and
test_*.json / test_eval.json. Root-level training logs (train.log, train_run*.log) are imported with
--main-run NAME LOG [CONTRASTIVE_JSON [RL_JSON]]. Only runs on the current dataset (taxa8) are taken
from experiments/ by default; earlier 3-class runs used a contaminated benchmark. Re-running skips runs
that are already imported.

    python scripts/import_runs_to_mlflow.py
    python scripts/import_runs_to_mlflow.py --main-run main_50ep train.log benchmarks/taxa8_50ep_contrastive_test.json benchmarks/taxa8_50ep_rl_test.json
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TS = r"(?:\[)?(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})(?:\])?"
NUM = r"(-?[\d.]+(?:e-?\d+)?)"

RE_CONTRASTIVE = re.compile(
    rf"{TS}.*\[Contrastive\] Epoch (\d+)/(\d+) \| Loss: {NUM}(?: \| Val loss: {NUM})? \| Best(?: val)?: {NUM} \| Time: {NUM}s"
)
RE_SIM = re.compile(rf"Similarity: pos={NUM}, neg={NUM}, gap={NUM}")
RE_GRAD = re.compile(rf"Gradients: avg_norm={NUM}")
RE_STD = re.compile(rf"Embedding std={NUM}")
RE_THROUGHPUT = re.compile(rf"Throughput: {NUM} samples/s")
RE_REPR = re.compile(
    rf"alignment={NUM}, uniformity={NUM}, erank\(projection\)={NUM}/\d+, erank\(backbone\)={NUM}/\d+"
)
RE_PROBE = re.compile(
    rf"Linear probe epoch (\d+)/\d+: (?:train_loss|loss)={NUM}, (?:train_acc|acc)={NUM}(?:, val_acc={NUM})?"
)
RE_RL = re.compile(
    rf"{TS}.*\[RL\] Epoch (\d+)/(\d+) \| Reward: {NUM} \| (?:Train acc|Accuracy): {NUM}(?: \| Val acc: {NUM})? \| Loss: {NUM}.*Time: {NUM}s"
)
RE_TS = re.compile(TS)


def _ms(ts: str) -> int:
    return int(datetime.strptime(ts, "%Y-%m-%d %H:%M:%S,%f").timestamp() * 1000)


def parse_log(text: str) -> tuple[list[tuple[str, float, int, int]], int | None, int | None]:
    """Return ([(metric, value, step, timestamp_ms)], first_ts_ms, last_ts_ms)."""
    metrics: list[tuple[str, float, int, int]] = []
    pending: dict[str, float] = {}
    first = last = None
    last_ts = 0
    for line in re.split(r"[\r\n]+", text):
        m = RE_TS.search(line)
        if m:
            last_ts = _ms(m.group(1))
            first = first or last_ts
            last = last_ts

        if m := RE_SIM.search(line):
            pending.update({"contrastive/pos_sim": float(m[1]), "contrastive/neg_sim": float(m[2]),
                            "contrastive/sim_gap": float(m[3])})
        elif m := RE_GRAD.search(line):
            pending["contrastive/grad_norm"] = float(m[1])
        elif m := RE_STD.search(line):
            pending["contrastive/embedding_std"] = float(m[1])
        elif m := RE_THROUGHPUT.search(line):
            pending["contrastive/samples_per_sec"] = float(m[1])
        elif m := RE_REPR.search(line):
            pending.update({"contrastive/val_alignment": float(m[1]), "contrastive/val_uniformity": float(m[2]),
                            "contrastive/val_erank_projection": float(m[3]), "contrastive/val_erank_backbone": float(m[4])})
        elif m := RE_CONTRASTIVE.search(line):
            step, ts = int(m[2]), _ms(m[1])
            metrics.append(("contrastive/train_loss", float(m[4]), step, ts))
            if m[5] is not None:
                metrics.append(("contrastive/val_loss", float(m[5]), step, ts))
            metrics.append(("contrastive/best_val_loss", float(m[6]), step, ts))
            metrics.append(("contrastive/epoch_seconds", float(m[7]), step, ts))
            metrics += [(k, v, step, ts) for k, v in pending.items()]
            pending = {}
        elif m := RE_PROBE.search(line):
            step = int(m[1])
            metrics.append(("probe/train_loss", float(m[2]), step, last_ts))
            metrics.append(("probe/train_acc", float(m[3]), step, last_ts))
            if m[4] is not None:
                metrics.append(("probe/val_acc", float(m[4]), step, last_ts))
        elif m := RE_RL.search(line):
            step, ts = int(m[2]), _ms(m[1])
            metrics.append(("rl/reward", float(m[4]), step, ts))
            metrics.append(("rl/train_acc", float(m[5]), step, ts))
            if m[6] is not None:
                metrics.append(("rl/val_acc", float(m[6]), step, ts))
            metrics.append(("rl/loss", float(m[7]), step, ts))
            metrics.append(("rl/epoch_seconds", float(m[8]), step, ts))
    return metrics, first, last


def flatten(d: dict, prefix: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = str(v)[:500]
    return out


def import_run(client, experiment: str, run_name: str, source: str, log_text: str,
               params: dict[str, str], tags: dict[str, str], eval_jsons: dict[str, Path], dry_run: bool,
               native_dir: str | None = None) -> bool:
    from mlflow.entities import Metric, Param, RunTag

    from metapathpredict.cli import _report_to_metrics

    exp = client.get_experiment_by_name(experiment)
    exp_id = exp.experiment_id if exp else client.create_experiment(experiment)
    if client.search_runs([exp_id], filter_string=f"tags.imported_from = '{source}'", max_results=1):
        print(f"  skip (already imported): {source}")
        return False
    if native_dir and client.search_runs([exp_id], filter_string=f"tags.hydra_run_dir = '{native_dir}'", max_results=1):
        print(f"  skip (logged to MLflow directly): {native_dir}")
        return False

    metrics, first, last = parse_log(log_text)
    now = int(datetime.now().timestamp() * 1000)
    present = {tag: path for tag, path in eval_jsons.items() if path.exists()}
    finished = "Best probe val accuracy" in log_text or "Total time:" in log_text
    if not (finished and present):
        why = "not finished" if not finished else "no test evaluation yet"
        print(f"  skip ({why}): {run_name}")
        return False
    for tag, path in present.items():
        for key, value in _report_to_metrics(json.loads(path.read_text()), f"test/{tag}").items():
            metrics.append((key, value, 0, last or now))
    print(f"  {run_name}: {len(metrics)} metric points, {len(params)} params, tests={list(present)}")
    if dry_run:
        return True

    run = client.create_run(exp_id, start_time=first or now, run_name=run_name,
                            tags={"imported_from": source, "imported": "true", **tags})
    rid = run.info.run_id
    ps = [Param(k, v) for k, v in params.items()]
    for i in range(0, len(ps), 90):
        client.log_batch(rid, params=ps[i:i + 90])
    ms = [Metric(k, v, ts, step) for k, v, step, ts in metrics]
    for i in range(0, len(ms), 900):
        client.log_batch(rid, metrics=ms[i:i + 900])
    client.log_batch(rid, tags=[RunTag(k, v) for k, v in tags.items()])
    client.set_terminated(rid, "FINISHED", end_time=last or now)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments-dir", default=str(REPO_ROOT / "experiments"))
    ap.add_argument("--tracking-uri", default=f"sqlite:///{REPO_ROOT / 'mlflow.db'}")
    ap.add_argument("--dataset-marker", default="taxa8",
                    help="only import Hydra runs whose datasets_dir contains this")
    ap.add_argument("--main-run", nargs="+", action="append", metavar="NAME LOG [JSON...]", default=[])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import os

    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient(args.tracking_uri)

    imported = 0
    for run_dir in sorted(Path(args.experiments_dir).glob("*/*")):
        cfg_path, log_path = run_dir / ".hydra" / "config.yaml", run_dir / "run_experiment.log"
        if not (cfg_path.exists() and log_path.exists()):
            continue
        cfg = yaml.safe_load(cfg_path.read_text())
        if args.dataset_marker not in str(cfg.get("paths", {}).get("datasets_dir", "")):
            continue
        overrides_path = run_dir / ".hydra" / "overrides.yaml"
        overrides = ",".join(yaml.safe_load(overrides_path.read_text()) or []) if overrides_path.exists() else ""
        evals = {"contrastive": run_dir / "test_contrastive.json", "rl": run_dir / "test_rl.json"}
        if (run_dir / "test_eval.json").exists() and not evals["contrastive"].exists():
            evals = {"contrastive": run_dir / "test_eval.json"}
        imported += import_run(
            client, cfg.get("experiment_name", run_dir.parent.name), f"{run_dir.name} {overrides}".strip(),
            str(run_dir.relative_to(REPO_ROOT) if run_dir.is_relative_to(REPO_ROOT) else run_dir),
            log_path.read_text(errors="ignore"), flatten(cfg),
            {"pipeline": str(cfg.get("pipeline", "full")), "overrides": overrides, "source": "hydra"},
            evals, args.dry_run, native_dir=str(run_dir),
        )

    for spec in args.main_run:
        name, log, *jsons = spec
        evals = dict(zip(("contrastive", "rl"), map(Path, jsons)))
        imported += import_run(
            client, "main_runs", name, log, Path(log).read_text(errors="ignore"), {},
            {"source": "train log"}, evals, args.dry_run,
        )
    print(f"imported {imported} run(s)")


if __name__ == "__main__":
    main()
