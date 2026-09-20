"""
Hydra-tracked launcher for the contrastive (+ optional RL) pipeline.

Every invocation gets its own directory under experiments/<experiment_name>/<timestamp>/ with the
fully-resolved config (.hydra/config.yaml) and the run log, and one MLflow run with the same
parameters, per-epoch metrics, system metrics (CPU/GPU/memory) and the final test-split evaluation.

View results:
    mlflow ui --backend-store-uri sqlite:///mlflow.db        # then open http://localhost:5000

Examples:
    python scripts/run_experiment.py
    python scripts/run_experiment.py experiment_name=bs512 contrastive.batch_size=512
    python scripts/run_experiment.py -m experiment_name=lr contrastive.learning_rate=1e-4,3e-4
    python scripts/run_experiment.py tracking.enabled=false      # Hydra only, no MLflow
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from metapathpredict.cli import (
    _apply_dataset_labels,
    _cap_threads,
    _evaluation_report,
    _predict_split,
    _report_to_metrics,
    _train_contrastive,
    _train_full_pipeline,
    logger,
)
from metapathpredict.config import Settings
from metapathpredict.data import SequenceDataModule
from metapathpredict.experiment_tracking import MLflowSink, NullSink, flatten_params
from metapathpredict.genome_eval import (
    evaluate_by_genome,
    format_genome_report,
    genome_report_to_metrics,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_sink(cfg_dict: dict, tracking: dict, run_dir: Path, pipeline: str):
    if not tracking.get("enabled", True):
        return NullSink()
    overrides = HydraConfig.get().job.get("override_dirname", "") or ""
    uri = tracking.get("uri") or f"sqlite:///{REPO_ROOT / 'mlflow.db'}"
    sink = MLflowSink(
        experiment=cfg_dict["experiment_name"],
        run_name=f"{run_dir.name} {overrides}".strip(),
        tracking_uri=uri,
        tags={"pipeline": pipeline, "hydra_run_dir": str(run_dir), "overrides": overrides},
        system_metrics=tracking.get("system_metrics", True),
    )
    sink.log_params(flatten_params(cfg_dict))
    return sink


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run_dir = Path(HydraConfig.get().runtime.output_dir)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    settings_dict = dict(cfg_dict)
    pipeline = settings_dict.pop("pipeline", "full")  # "full" or "contrastive" (encoder + probe, no RL)
    max_threads = settings_dict.pop("max_threads", 16)
    tracking = settings_dict.pop("tracking", {}) or {}
    settings = Settings(**settings_dict)
    _cap_threads(settings, max_threads)

    # Isolate this run's checkpoints/logs instead of overwriting a shared dir.
    settings.paths.weights_dir = run_dir / "weights"
    settings.paths.logs_dir = run_dir / "logs"

    device = settings.get_device()
    logger.info(f"Experiment: {cfg.experiment_name}")
    logger.info(f"Run dir:    {run_dir}")
    logger.info(f"Device:     {device}")

    data_module = SequenceDataModule.from_config(settings)
    output_dir = settings.paths.weights_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _apply_dataset_labels(settings, data_module)

    with _make_sink(cfg_dict, tracking, run_dir, pipeline) as sink:
        sink.set_tags({"classes": ",".join(settings.data.class_names)})
        sink.log_metrics({
            "data/train_size": len(data_module.train_dataset),
            "data/val_size": len(data_module.val_dataset),
            "data/test_size": len(data_module.test_dataset),
        })

        if pipeline == "contrastive":
            rc = _train_contrastive(settings, device, data_module, output_dir, sink=sink)
        else:
            rc = _train_full_pipeline(None, settings, device, data_module, output_dir, sink=sink)
        if rc != 0:
            raise SystemExit(rc)

        # Score the checkpoints on the species-disjoint test split.
        frag = settings.data.default_fragment_size
        test_path = settings.paths.datasets_dir / f"encoded_test_{frag}.hdf5"
        artifacts = [run_dir / ".hydra", run_dir / "run_experiment.log"]
        for tag, checkpoint in (("contrastive", "contrastive_best.pt"), ("rl", "rl_best.pt")):
            checkpoint_path = output_dir / checkpoint
            if not checkpoint_path.exists():
                continue
            _, class_names, targets, preds = _predict_split(checkpoint_path, str(test_path), device)
            report = _evaluation_report(class_names, targets, preds)
            report_path = run_dir / f"test_{tag}.json"
            report_path.write_text(json.dumps(report, indent=2))
            artifacts.append(report_path)
            sink.log_metrics(_report_to_metrics(report, f"test/{tag}"))
            agg = report["aggregated_3class"]
            logger.info(
                f"Test {tag}: accuracy {report['accuracy']:.4f}"
                + (f", 3-class {agg['accuracy']:.4f}" if agg else "")
            )
            try:  # extra reporting must never cost the run its main test results
                genome = evaluate_by_genome(settings.paths.datasets_dir, targets, preds, class_names)
                if genome:
                    genome_path = run_dir / f"test_{tag}_genomes.json"
                    genome_path.write_text(json.dumps(genome, indent=2))
                    artifacts.append(genome_path)
                    sink.log_metrics(genome_report_to_metrics(genome, f"test/{tag}"))
                    logger.info(format_genome_report(genome))
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Genome-level evaluation failed: {exc}")
        for path in artifacts:
            sink.log_artifact(path)


if __name__ == "__main__":
    main()
