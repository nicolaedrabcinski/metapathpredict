"""
Hydra-tracked launcher for the contrastive+RL pipeline.

Every invocation gets its own directory under experiments/<experiment_name>/<timestamp>/
containing the fully-resolved config (.hydra/config.yaml) and the run log
(run_experiment.log) — so two experiments can always be diffed to see exactly
what changed and what happened.

Examples:
    python scripts/run_experiment.py
    python scripts/run_experiment.py experiment_name=temp_0.5 contrastive.temperature=0.5
    python scripts/run_experiment.py -m experiment_name=temp_sweep contrastive.temperature=0.07,0.3,0.5
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from metapathpredict.cli import _train_full_pipeline, logger
from metapathpredict.config import Settings
from metapathpredict.data import SequenceDataModule


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    settings_dict = OmegaConf.to_container(cfg, resolve=True)
    settings = Settings(**settings_dict)

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

    rc = _train_full_pipeline(None, settings, device, data_module, output_dir)
    if rc != 0:
        raise SystemExit(rc)


if __name__ == "__main__":
    main()
