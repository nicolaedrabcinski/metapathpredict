"""
Experiment tracking: the training loops report metrics to a `MetricsSink`.

`NullSink` (the default) does nothing, so `metapathpredict train` behaves exactly as before.
`MLflowSink` sends parameters, per-epoch metrics, system metrics (CPU, GPU, memory) and artifacts to
a local MLflow store; scripts/run_experiment.py enables it for Hydra runs. A tracking failure is
logged and never stops training.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


class MetricsSink:
    """Interface the training loops call. Does nothing by itself."""

    def log_params(self, params: Mapping[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        pass

    def log_artifact(self, path: str | Path) -> None:
        pass

    def set_tags(self, tags: Mapping[str, str]) -> None:
        pass

    def close(self, status: str = "FINISHED") -> None:
        pass

    def __enter__(self) -> "MetricsSink":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close("FAILED" if exc_type else "FINISHED")


class NullSink(MetricsSink):
    """Explicit no-op sink."""


def flatten_params(config: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    """Flatten a nested config to {"a.b": "value"} with values short enough for MLflow."""
    flat: dict[str, str] = {}
    for key, value in config.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten_params(value, prefix=f"{name}."))
        else:
            flat[name] = str(value)[:500]
    return flat


def _git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


class MLflowSink(MetricsSink):
    """Logs to an MLflow run. Every call is guarded: a failure only produces a warning."""

    def __init__(
        self,
        experiment: str,
        run_name: str | None = None,
        tracking_uri: str = DEFAULT_TRACKING_URI,
        tags: Mapping[str, str] | None = None,
        system_metrics: bool = True,
    ):
        self._mlflow = None
        try:
            os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
            import mlflow

            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment)
            if system_metrics:
                mlflow.enable_system_metrics_logging()
            self._run = mlflow.start_run(run_name=run_name)
            self._mlflow = mlflow
            self.set_tags({"git_revision": _git_revision(), **(tags or {})})
            logger.info(f"MLflow run {self._run.info.run_id} in experiment '{experiment}' ({tracking_uri})")
        except Exception as e:  # tracking must never break training
            logger.warning(f"MLflow disabled: {e}")

    @property
    def active(self) -> bool:
        return self._mlflow is not None

    def _guard(self, what: str, fn) -> None:
        if not self.active:
            return
        try:
            fn()
        except Exception as e:
            logger.warning(f"MLflow {what} failed: {e}")

    def log_params(self, params: Mapping[str, Any]) -> None:
        self._guard("log_params", lambda: self._mlflow.log_params(dict(params)))

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        clean = {k: float(v) for k, v in metrics.items() if v is not None}
        self._guard("log_metrics", lambda: self._mlflow.log_metrics(clean, step=step))

    def log_artifact(self, path: str | Path) -> None:
        path = Path(path)
        if path.is_dir():
            self._guard("log_artifacts", lambda: self._mlflow.log_artifacts(str(path), artifact_path=path.name))
        elif path.exists():
            self._guard("log_artifact", lambda: self._mlflow.log_artifact(str(path)))

    def set_tags(self, tags: Mapping[str, str]) -> None:
        self._guard("set_tags", lambda: self._mlflow.set_tags(dict(tags)))

    def close(self, status: str = "FINISHED") -> None:
        if self.active:
            self._guard("end_run", lambda: self._mlflow.end_run(status=status))
            self._mlflow = None
