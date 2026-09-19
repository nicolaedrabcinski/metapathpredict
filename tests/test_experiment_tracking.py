"""Metrics sink, MLflow integration and the log importer's parsers."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from metapathpredict.cli import _evaluation_report, _report_to_metrics
from metapathpredict.experiment_tracking import MLflowSink, MetricsSink, NullSink, flatten_params

ROOT = Path(__file__).resolve().parents[1]


def _load_importer():
    spec = importlib.util.spec_from_file_location("import_runs_to_mlflow", ROOT / "scripts" / "import_runs_to_mlflow.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSinks:
    def test_null_sink_accepts_every_call(self):
        sink = NullSink()
        with sink:
            sink.log_params({"a": 1})
            sink.log_metrics({"m": 1.0}, step=1)
            sink.log_artifact("nowhere")
            sink.set_tags({"t": "v"})
        assert isinstance(sink, MetricsSink)

    def test_flatten_params_nests_with_dots_and_truncates(self):
        flat = flatten_params({"a": {"b": 1, "c": {"d": "x"}}, "e": "y" * 900})
        assert flat["a.b"] == "1" and flat["a.c.d"] == "x"
        assert len(flat["e"]) == 500

    def test_mlflow_sink_records_params_metrics_and_tags(self, tmp_path):
        pytest.importorskip("mlflow")
        from mlflow import MlflowClient

        uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
        with MLflowSink("exp", run_name="r1", tracking_uri=uri, tags={"k": "v"}, system_metrics=False) as sink:
            assert sink.active
            sink.log_params({"lr": "0.1"})
            sink.log_metrics({"loss": 2.0}, step=1)
            sink.log_metrics({"loss": 1.5}, step=2)
        client = MlflowClient(uri)
        run = client.search_runs([client.get_experiment_by_name("exp").experiment_id])[0]
        assert run.data.params["lr"] == "0.1"
        assert run.data.tags["k"] == "v" and run.data.tags["git_revision"]
        assert [m.value for m in client.get_metric_history(run.info.run_id, "loss")] == [2.0, 1.5]
        assert run.info.status == "FINISHED"

    def test_exception_marks_the_run_failed(self, tmp_path):
        pytest.importorskip("mlflow")
        from mlflow import MlflowClient

        uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
        with pytest.raises(RuntimeError):
            with MLflowSink("exp", tracking_uri=uri, system_metrics=False):
                raise RuntimeError("boom")
        client = MlflowClient(uri)
        assert client.search_runs([client.get_experiment_by_name("exp").experiment_id])[0].info.status == "FAILED"

    def test_broken_store_never_raises(self):
        pytest.importorskip("mlflow")
        sink = MLflowSink("exp", tracking_uri="sqlite:////nonexistent_dir/x/y.db", system_metrics=False)
        sink.log_params({"a": "b"})  # must not raise even though tracking is disabled
        sink.log_metrics({"m": 1.0}, step=1)
        sink.close()
        assert not sink.active


class TestEvaluationMetrics:
    def test_report_flattens_to_scalar_metrics_with_three_class_roll_up(self):
        names = ["bacteria", "archaea", "fungi", "protozoa", "plant", "invertebrate", "vertebrate", "virus"]
        targets = np.array(list(range(8)) * 3)
        preds = targets.copy()
        preds[0] = 1  # bacteria -> archaea: wrong at 8 classes, right after roll-up to prokaryote
        metrics = _report_to_metrics(_evaluation_report(names, targets, preds), "test/x")
        assert metrics["test/x/accuracy"] == pytest.approx(23 / 24)
        assert metrics["test/x/accuracy_3class"] == 1.0
        assert metrics["test/x/recall_bacteria"] == pytest.approx(2 / 3)
        assert metrics["test/x/recall_3class_virus"] == 1.0


class TestImporterParsers:
    def test_parses_all_line_formats_from_both_log_styles(self):
        text = "\r".join([
            "2026-09-19 05:16:42,199 - metapathpredict.models.contrastive - INFO -     Similarity: pos=0.7887, neg=0.3866, gap=0.4020",
            "2026-09-19 05:16:42,199 - x - INFO -     Gradients: avg_norm=4.2526, max_norm=6.2582",
            "2026-09-19 05:16:42,199 - x - INFO -     Embedding std=0.0335 (healthy)",
            "2026-09-19 05:16:42,199 - x - INFO -     Throughput: 272 samples/s, 265.0s total",
            "[2026-09-19 06:59:34,990][metapathpredict.cli][INFO] -   Val representation: alignment=0.2798, uniformity=-1.1180, "
            "erank(projection)=122.0/512, erank(backbone)=142.2/512",
            "Contrastive Epochs:   0%|  | 0/50 [00:00]2026-09-19 05:28:21,407 - metapathpredict.cli - INFO - "
            "[Contrastive] Epoch 3/15 | Loss: 3.251246 | Val loss: 5.522642 | Best val: 5.522642 | Time: 293.3s [BEST - saved]",
            "2026-09-19 06:00:00,000 - x - INFO -   Linear probe epoch 2/15: train_loss=1.1894, train_acc=0.5694, val_acc=0.5529",
            "2026-09-19 06:10:00,000 - x - INFO - [RL] Epoch 4/50 | Reward: 0.5074 | Train acc: 0.6716 | Val acc: 0.5884 | "
            "Loss: 0.1631 | Time: 14.5s [BEST - saved]",
        ])
        metrics, first, last = _load_importer().parse_log(text)
        by = {(k, step): v for k, v, step, _ in metrics}
        assert by[("contrastive/train_loss", 3)] == pytest.approx(3.251246)
        assert by[("contrastive/val_loss", 3)] == pytest.approx(5.522642)
        assert by[("contrastive/sim_gap", 3)] == pytest.approx(0.4020)
        assert by[("contrastive/val_erank_projection", 3)] == pytest.approx(122.0)
        assert by[("contrastive/grad_norm", 3)] == pytest.approx(4.2526)
        assert by[("probe/val_acc", 2)] == pytest.approx(0.5529)
        assert by[("rl/val_acc", 4)] == pytest.approx(0.5884)
        assert first is not None and last >= first

    def test_old_format_without_val_metrics(self):
        text = ("2026-09-18 17:00:00,000 - x - INFO - [Contrastive] Epoch 1/1 | Loss: 4.857435 | Best: 4.857435 | Time: 65.7s\n"
                "2026-09-18 17:01:00,000 - x - INFO - [RL] Epoch 1/1 | Reward: -0.0095 | Accuracy: 0.3270 | Loss: 0.2464 | Time: 5.5s\n")
        metrics, _, _ = _load_importer().parse_log(text)
        names = {k for k, *_ in metrics}
        assert "contrastive/train_loss" in names and "contrastive/val_loss" not in names
        assert "rl/train_acc" in names and "rl/val_acc" not in names
