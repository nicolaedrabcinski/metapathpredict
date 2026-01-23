# Experiment Tracking

MetaPathPredict supports multiple experiment tracking backends for logging metrics, parameters, and artifacts.

## Supported Backends

| Backend | Best For | Features |
|---------|----------|----------|
| **MLflow** | Teams, MLOps | Model registry, serving |
| **W&B** | Visualization | Real-time plots, reports |
| **DuckDB** | Local, offline | SQL queries, lightweight |

## Configuration

### MLflow

```yaml
experiment:
  tracker: mlflow
  experiment_name: metapathpredict
  mlflow_tracking_uri: http://localhost:5000
```

### Weights & Biases

```yaml
experiment:
  tracker: wandb
  wandb_project: metapathpredict
  wandb_entity: my_team
  wandb_mode: online
```

### DuckDB (Local)

```yaml
experiment:
  tracker: duckdb
  duckdb_path: experiments.duckdb
```

## Usage

### Basic Tracking

```python
from metapathpredict.training.tracking import (
    ExperimentConfig,
    create_tracker,
)

# Create config
config = ExperimentConfig(
    tracker="mlflow",
    experiment_name="my_experiment",
)

# Create tracker
tracker = create_tracker(config)

# Start run
tracker.start_run(run_name="run_001")

# Log parameters
tracker.log_params({
    "learning_rate": 0.001,
    "batch_size": 32,
    "kernel_preset": "medium",
})

# Log metrics
for epoch in range(100):
    tracker.log_metrics({
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    }, step=epoch)

# Log model
tracker.log_model(model, "final_model")

# End run
tracker.end_run()
```

### With TrackedTrainer

```python
from metapathpredict.training.tracking import TrackedTrainer

# Wrap existing trainer
tracked = TrackedTrainer(trainer, tracker)

# Train with automatic tracking
tracked.train(
    train_loader,
    val_loader,
    epochs=100,
    run_name="experiment_001",
    config=config.model_dump(),
)
```

## MLflow

### Setup

```bash
# Install
pip install mlflow

# Start server
mlflow server --host 0.0.0.0 --port 5000
```

### Tracking

```python
from metapathpredict.training.tracking import MLflowTracker

tracker = MLflowTracker(ExperimentConfig(
    tracker="mlflow",
    mlflow_tracking_uri="http://localhost:5000",
))

tracker.start_run("experiment_1")
tracker.log_params({"lr": 0.001})
tracker.log_metrics({"loss": 0.5}, step=1)
tracker.log_artifact("results.csv")
tracker.log_model(model, "model")
tracker.end_run()
```

### Model Registry

```python
import mlflow

# Register model
mlflow.register_model(
    "runs:/run_id/model",
    "metapathpredict-classifier",
)

# Load registered model
model = mlflow.pytorch.load_model(
    "models:/metapathpredict-classifier/Production"
)
```

## Weights & Biases

### Setup

```bash
# Install
pip install wandb

# Login
wandb login
```

### Tracking

```python
from metapathpredict.training.tracking import WandbTracker

tracker = WandbTracker(ExperimentConfig(
    tracker="wandb",
    wandb_project="metapathpredict",
))

tracker.start_run("experiment_1")
tracker.log_params({"lr": 0.001})
tracker.log_metrics({"loss": 0.5}, step=1)

# W&B specific: watch model gradients
tracker.watch_model(model, log="gradients", log_freq=100)

tracker.end_run()
```

### Reports

Create shareable reports with W&B:

```python
import wandb

# Log rich media
wandb.log({
    "confusion_matrix": wandb.plot.confusion_matrix(
        y_true=y_true,
        preds=y_pred,
        class_names=["bacteria", "virus", "eucaryotic"],
    ),
    "roc": wandb.plot.roc_curve(y_true, y_probs),
})
```

## DuckDB (Local)

### Setup

```bash
pip install duckdb
```

### Tracking

```python
from metapathpredict.training.tracking import DuckDBTracker

tracker = DuckDBTracker(ExperimentConfig(
    tracker="duckdb",
    duckdb_path="experiments.duckdb",
))

tracker.start_run("experiment_1")
tracker.log_params({"lr": 0.001})
tracker.log_metrics({"loss": 0.5}, step=1)
tracker.end_run()
```

### Query Results

```python
import duckdb

conn = duckdb.connect("experiments.duckdb")

# Get all runs
runs = conn.execute("""
    SELECT run_id, run_name, start_time, status
    FROM runs
    ORDER BY start_time DESC
""").fetchall()

# Get metrics for a run
metrics = conn.execute("""
    SELECT key, value, step
    FROM metrics
    WHERE run_id = ?
    ORDER BY step
""", [run_id]).fetchall()

# Compare runs
comparison = conn.execute("""
    SELECT 
        r.run_name,
        MAX(CASE WHEN m.key = 'val_accuracy' THEN m.value END) as best_accuracy
    FROM runs r
    JOIN metrics m ON r.run_id = m.run_id
    GROUP BY r.run_name
    ORDER BY best_accuracy DESC
""").fetchall()
```

## Environment Variables

Set tracking via environment:

```bash
# MLflow
export MLFLOW_TRACKING_URI=http://localhost:5000

# W&B
export WANDB_API_KEY=your_key
export WANDB_PROJECT=metapathpredict

# MetaPathPredict
export METAPATH_TRACKER=mlflow
```

## Logging Best Practices

### Parameters to Log

```python
tracker.log_params({
    # Model
    "model/architecture": "cnn",
    "model/kernel_preset": "medium",
    "model/hidden_channels": "[32, 64, 128]",
    "model/dropout": 0.3,
    
    # Training
    "training/epochs": 100,
    "training/batch_size": 32,
    "training/learning_rate": 0.001,
    "training/scheduler": "cosine",
    
    # Data
    "data/max_length": 500,
    "data/augmentation": True,
    
    # System
    "system/gpu": torch.cuda.get_device_name(0),
    "system/pytorch_version": torch.__version__,
})
```

### Metrics to Log

```python
# Per epoch
tracker.log_metrics({
    "train/loss": train_loss,
    "train/accuracy": train_acc,
    "val/loss": val_loss,
    "val/accuracy": val_acc,
    "learning_rate": lr,
}, step=epoch)

# Final metrics
tracker.log_metrics({
    "test/accuracy": test_acc,
    "test/f1": test_f1,
    "test/auc": test_auc,
})
```

### Artifacts to Log

```python
# Model checkpoints
tracker.log_artifact("checkpoints/best_model.pt")

# Plots
tracker.log_artifact("plots/confusion_matrix.png")

# Results
tracker.log_artifact("results/predictions.csv")

# Config
tracker.log_artifact("config.yaml")
```

## Hyperparameter Tuning

### With Ray Tune + Tracking

```python
from ray import tune
from ray.air.integrations.mlflow import MLflowLoggerCallback

analysis = tune.run(
    train_fn,
    config=search_space,
    callbacks=[
        MLflowLoggerCallback(
            tracking_uri="http://localhost:5000",
            experiment_name="hyperparameter_tuning",
        )
    ],
)
```

## Dashboard

### MLflow UI

```bash
mlflow ui --port 5000
```

Access at http://localhost:5000

### W&B Dashboard

Access at https://wandb.ai/your_team/metapathpredict

### Custom DuckDB Dashboard

```python
import streamlit as st
import duckdb

conn = duckdb.connect("experiments.duckdb")

st.title("Experiment Dashboard")

# Show runs
runs = conn.execute("SELECT * FROM runs").df()
st.dataframe(runs)

# Plot metrics
selected_run = st.selectbox("Select run", runs["run_id"])
metrics = conn.execute(f"""
    SELECT step, value FROM metrics 
    WHERE run_id = '{selected_run}' AND key = 'val_accuracy'
""").df()
st.line_chart(metrics.set_index("step"))
```

## Next Steps

- [Training Guide](training.md) - Training with tracking
- [Deployment](../deployment/docker.md) - Deploy tracked models
- [API Reference](../api/training.md) - Tracking API docs
