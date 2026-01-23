# Inference Guide

This guide covers making predictions with trained models.

## Quick Start

```bash
metapathpredict predict \
    --model checkpoints/best_model.pt \
    --input sequences.fasta \
    --output predictions.csv
```

## Using the Predictor

### Basic Usage

```python
from metapathpredict.inference import Predictor

# Load predictor
predictor = Predictor.from_checkpoint("checkpoints/best_model.pt")

# Predict from FASTA
results = predictor.predict_fasta("sequences.fasta")

for seq_id, pred in results.items():
    print(f"{seq_id}: {pred['class']} ({pred['confidence']:.2%})")
```

### Predict Single Sequence

```python
# Predict single sequence
result = predictor.predict_sequence(
    "ATGCGATCGATCGATCGATCGATCGATCG..."
)

print(f"Class: {result['class']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Probabilities: {result['probabilities']}")
```

### Batch Prediction

```python
sequences = [
    "ATGCGATCGATCGATCG...",
    "GCTAGCTAGCTAGCTAG...",
    "TATATATATATATATA...",
]

results = predictor.predict_batch(sequences)
```

## Output Formats

### CSV

```bash
metapathpredict predict --output predictions.csv
```

```csv
sequence_id,predicted_class,confidence,prob_bacteria,prob_virus,prob_eucaryotic
seq_001,bacteria,0.95,0.95,0.03,0.02
seq_002,virus,0.88,0.08,0.88,0.04
```

### JSON

```bash
metapathpredict predict --output predictions.json
```

```json
{
  "seq_001": {
    "class": "bacteria",
    "confidence": 0.95,
    "probabilities": {
      "bacteria": 0.95,
      "virus": 0.03,
      "eucaryotic": 0.02
    }
  }
}
```

### FASTA with Headers

```bash
metapathpredict predict --output predictions.fasta
```

```
>seq_001 predicted=bacteria confidence=0.95
ATGCGATCGATCGATCG...
```

## Ensemble Predictions

### Multiple Models

```python
from metapathpredict.inference import EnsemblePredictor

ensemble = EnsemblePredictor([
    "checkpoints/model_k5.pt",
    "checkpoints/model_k7.pt",
    "checkpoints/model_k10.pt",
])

results = ensemble.predict_fasta("sequences.fasta")
```

### Ensemble Strategies

```python
# Average probabilities (default)
ensemble = EnsemblePredictor(models, strategy="average")

# Majority voting
ensemble = EnsemblePredictor(models, strategy="vote")

# Weighted average
ensemble = EnsemblePredictor(
    models,
    strategy="weighted",
    weights=[0.3, 0.4, 0.3],
)
```

## Test-Time Augmentation

Improve predictions with augmentation:

```python
from metapathpredict.inference import TTAPredictor

tta_predictor = TTAPredictor(
    model=model,
    augmentations=["crop", "reverse_complement"],
    num_augmentations=5,
)

results = tta_predictor.predict_fasta("sequences.fasta")
```

## Streaming Predictions

For large files:

```python
from metapathpredict.inference import StreamingPredictor

predictor = StreamingPredictor(
    model=model,
    batch_size=100,
    max_memory_gb=4,
)

for batch_results in predictor.stream_predict("large_file.fasta"):
    save_results(batch_results)
```

## GPU Inference

### Single GPU

```python
predictor = Predictor.from_checkpoint(
    "model.pt",
    device="cuda:0",
)
```

### Multi-GPU

```python
predictor = Predictor.from_checkpoint(
    "model.pt",
    device="cuda",
    data_parallel=True,
)
```

### Batch Size Optimization

```python
# Auto-tune batch size for GPU memory
predictor = Predictor.from_checkpoint(
    "model.pt",
    device="cuda",
    auto_batch_size=True,
)
```

## Confidence Filtering

### Filter Low Confidence

```python
results = predictor.predict_fasta(
    "sequences.fasta",
    min_confidence=0.8,
)

# Low confidence sequences returned separately
confident = results["confident"]
uncertain = results["uncertain"]
```

### Custom Thresholds

```python
thresholds = {
    "bacteria": 0.9,
    "virus": 0.85,
    "eucaryotic": 0.8,
}

results = predictor.predict_with_thresholds(
    sequences,
    thresholds=thresholds,
)
```

## Model Optimization

### TorchScript Export

```python
# Export for production
scripted = torch.jit.script(model)
scripted.save("model_scripted.pt")

# Load and predict
model = torch.jit.load("model_scripted.pt")
```

### ONNX Export

```python
import torch.onnx

dummy_input = torch.randn(1, 4, 500)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["sequence"],
    output_names=["probabilities"],
)
```

### torch.compile (PyTorch 2.0+)

```python
model = torch.compile(model, mode="reduce-overhead")
```

## Evaluation

### Metrics

```python
from metapathpredict.inference import evaluate_model

metrics = evaluate_model(
    model,
    test_loader,
    metrics=["accuracy", "f1", "precision", "recall", "auc"],
)

print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"F1 Score: {metrics['f1']:.3f}")
```

### Confusion Matrix

```python
from metapathpredict.inference import plot_confusion_matrix

plot_confusion_matrix(
    y_true,
    y_pred,
    class_names=["bacteria", "virus", "eucaryotic"],
    save_path="confusion_matrix.png",
)
```

### Per-Class Metrics

```python
from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred, target_names=class_names))
```

## CLI Options

```bash
metapathpredict predict \
    --model checkpoints/best_model.pt \
    --input sequences.fasta \
    --output predictions.csv \
    --batch-size 64 \
    --device cuda \
    --min-confidence 0.8 \
    --format csv \
    --tta \
    --verbose
```

## Performance Tips

### Speed Optimization

1. Use GPU with appropriate batch size
2. Use `torch.compile()` for PyTorch 2.0+
3. Disable gradient computation
4. Use TorchScript for production

### Memory Optimization

1. Use streaming for large files
2. Reduce batch size
3. Use `torch.inference_mode()`
4. Clear CUDA cache periodically

```python
@torch.inference_mode()
def predict(self, sequences):
    # Faster inference without gradient tracking
    return self.model(sequences)
```

## Next Steps

- [Experiment Tracking](tracking.md) - Track predictions
- [Deployment](../deployment/docker.md) - Deploy models
- [API Reference](../api/inference.md) - Inference API docs
