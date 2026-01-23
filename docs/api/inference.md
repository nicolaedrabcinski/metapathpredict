# API Reference: Inference

Inference and prediction utilities.

## Predictor

```python
class Predictor:
    """High-level inference interface.
    
    Args:
        model: Trained model or path to checkpoint.
        config: Model configuration.
        device: Inference device.
        compile: Whether to use torch.compile().
    
    Example:
        >>> predictor = Predictor("checkpoints/best.pt")
        >>> predictions = predictor.predict(sequences)
    """
    
    def __init__(
        self,
        model: Union[nn.Module, str, Path],
        config: Optional[Config] = None,
        device: str = "cuda",
        compile: bool = True,
    ):
        pass
    
    def predict(
        self,
        sequences: Union[str, List[str], np.ndarray, torch.Tensor],
        batch_size: int = 32,
        return_probs: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict organism type for sequences.
        
        Args:
            sequences: Input sequences or encoded array.
            batch_size: Inference batch size.
            return_probs: Whether to return probabilities.
        
        Returns:
            Predicted labels (and probabilities if requested).
        """
        pass
    
    def predict_proba(
        self,
        sequences: Union[str, List[str], np.ndarray],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Get prediction probabilities.
        
        Args:
            sequences: Input sequences.
            batch_size: Inference batch size.
        
        Returns:
            Array of shape (n_samples, n_classes).
        """
        pass
    
    def predict_fasta(
        self,
        fasta_path: str,
        output_path: Optional[str] = None,
        batch_size: int = 32,
    ) -> pd.DataFrame:
        """Predict from FASTA file.
        
        Args:
            fasta_path: Path to input FASTA.
            output_path: Path to save results CSV.
            batch_size: Inference batch size.
        
        Returns:
            DataFrame with predictions.
        """
        pass
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cuda",
    ) -> "Predictor":
        """Load predictor from checkpoint.
        
        Args:
            checkpoint_path: Path to .pt file.
            device: Inference device.
        
        Returns:
            Configured Predictor instance.
        """
        pass
```

## EnsemblePredictor

```python
class EnsemblePredictor:
    """Ensemble multiple models for improved predictions.
    
    Args:
        models: List of models or paths.
        weights: Optional model weights (default: equal).
        method: Ensemble method ("mean", "vote", "stack").
        device: Inference device.
    
    Example:
        >>> ensemble = EnsemblePredictor([
        ...     "checkpoints/model_k5.pt",
        ...     "checkpoints/model_k7.pt",
        ...     "checkpoints/model_k10.pt",
        ... ])
        >>> predictions = ensemble.predict(sequences)
    """
    
    def __init__(
        self,
        models: List[Union[nn.Module, str]],
        weights: Optional[List[float]] = None,
        method: str = "mean",
        device: str = "cuda",
    ):
        pass
    
    def predict(
        self,
        sequences: Union[str, List[str], np.ndarray],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Get ensemble predictions.
        
        Returns:
            Predicted labels.
        """
        pass
    
    def predict_proba(
        self,
        sequences: Union[str, List[str], np.ndarray],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Get ensemble probabilities.
        
        Returns:
            Averaged or stacked probabilities.
        """
        pass
```

## TTAPredictor

```python
class TTAPredictor:
    """Test-time augmentation predictor.
    
    Args:
        predictor: Base predictor.
        augmentations: List of augmentation functions.
        n_augment: Number of augmented versions.
    
    Example:
        >>> tta = TTAPredictor(predictor, n_augment=5)
        >>> predictions = tta.predict(sequences)
    """
    
    def __init__(
        self,
        predictor: Predictor,
        augmentations: Optional[List[Callable]] = None,
        n_augment: int = 5,
    ):
        pass
    
    def predict(
        self,
        sequences: Union[str, List[str]],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Predict with TTA.
        
        Returns:
            Labels averaged over augmentations.
        """
        pass
```

## Batch Processing

### InferencePipeline

```python
class InferencePipeline:
    """Pipeline for large-scale batch inference.
    
    Args:
        predictor: Predictor instance.
        output_dir: Directory for output files.
        chunk_size: Sequences per output file.
    
    Example:
        >>> pipeline = InferencePipeline(predictor)
        >>> pipeline.process_directory("input/", "output/")
    """
    
    def __init__(
        self,
        predictor: Predictor,
        output_dir: str = "output",
        chunk_size: int = 10000,
    ):
        pass
    
    def process_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
    ) -> str:
        """Process single file.
        
        Returns:
            Path to output file.
        """
        pass
    
    def process_directory(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        pattern: str = "*.fasta",
        parallel: bool = True,
    ) -> List[str]:
        """Process all files in directory.
        
        Args:
            input_dir: Input directory.
            output_dir: Output directory.
            pattern: File pattern to match.
            parallel: Use parallel processing.
        
        Returns:
            List of output file paths.
        """
        pass
```

### StreamingPredictor

```python
class StreamingPredictor:
    """Memory-efficient streaming inference.
    
    Args:
        predictor: Base predictor.
        buffer_size: Number of sequences to buffer.
    
    Example:
        >>> streaming = StreamingPredictor(predictor)
        >>> for batch_result in streaming.predict_stream(fasta_path):
        ...     process(batch_result)
    """
    
    def __init__(
        self,
        predictor: Predictor,
        buffer_size: int = 1000,
    ):
        pass
    
    def predict_stream(
        self,
        fasta_path: str,
    ) -> Iterator[pd.DataFrame]:
        """Stream predictions from FASTA.
        
        Yields:
            DataFrames with batch predictions.
        """
        pass
```

## Model Serialization

### save_model

```python
def save_model(
    model: nn.Module,
    path: str,
    config: Optional[Config] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    metadata: Optional[Dict] = None,
):
    """Save model checkpoint.
    
    Args:
        model: Model to save.
        path: Output path (.pt).
        config: Model configuration.
        optimizer: Optimizer state (for resume).
        metadata: Additional metadata.
    
    Example:
        >>> save_model(model, "checkpoints/model.pt", config)
    """
    pass
```

### load_model

```python
def load_model(
    path: str,
    device: str = "cuda",
    compile: bool = False,
) -> Tuple[nn.Module, Config]:
    """Load model from checkpoint.
    
    Args:
        path: Checkpoint path.
        device: Target device.
        compile: Apply torch.compile().
    
    Returns:
        Tuple of (model, config).
    """
    pass
```

### export_onnx

```python
def export_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, ...],
    opset_version: int = 14,
):
    """Export model to ONNX format.
    
    Args:
        model: PyTorch model.
        output_path: ONNX file path.
        input_shape: Example input shape.
        opset_version: ONNX opset version.
    
    Example:
        >>> export_onnx(model, "model.onnx", (1, 4, 1000))
    """
    pass
```

### export_torchscript

```python
def export_torchscript(
    model: nn.Module,
    output_path: str,
    example_input: torch.Tensor,
    method: str = "trace",
):
    """Export model to TorchScript.
    
    Args:
        model: PyTorch model.
        output_path: Output path.
        example_input: Example input tensor.
        method: "trace" or "script".
    
    Example:
        >>> export_torchscript(model, "model.pt", example_input)
    """
    pass
```

## Uncertainty Estimation

### MCDropoutPredictor

```python
class MCDropoutPredictor:
    """Monte Carlo Dropout for uncertainty estimation.
    
    Args:
        predictor: Base predictor with dropout.
        n_samples: Number of MC samples.
    
    Example:
        >>> mc = MCDropoutPredictor(predictor, n_samples=30)
        >>> preds, uncertainty = mc.predict_with_uncertainty(sequences)
    """
    
    def __init__(
        self,
        predictor: Predictor,
        n_samples: int = 30,
    ):
        pass
    
    def predict_with_uncertainty(
        self,
        sequences: Union[str, List[str]],
        batch_size: int = 32,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty estimates.
        
        Returns:
            Tuple of (predictions, uncertainties).
        """
        pass
```

## Result Formatting

### PredictionResult

```python
@dataclass
class PredictionResult:
    """Container for prediction results.
    
    Attributes:
        sequence_ids: Sequence identifiers.
        predictions: Predicted labels.
        probabilities: Class probabilities.
        confidence: Prediction confidence scores.
        metadata: Additional metadata.
    """
    sequence_ids: List[str]
    predictions: np.ndarray
    probabilities: np.ndarray
    confidence: np.ndarray
    metadata: Dict[str, Any]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        pass
    
    def to_csv(self, path: str):
        """Save to CSV file."""
        pass
    
    def to_json(self, path: str):
        """Save to JSON file."""
        pass
    
    def summary(self) -> Dict[str, Any]:
        """Get prediction summary statistics."""
        pass
```

### format_predictions

```python
def format_predictions(
    sequence_ids: List[str],
    predictions: np.ndarray,
    probabilities: np.ndarray,
    class_names: List[str] = ["bacteria", "eukaryote", "virus"],
) -> pd.DataFrame:
    """Format predictions as DataFrame.
    
    Args:
        sequence_ids: Sequence identifiers.
        predictions: Predicted class indices.
        probabilities: Class probabilities.
        class_names: Human-readable class names.
    
    Returns:
        Formatted DataFrame with columns:
        - sequence_id
        - prediction
        - confidence
        - prob_bacteria, prob_eukaryote, prob_virus
    """
    pass
```
