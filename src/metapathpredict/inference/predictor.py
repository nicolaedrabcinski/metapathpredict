"""
Predictor class for model inference.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..config import InferenceConfig, CLASS_NAMES as CONFIG_CLASS_NAMES

logger = logging.getLogger(__name__)


class Predictor:
    """
    Model predictor for inference with support for:
    - Test-time augmentation (TTA)
    - Batch prediction
    - Multiple output formats
    - Confidence thresholding
    """
    
    # Use centralized class names from config (alphabetical: bacteria, eukaryotic, virus)
    CLASS_NAMES = CONFIG_CLASS_NAMES
    
    def __init__(
        self,
        model: nn.Module,
        config: InferenceConfig | None = None,
        device: torch.device | str | None = None,
        class_names: list[str] | None = None,
    ):
        """
        Initialize predictor.
        
        Args:
            model: Trained PyTorch model.
            config: Inference configuration.
            device: Device to run inference on.
            class_names: Names for each class.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        
        self.model = model.to(self.device)
        self.model.eval()
        
        self.config = config or InferenceConfig()
        self.class_names = class_names or self.CLASS_NAMES
        
        self.use_amp = self.config.use_mixed_precision and self.device.type == "cuda"
        
        logger.info(f"Predictor initialized on {self.device}")
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        model_class: type,
        model_kwargs: dict[str, Any] | None = None,
        config: InferenceConfig | None = None,
        device: torch.device | str | None = None,
    ) -> "Predictor":
        """
        Create predictor from a checkpoint file.
        
        Args:
            checkpoint_path: Path to model checkpoint.
            model_class: Class of the model to instantiate.
            model_kwargs: Arguments for model instantiation.
            config: Inference configuration.
            device: Device to use.
        
        Returns:
            Predictor instance with loaded model.
        """
        checkpoint_path = Path(checkpoint_path)
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Create model
        model_kwargs = model_kwargs or {}
        model = model_class(**model_kwargs)
        
        # Load state dict
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        
        return cls(model=model, config=config, device=device)
    
    def _reverse_complement(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply reverse complement to one-hot encoded sequence.
        
        For DNA: A<->T, G<->C (indices 0<->3, 1<->2 in ACGT encoding)
        
        Args:
            x: One-hot encoded tensor [batch, 4, length]
        
        Returns:
            Reverse complement tensor.
        """
        # Reverse the sequence
        x_rev = torch.flip(x, dims=[-1])
        
        # Complement: swap A<->T (0<->3) and G<->C (1<->2)
        # ACGT order -> TGCA order
        idx = torch.tensor([3, 2, 1, 0], device=x.device)
        x_comp = x_rev.index_select(dim=1, index=idx)
        
        return x_comp
    
    @torch.no_grad()
    def predict(
        self,
        inputs: torch.Tensor | np.ndarray,
        return_probabilities: bool = True,
        use_tta: bool = False,
    ) -> dict[str, Any]:
        """
        Make predictions on input data.
        
        Args:
            inputs: Input tensor [batch, channels, length] or numpy array.
            return_probabilities: Whether to return class probabilities.
            use_tta: Whether to use test-time augmentation.
        
        Returns:
            Dictionary with predictions and optional probabilities.
        """
        # Convert to tensor if needed
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).float()
        
        inputs = inputs.to(self.device)
        
        # Ensure batch dimension
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(0)
        
        with autocast(enabled=self.use_amp):
            if use_tta:
                # Test-time augmentation
                outputs_list = []
                
                # Original
                outputs_list.append(self.model(inputs))
                
                # Reverse complement
                inputs_rc = self._reverse_complement(inputs)
                outputs_list.append(self.model(inputs_rc))
                
                # Average predictions
                outputs = torch.stack(outputs_list, dim=0).mean(dim=0)
            else:
                outputs = self.model(inputs)
        
        # Get probabilities
        probabilities = F.softmax(outputs, dim=1)
        
        # Get predictions
        predicted_classes = probabilities.argmax(dim=1)
        confidence = probabilities.max(dim=1).values
        
        result = {
            "predicted_class": predicted_classes.cpu().numpy(),
            "predicted_label": [self.class_names[i] for i in predicted_classes.cpu().tolist()],
            "confidence": confidence.cpu().numpy(),
        }
        
        if return_probabilities:
            result["probabilities"] = probabilities.cpu().numpy()
            result["class_probabilities"] = {
                name: probabilities[:, i].cpu().numpy()
                for i, name in enumerate(self.class_names)
            }
        
        return result
    
    @torch.no_grad()
    def predict_dataloader(
        self,
        dataloader: DataLoader,
        return_probabilities: bool = True,
        use_tta: bool = False,
        show_progress: bool = True,
    ) -> dict[str, Any]:
        """
        Make predictions on a dataloader.
        
        Args:
            dataloader: DataLoader with input data.
            return_probabilities: Whether to return probabilities.
            use_tta: Whether to use test-time augmentation.
            show_progress: Whether to show progress bar.
        
        Returns:
            Dictionary with predictions and optional probabilities.
        """
        all_predictions = []
        all_probabilities = []
        all_confidences = []
        all_targets = []
        
        iterator = tqdm(dataloader, desc="Predicting") if show_progress else dataloader
        
        for batch in iterator:
            # Get inputs
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
                targets = batch[1] if len(batch) > 1 else None
            elif isinstance(batch, dict):
                inputs = batch["input"]
                targets = batch.get("target")
            else:
                inputs = batch
                targets = None
            
            # Predict
            result = self.predict(
                inputs,
                return_probabilities=return_probabilities,
                use_tta=use_tta,
            )
            
            all_predictions.append(result["predicted_class"])
            all_confidences.append(result["confidence"])
            
            if return_probabilities:
                all_probabilities.append(result["probabilities"])
            
            if targets is not None:
                if isinstance(targets, torch.Tensor):
                    targets = targets.cpu().numpy()
                all_targets.append(targets)
        
        # Concatenate results
        predictions = np.concatenate(all_predictions, axis=0)
        confidences = np.concatenate(all_confidences, axis=0)
        
        result = {
            "predicted_class": predictions,
            "predicted_label": [self.class_names[i] for i in predictions],
            "confidence": confidences,
        }
        
        if return_probabilities:
            probabilities = np.concatenate(all_probabilities, axis=0)
            result["probabilities"] = probabilities
            result["class_probabilities"] = {
                name: probabilities[:, i]
                for i, name in enumerate(self.class_names)
            }
        
        if all_targets:
            result["targets"] = np.concatenate(all_targets, axis=0)
            result["accuracy"] = (predictions == result["targets"]).mean()
        
        return result
    
    def predict_sequences(
        self,
        sequences: Sequence[str],
        preprocessor: Any,
        batch_size: int = 32,
        use_tta: bool = False,
    ) -> dict[str, Any]:
        """
        Make predictions on raw DNA sequences.
        
        Args:
            sequences: List of DNA sequences.
            preprocessor: Preprocessor to encode sequences.
            batch_size: Batch size for prediction.
            use_tta: Whether to use TTA.
        
        Returns:
            Prediction results.
        """
        all_predictions = []
        all_probabilities = []
        all_confidences = []
        
        # Process in batches
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i + batch_size]
            
            # Encode sequences
            encoded = []
            for seq in batch_sequences:
                enc = preprocessor.encode_sequence(seq)
                encoded.append(enc)
            
            # Stack into tensor
            inputs = torch.stack([torch.from_numpy(e).float() for e in encoded])
            
            # Predict
            result = self.predict(
                inputs,
                return_probabilities=True,
                use_tta=use_tta,
            )
            
            all_predictions.append(result["predicted_class"])
            all_confidences.append(result["confidence"])
            all_probabilities.append(result["probabilities"])
        
        # Concatenate
        predictions = np.concatenate(all_predictions, axis=0)
        confidences = np.concatenate(all_confidences, axis=0)
        probabilities = np.concatenate(all_probabilities, axis=0)
        
        return {
            "predicted_class": predictions,
            "predicted_label": [self.class_names[i] for i in predictions],
            "confidence": confidences,
            "probabilities": probabilities,
            "class_probabilities": {
                name: probabilities[:, i]
                for i, name in enumerate(self.class_names)
            },
        }


class BatchPredictor:
    """
    Efficient batch predictor for large-scale inference.
    
    Uses memory-efficient processing with generators.
    """
    
    def __init__(
        self,
        predictor: Predictor,
        batch_size: int = 256,
        num_workers: int = 4,
    ):
        """
        Initialize batch predictor.
        
        Args:
            predictor: Base predictor.
            batch_size: Batch size for processing.
            num_workers: Number of data loading workers.
        """
        self.predictor = predictor
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def predict_dataset(
        self,
        dataset: Dataset,
        use_tta: bool = False,
        show_progress: bool = True,
    ) -> dict[str, np.ndarray]:
        """
        Predict on entire dataset.
        
        Args:
            dataset: PyTorch dataset.
            use_tta: Whether to use TTA.
            show_progress: Whether to show progress.
        
        Returns:
            Prediction results.
        """
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        
        return self.predictor.predict_dataloader(
            dataloader,
            use_tta=use_tta,
            show_progress=show_progress,
        )
