"""
Ensemble prediction combining multiple models.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import InferenceConfig
from .predictor import Predictor

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Ensemble predictor combining predictions from multiple models.
    
    Supports:
    - Averaging (soft voting)
    - Weighted averaging
    - Max voting (hard voting)
    - Stacking (meta-learner)
    """
    
    def __init__(
        self,
        predictors: list[Predictor],
        weights: list[float] | None = None,
        strategy: str = "average",
        class_names: list[str] | None = None,
    ):
        """
        Initialize ensemble.
        
        Args:
            predictors: List of Predictor instances.
            weights: Weights for each predictor (must sum to 1).
            strategy: Ensemble strategy - "average", "weighted", "vote", "max".
            class_names: Names for each class.
        """
        self.predictors = predictors
        self.strategy = strategy
        self.class_names = class_names or predictors[0].class_names
        
        # Normalize weights
        if weights is None:
            self.weights = [1.0 / len(predictors)] * len(predictors)
        else:
            weight_sum = sum(weights)
            self.weights = [w / weight_sum for w in weights]
        
        logger.info(
            f"Ensemble initialized with {len(predictors)} models, "
            f"strategy: {strategy}"
        )
    
    @classmethod
    def from_checkpoints(
        cls,
        checkpoint_paths: list[str | Path],
        model_class: type,
        model_kwargs: dict[str, Any] | None = None,
        weights: list[float] | None = None,
        strategy: str = "average",
        config: InferenceConfig | None = None,
        device: torch.device | str | None = None,
    ) -> "EnsemblePredictor":
        """
        Create ensemble from multiple checkpoints.
        
        Args:
            checkpoint_paths: Paths to model checkpoints.
            model_class: Class of models to load.
            model_kwargs: Arguments for model instantiation.
            weights: Weights for ensemble.
            strategy: Ensemble strategy.
            config: Inference configuration.
            device: Device to use.
        
        Returns:
            EnsemblePredictor instance.
        """
        predictors = []
        
        for path in checkpoint_paths:
            predictor = Predictor.from_checkpoint(
                checkpoint_path=path,
                model_class=model_class,
                model_kwargs=model_kwargs,
                config=config,
                device=device,
            )
            predictors.append(predictor)
        
        return cls(
            predictors=predictors,
            weights=weights,
            strategy=strategy,
        )
    
    @torch.no_grad()
    def predict(
        self,
        inputs: torch.Tensor | np.ndarray,
        return_probabilities: bool = True,
        use_tta: bool = False,
    ) -> dict[str, Any]:
        """
        Make ensemble predictions.
        
        Args:
            inputs: Input tensor.
            return_probabilities: Whether to return probabilities.
            use_tta: Whether to use TTA for individual models.
        
        Returns:
            Ensemble predictions.
        """
        # Get predictions from all models
        all_probs = []
        
        for predictor in self.predictors:
            result = predictor.predict(
                inputs,
                return_probabilities=True,
                use_tta=use_tta,
            )
            all_probs.append(result["probabilities"])
        
        # Stack probabilities [n_models, batch, n_classes]
        all_probs = np.stack(all_probs, axis=0)
        
        # Apply ensemble strategy
        if self.strategy == "average":
            ensemble_probs = all_probs.mean(axis=0)
        
        elif self.strategy == "weighted":
            weights = np.array(self.weights).reshape(-1, 1, 1)
            ensemble_probs = (all_probs * weights).sum(axis=0)
        
        elif self.strategy == "vote":
            # Hard voting
            predictions = all_probs.argmax(axis=2)  # [n_models, batch]
            ensemble_preds = []
            
            for i in range(predictions.shape[1]):
                votes = predictions[:, i]
                counts = np.bincount(votes, minlength=all_probs.shape[2])
                ensemble_preds.append(counts.argmax())
            
            ensemble_preds = np.array(ensemble_preds)
            
            # Create one-hot probabilities for votes
            ensemble_probs = np.zeros((len(ensemble_preds), all_probs.shape[2]))
            ensemble_probs[np.arange(len(ensemble_preds)), ensemble_preds] = 1.0
        
        elif self.strategy == "max":
            # Max confidence per class
            ensemble_probs = all_probs.max(axis=0)
            ensemble_probs = ensemble_probs / ensemble_probs.sum(axis=1, keepdims=True)
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Get predictions
        predicted_classes = ensemble_probs.argmax(axis=1)
        confidence = ensemble_probs.max(axis=1)
        
        result = {
            "predicted_class": predicted_classes,
            "predicted_label": [self.class_names[i] for i in predicted_classes],
            "confidence": confidence,
        }
        
        if return_probabilities:
            result["probabilities"] = ensemble_probs
            result["class_probabilities"] = {
                name: ensemble_probs[:, i]
                for i, name in enumerate(self.class_names)
            }
            result["individual_probabilities"] = all_probs
        
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
            use_tta: Whether to use TTA.
            show_progress: Whether to show progress bar.
        
        Returns:
            Ensemble predictions.
        """
        all_predictions = []
        all_probabilities = []
        all_confidences = []
        all_targets = []
        
        iterator = tqdm(dataloader, desc="Ensemble Predicting") if show_progress else dataloader
        
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
        
        # Concatenate
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
        
        if all_targets:
            result["targets"] = np.concatenate(all_targets, axis=0)
            result["accuracy"] = (predictions == result["targets"]).mean()
        
        return result


class KFoldEnsemble:
    """
    Ensemble from K-fold cross-validation models.
    
    Automatically loads models from fold directories.
    """
    
    def __init__(
        self,
        fold_dir: str | Path,
        model_class: type,
        model_kwargs: dict[str, Any] | None = None,
        checkpoint_name: str = "best_model.pt",
        config: InferenceConfig | None = None,
        device: torch.device | str | None = None,
    ):
        """
        Initialize K-fold ensemble.
        
        Args:
            fold_dir: Directory containing fold_0, fold_1, ... subdirectories.
            model_class: Class of model to load.
            model_kwargs: Arguments for model.
            checkpoint_name: Name of checkpoint file in each fold.
            config: Inference configuration.
            device: Device to use.
        """
        fold_dir = Path(fold_dir)
        
        # Find all fold directories
        fold_paths = sorted(fold_dir.glob("fold_*"))
        
        if not fold_paths:
            raise ValueError(f"No fold directories found in {fold_dir}")
        
        logger.info(f"Found {len(fold_paths)} folds in {fold_dir}")
        
        # Load checkpoints
        checkpoint_paths = [p / checkpoint_name for p in fold_paths]
        
        self.ensemble = EnsemblePredictor.from_checkpoints(
            checkpoint_paths=checkpoint_paths,
            model_class=model_class,
            model_kwargs=model_kwargs,
            strategy="average",
            config=config,
            device=device,
        )
    
    def predict(self, *args, **kwargs) -> dict[str, Any]:
        """Forward to ensemble."""
        return self.ensemble.predict(*args, **kwargs)
    
    def predict_dataloader(self, *args, **kwargs) -> dict[str, Any]:
        """Forward to ensemble."""
        return self.ensemble.predict_dataloader(*args, **kwargs)
