"""Prediction service using Contrastive Learning and RL models."""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

from metapathpredict.api.schemas import (
    PredictionComparison,
    MethodPrediction,
    ClassProbabilities,
    MethodType,
    ClassLabel,
)
from metapathpredict.api.services.sample_service import SampleService
from metapathpredict.models.contrastive import ContrastiveEncoder
from metapathpredict.models.reinforcement import (
    DQNAgent,
    PolicyGradientAgent,
    ActorCriticAgent,
)

# Label index → ClassLabel mapping
INDEX_TO_CLASS = {0: ClassLabel.BACTERIA, 1: ClassLabel.EUKARYOTIC, 2: ClassLabel.VIRUS}

# Nucleotide → one-hot channel index
NUC_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


def encode_sequence(sequence: str, fragment_size: int = 500) -> torch.Tensor:
    """One-hot encode a DNA sequence to tensor [1, 4, fragment_size]."""
    seq = sequence.upper()[:fragment_size]
    tensor = torch.zeros(1, 4, fragment_size)
    for i, nuc in enumerate(seq):
        idx = NUC_TO_IDX.get(nuc)
        if idx is not None:
            tensor[0, idx, i] = 1.0
    return tensor


class PredictionService:
    """Service for predictions using contrastive and RL models."""

    def __init__(self, weights_dir: str = "data/weights/unified"):
        self.sample_service = SampleService()
        self.weights_dir = Path(weights_dir)
        self.device = torch.device("cpu")
        self._contrastive_encoder: Optional[ContrastiveEncoder] = None
        self._rl_agent = None
        self._rl_algorithm: Optional[str] = None
        self._models_loaded = False
        self._load_models()

    def _load_models(self) -> None:
        """Load trained contrastive and RL models from checkpoints."""
        contrastive_path = self.weights_dir / "contrastive_best.pt"
        rl_path = self.weights_dir / "rl_best.pt"

        # Load contrastive encoder
        if contrastive_path.exists():
            try:
                checkpoint = torch.load(contrastive_path, map_location=self.device, weights_only=False)
                config = checkpoint.get("config", {})
                self._contrastive_encoder = ContrastiveEncoder(
                    in_channels=4,
                    backbone=config.get("backbone", "medium"),
                    projection_dim=config.get("projection_dim", 128),
                    hidden_dim=config.get("hidden_dim", 256),
                    base_channels=config.get("base_channels", 64),
                )
                self._contrastive_encoder.load_state_dict(
                    checkpoint["encoder_state_dict"], strict=False
                )
                self._contrastive_encoder.to(self.device)
                self._contrastive_encoder.eval()
                logger.info(f"Loaded contrastive encoder from {contrastive_path}")
            except Exception as e:
                logger.warning(f"Failed to load contrastive model: {e}")
                self._contrastive_encoder = None

        # Load RL agent
        if rl_path.exists():
            try:
                checkpoint = torch.load(rl_path, map_location=self.device, weights_only=False)
                algorithm = checkpoint.get("algorithm", "actor_critic")
                config = checkpoint.get("config", {})
                self._rl_algorithm = algorithm

                agent_cls = {
                    "dqn": DQNAgent,
                    "policy_gradient": PolicyGradientAgent,
                    "actor_critic": ActorCriticAgent,
                }.get(algorithm, ActorCriticAgent)

                self._rl_agent = agent_cls(
                    in_channels=4,
                    num_actions=3,
                    backbone=config.get("backbone", "medium"),
                    hidden_dim=config.get("hidden_dim", 256),
                    base_channels=checkpoint.get("base_channels", 64),
                )
                self._rl_agent.load_state_dict(
                    checkpoint["agent_state_dict"], strict=False
                )
                self._rl_agent.to(self.device)
                self._rl_agent.eval()
                logger.info(f"Loaded RL agent ({algorithm}) from {rl_path}")
            except Exception as e:
                logger.warning(f"Failed to load RL model: {e}")
                self._rl_agent = None

        self._models_loaded = (
            self._contrastive_encoder is not None or self._rl_agent is not None
        )
        if not self._models_loaded:
            logger.warning(
                "No trained models found. Run training first: "
                "metapathpredict train --pipeline full"
            )

    def _predict_contrastive(self, sequence: str) -> MethodPrediction:
        """Run contrastive encoder inference on a sequence."""
        x = encode_sequence(sequence).to(self.device)

        with torch.no_grad():
            embeddings = self._contrastive_encoder.get_embeddings(x)
            # Use encoder's classifier head if available, otherwise use embedding similarity
            if hasattr(self._contrastive_encoder.encoder, "classifier"):
                logits = self._contrastive_encoder.encoder.classifier(embeddings)
            else:
                # Fallback: linear probe not available, use projection norm as proxy
                logits = self._contrastive_encoder.encoder(x)
                if logits.shape[-1] != 3:
                    # No classifier head — return uniform with low confidence
                    return MethodPrediction(
                        method=MethodType.CONTRASTIVE,
                        predicted_class=ClassLabel.BACTERIA,
                        confidence=0.34,
                        probabilities=ClassProbabilities(
                            bacteria=0.34, eukaryotic=0.33, virus=0.33
                        ),
                    )

            probs = F.softmax(logits, dim=1).squeeze(0)

        probs_dict = {
            ClassLabel.BACTERIA: probs[0].item(),
            ClassLabel.EUKARYOTIC: probs[1].item(),
            ClassLabel.VIRUS: probs[2].item(),
        }
        predicted = max(probs_dict, key=probs_dict.get)

        return MethodPrediction(
            method=MethodType.CONTRASTIVE,
            predicted_class=predicted,
            confidence=probs_dict[predicted],
            probabilities=ClassProbabilities(
                bacteria=probs_dict[ClassLabel.BACTERIA],
                eukaryotic=probs_dict[ClassLabel.EUKARYOTIC],
                virus=probs_dict[ClassLabel.VIRUS],
            ),
        )

    def _predict_rl(self, sequence: str) -> MethodPrediction:
        """Run RL agent inference on a sequence."""
        x = encode_sequence(sequence).to(self.device)

        with torch.no_grad():
            if self._rl_algorithm == "dqn":
                q_values = self._rl_agent(x)
                probs = F.softmax(q_values, dim=1).squeeze(0)
            elif self._rl_algorithm == "policy_gradient":
                action_probs, _ = self._rl_agent(x)
                probs = action_probs.squeeze(0)
            else:  # actor_critic
                logits, _ = self._rl_agent(x)
                probs = F.softmax(logits, dim=1).squeeze(0)

        probs_dict = {
            ClassLabel.BACTERIA: probs[0].item(),
            ClassLabel.EUKARYOTIC: probs[1].item(),
            ClassLabel.VIRUS: probs[2].item(),
        }
        predicted = max(probs_dict, key=probs_dict.get)

        return MethodPrediction(
            method=MethodType.REINFORCEMENT,
            predicted_class=predicted,
            confidence=probs_dict[predicted],
            probabilities=ClassProbabilities(
                bacteria=probs_dict[ClassLabel.BACTERIA],
                eukaryotic=probs_dict[ClassLabel.EUKARYOTIC],
                virus=probs_dict[ClassLabel.VIRUS],
            ),
        )

    def _predict(self, method: MethodType, sequence: str) -> MethodPrediction:
        """Route prediction to the appropriate model."""
        if method == MethodType.CONTRASTIVE and self._contrastive_encoder is not None:
            return self._predict_contrastive(sequence)
        elif method == MethodType.REINFORCEMENT and self._rl_agent is not None:
            return self._predict_rl(sequence)
        else:
            # Model not available — return low-confidence fallback
            return MethodPrediction(
                method=method,
                predicted_class=ClassLabel.BACTERIA,
                confidence=0.34,
                probabilities=ClassProbabilities(
                    bacteria=0.34, eukaryotic=0.33, virus=0.33
                ),
            )

    async def get_predictions(
        self,
        sample_id: int,
        methods: Optional[list[MethodType]] = None,
    ) -> Optional[PredictionComparison]:
        """Get predictions for a sample from all methods."""
        sample = await self.sample_service.get_sample(sample_id)
        if not sample:
            return None

        if methods is None:
            methods = [MethodType.CONTRASTIVE, MethodType.REINFORCEMENT]

        predictions = []
        for method in methods:
            pred = self._predict(method, sample.sequence)
            predictions.append(pred)

        # Compute consensus
        class_votes: dict[ClassLabel, int] = {}
        for pred in predictions:
            cls = pred.predicted_class
            class_votes[cls] = class_votes.get(cls, 0) + 1

        consensus = max(class_votes, key=class_votes.get) if class_votes else None
        max_votes = max(class_votes.values()) if class_votes else 0
        agreement = max_votes / len(predictions) if predictions else 0

        return PredictionComparison(
            sample_id=sample_id,
            sample_name=sample.name,
            true_label=sample.true_label,
            predictions=predictions,
            consensus=consensus,
            agreement=agreement,
        )

    async def predict_sequence(
        self,
        sequence: str,
        methods: list[MethodType],
    ) -> PredictionComparison:
        """Predict class for a new sequence."""
        temp_sample = await self.sample_service.create_sample(
            type(
                "SampleCreate",
                (),
                {
                    "model_dump": lambda self: {
                        "name": "User upload",
                        "sequence": sequence,
                        "source": "upload",
                    }
                },
            )()
        )

        return await self.get_predictions(temp_sample.id, methods)

    async def get_fragment_predictions(
        self,
        sample_id: int,
        method: MethodType,
        fragment_size: int,
    ) -> list[MethodPrediction]:
        """Get predictions for each fragment of a sample."""
        sample = await self.sample_service.get_sample(sample_id)
        if not sample:
            return []

        sequence = sample.sequence
        num_fragments = max(1, len(sequence) // fragment_size)
        predictions = []

        for i in range(num_fragments):
            start = i * fragment_size
            fragment = sequence[start : start + fragment_size]
            pred = self._predict(method, fragment)
            predictions.append(pred)

        return predictions

    async def get_disagreements(
        self,
        min_disagreement: float,
        limit: int,
    ) -> list[PredictionComparison]:
        """Find samples where methods disagree."""
        results = []
        samples = await self.sample_service.list_samples(page_size=100)

        for sample in samples.items:
            comparison = await self.get_predictions(sample.id)
            if comparison and comparison.agreement < (1 - min_disagreement):
                results.append(comparison)
                if len(results) >= limit:
                    break

        return results

    async def get_by_class(
        self,
        class_label: ClassLabel,
        method: MethodType,
        confidence_min: float,
        confidence_max: float,
        page: int,
        page_size: int,
    ) -> list[PredictionComparison]:
        """Get predictions filtered by predicted class."""
        results = []
        samples = await self.sample_service.list_samples(page=page, page_size=page_size)

        for sample in samples.items:
            comparison = await self.get_predictions(sample.id, [method])
            if comparison:
                pred = comparison.predictions[0]
                if (
                    pred.predicted_class == class_label
                    and confidence_min <= pred.confidence <= confidence_max
                ):
                    results.append(comparison)

        return results
