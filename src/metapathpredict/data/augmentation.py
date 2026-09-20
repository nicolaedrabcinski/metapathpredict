"""
Data augmentation strategies for DNA sequences.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from metapathpredict.data.preprocessing import reverse_complement


@dataclass
class AugmentationResult:
    """Result of an augmentation operation."""
    sequence: str
    augmentation_name: str
    params: dict


class BaseAugmentation(ABC):
    """Base class for sequence augmentations."""

    def __init__(self, p: float = 0.5):
        """
        Initialize augmentation.

        Args:
            p: Probability of applying the augmentation.
        """
        self.p = p

    @abstractmethod
    def __call__(self, sequence: str) -> str:
        """Apply augmentation to sequence."""
        pass

    def maybe_apply(self, sequence: str) -> str:
        """Apply augmentation with probability p."""
        if random.random() < self.p:
            return self(sequence)
        return sequence


class ReverseComplement(BaseAugmentation):
    """Reverse complement augmentation."""

    def __call__(self, sequence: str) -> str:
        """Apply reverse complement."""
        return reverse_complement(sequence)


class RandomMutation(BaseAugmentation):
    """Random point mutation augmentation."""

    def __init__(self, p: float = 0.5, mutation_rate: float = 0.01):
        """
        Initialize mutation augmentation.

        Args:
            p: Probability of applying augmentation.
            mutation_rate: Probability of mutating each nucleotide.
        """
        super().__init__(p)
        self.mutation_rate = mutation_rate
        self.nucleotides = ["A", "C", "G", "T"]

    def __call__(self, sequence: str) -> str:
        """Apply random mutations."""
        seq_list = list(sequence.upper())

        for i in range(len(seq_list)):
            if seq_list[i] in self.nucleotides and random.random() < self.mutation_rate:
                # Mutate to a different nucleotide
                alternatives = [nt for nt in self.nucleotides if nt != seq_list[i]]
                seq_list[i] = random.choice(alternatives)

        return "".join(seq_list)


class RandomInsertion(BaseAugmentation):
    """Random nucleotide insertion augmentation."""

    def __init__(self, p: float = 0.5, insertion_rate: float = 0.005, max_insert_len: int = 3):
        """
        Initialize insertion augmentation.

        Args:
            p: Probability of applying augmentation.
            insertion_rate: Probability of insertion at each position.
            max_insert_len: Maximum length of insertion.
        """
        super().__init__(p)
        self.insertion_rate = insertion_rate
        self.max_insert_len = max_insert_len
        self.nucleotides = ["A", "C", "G", "T"]

    def __call__(self, sequence: str) -> str:
        """Apply random insertions."""
        result = []

        for nt in sequence:
            if random.random() < self.insertion_rate:
                # Insert random nucleotides
                insert_len = random.randint(1, self.max_insert_len)
                insertion = "".join(random.choices(self.nucleotides, k=insert_len))
                result.append(insertion)
            result.append(nt)

        return "".join(result)


class RandomDeletion(BaseAugmentation):
    """Random nucleotide deletion augmentation."""

    def __init__(self, p: float = 0.5, deletion_rate: float = 0.005):
        """
        Initialize deletion augmentation.

        Args:
            p: Probability of applying augmentation.
            deletion_rate: Probability of deleting each nucleotide.
        """
        super().__init__(p)
        self.deletion_rate = deletion_rate

    def __call__(self, sequence: str) -> str:
        """Apply random deletions."""
        return "".join(
            nt for nt in sequence
            if random.random() >= self.deletion_rate
        )


class RandomSubsequence(BaseAugmentation):
    """Extract random subsequence augmentation."""

    def __init__(self, p: float = 0.5, min_ratio: float = 0.8, max_ratio: float = 1.0):
        """
        Initialize subsequence augmentation.

        Args:
            p: Probability of applying augmentation.
            min_ratio: Minimum ratio of original length.
            max_ratio: Maximum ratio of original length.
        """
        super().__init__(p)
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __call__(self, sequence: str) -> str:
        """Extract random subsequence."""
        seq_len = len(sequence)
        target_len = int(seq_len * random.uniform(self.min_ratio, self.max_ratio))

        if target_len >= seq_len:
            return sequence

        start = random.randint(0, seq_len - target_len)
        return sequence[start:start + target_len]


class NoisyNucleotide(BaseAugmentation):
    """Replace random nucleotides with N (noise injection)."""

    def __init__(self, p: float = 0.5, noise_rate: float = 0.01):
        """
        Initialize noise augmentation.

        Args:
            p: Probability of applying augmentation.
            noise_rate: Probability of replacing each nucleotide with N.
        """
        super().__init__(p)
        self.noise_rate = noise_rate

    def __call__(self, sequence: str) -> str:
        """Apply noise injection."""
        return "".join(
            "N" if random.random() < self.noise_rate else nt
            for nt in sequence
        )


class SequenceShift(BaseAugmentation):
    """Circular shift of sequence."""

    def __init__(self, p: float = 0.5, max_shift_ratio: float = 0.1):
        """
        Initialize shift augmentation.

        Args:
            p: Probability of applying augmentation.
            max_shift_ratio: Maximum shift as ratio of sequence length.
        """
        super().__init__(p)
        self.max_shift_ratio = max_shift_ratio

    def __call__(self, sequence: str) -> str:
        """Apply circular shift."""
        max_shift = int(len(sequence) * self.max_shift_ratio)
        if max_shift == 0:
            return sequence

        shift = random.randint(-max_shift, max_shift)
        return sequence[shift:] + sequence[:shift]


class SequenceAugmentation:
    """
    Compose multiple augmentations for DNA sequences.

    Example:
        augmentor = SequenceAugmentation([
            ReverseComplement(p=0.5),
            RandomMutation(p=0.3, mutation_rate=0.01),
        ])
        augmented = augmentor(sequence)
    """

    def __init__(
        self,
        augmentations: list[BaseAugmentation] | None = None,
        target_length: int | None = None,
    ):
        """
        Initialize augmentation pipeline.

        Args:
            augmentations: List of augmentations to apply.
            target_length: If set, truncate/pad to this length after augmentation.
        """
        self.augmentations = augmentations or []
        self.target_length = target_length

    def __call__(self, sequence: str) -> str:
        """Apply all augmentations in sequence."""
        result = sequence

        for aug in self.augmentations:
            result = aug.maybe_apply(result)

        # Ensure target length if specified
        if self.target_length is not None:
            result = self._ensure_length(result, self.target_length)

        return result

    def _ensure_length(self, sequence: str, target_length: int) -> str:
        """Ensure sequence is exactly target_length."""
        if len(sequence) > target_length:
            # Random crop
            start = random.randint(0, len(sequence) - target_length)
            return sequence[start:start + target_length]
        elif len(sequence) < target_length:
            # Pad with N
            return sequence + "N" * (target_length - len(sequence))
        return sequence

    @classmethod
    def default(cls, target_length: int | None = None) -> SequenceAugmentation:
        """Create default augmentation pipeline."""
        return cls(
            augmentations=[
                ReverseComplement(p=0.5),
                RandomMutation(p=0.2, mutation_rate=0.005),
                NoisyNucleotide(p=0.1, noise_rate=0.005),
            ],
            target_length=target_length,
        )

    @classmethod
    def strong(cls, target_length: int | None = None) -> SequenceAugmentation:
        """Create strong augmentation pipeline."""
        return cls(
            augmentations=[
                ReverseComplement(p=0.5),
                RandomMutation(p=0.3, mutation_rate=0.01),
                RandomInsertion(p=0.1, insertion_rate=0.003),
                RandomDeletion(p=0.1, deletion_rate=0.003),
                NoisyNucleotide(p=0.2, noise_rate=0.01),
                SequenceShift(p=0.2, max_shift_ratio=0.05),
            ],
            target_length=target_length,
        )


class MixUp:
    """
    MixUp augmentation for one-hot encoded sequences.

    Implements the MixUp technique from:
    "mixup: Beyond Empirical Risk Minimization" (Zhang et al., 2017)
    """

    def __init__(self, alpha: float = 0.2):
        """
        Initialize MixUp.

        Args:
            alpha: Beta distribution parameter. Higher = more mixing.
        """
        self.alpha = alpha

    def __call__(
        self,
        x1: NDArray[np.float32],
        x2: NDArray[np.float32],
        y1: NDArray[np.float32],
        y2: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], float]:
        """
        Apply MixUp to a pair of samples.

        Args:
            x1, x2: Input features (one-hot encoded).
            y1, y2: Labels (one-hot encoded).

        Returns:
            Mixed features, mixed labels, mixing coefficient lambda.
        """
        lam = np.random.beta(self.alpha, self.alpha)

        x_mixed = lam * x1 + (1 - lam) * x2
        y_mixed = lam * y1 + (1 - lam) * y2

        return x_mixed, y_mixed, lam


class CutMix:
    """
    CutMix augmentation for sequences.

    Adapts CutMix from images to sequences by cutting and pasting
    contiguous regions.
    """

    def __init__(self, alpha: float = 1.0):
        """
        Initialize CutMix.

        Args:
            alpha: Beta distribution parameter for cut ratio.
        """
        self.alpha = alpha

    def __call__(
        self,
        x1: NDArray[np.float32],
        x2: NDArray[np.float32],
        y1: NDArray[np.float32],
        y2: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], float]:
        """
        Apply CutMix to a pair of samples.

        Args:
            x1, x2: Input features of shape (seq_len, channels).
            y1, y2: Labels (one-hot encoded).

        Returns:
            Mixed features, mixed labels, mixing coefficient lambda.
        """
        seq_len = x1.shape[0]

        # Sample cut ratio
        lam = np.random.beta(self.alpha, self.alpha)
        cut_len = int(seq_len * (1 - lam))

        # Random cut position
        cut_start = np.random.randint(0, seq_len - cut_len + 1) if cut_len < seq_len else 0
        cut_end = cut_start + cut_len

        # Apply cut
        x_mixed = x1.copy()
        x_mixed[cut_start:cut_end] = x2[cut_start:cut_end]

        # Adjust lambda based on actual cut
        actual_lam = 1 - cut_len / seq_len
        y_mixed = actual_lam * y1 + (1 - actual_lam) * y2

        return x_mixed, y_mixed, actual_lam
