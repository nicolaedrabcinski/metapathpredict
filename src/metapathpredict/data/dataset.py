"""
PyTorch Dataset classes for sequence data.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Sequence

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from metapathpredict.data.preprocessing import (
    OneHotEncoder,
    SequencePreprocessor,
    reverse_complement,
)
from metapathpredict.data.augmentation import SequenceAugmentation


class SequenceDataset(Dataset):
    """
    Dataset for DNA sequences with on-the-fly encoding.
    
    Supports:
    - On-the-fly one-hot encoding
    - Data augmentation
    - Reverse complement augmentation
    - Lazy loading from FASTA files
    """
    
    def __init__(
        self,
        sequences: Sequence[str],
        labels: Sequence[int],
        seq_length: int = 1000,
        augmentation: SequenceAugmentation | None = None,
        use_reverse_complement: bool = True,
        rc_probability: float = 0.5,
        encoder: OneHotEncoder | None = None,
    ):
        """
        Initialize dataset.
        
        Args:
            sequences: List of DNA sequences.
            labels: List of class labels (integers).
            seq_length: Expected sequence length.
            augmentation: Optional augmentation pipeline.
            use_reverse_complement: Whether to randomly use reverse complement.
            rc_probability: Probability of using reverse complement.
            encoder: One-hot encoder instance.
        """
        assert len(sequences) == len(labels), "Sequences and labels must have same length"
        
        self.sequences = sequences
        self.labels = labels
        self.seq_length = seq_length
        self.augmentation = augmentation
        self.use_reverse_complement = use_reverse_complement
        self.rc_probability = rc_probability
        self.encoder = encoder or OneHotEncoder()
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index.
        
        Returns:
            Tuple of (encoded_sequence, label).
        """
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Apply augmentation if available
        if self.augmentation is not None:
            sequence = self.augmentation(sequence)
        
        # Random reverse complement
        if self.use_reverse_complement and random.random() < self.rc_probability:
            sequence = reverse_complement(sequence)
        
        # Ensure correct length
        sequence = self._ensure_length(sequence)
        
        # Encode
        encoded = self.encoder.encode(sequence)
        
        return torch.from_numpy(encoded), torch.tensor(label, dtype=torch.long)
    
    def _ensure_length(self, sequence: str) -> str:
        """Ensure sequence is exactly seq_length."""
        if len(sequence) > self.seq_length:
            # Random crop
            start = random.randint(0, len(sequence) - self.seq_length)
            return sequence[start:start + self.seq_length]
        elif len(sequence) < self.seq_length:
            # Pad with N
            return sequence + "N" * (self.seq_length - len(sequence))
        return sequence
    
    @classmethod
    def from_fasta_files(
        cls,
        file_paths: dict[str, Path],
        seq_length: int = 1000,
        fragments_per_class: int | None = None,
        preprocessor: SequencePreprocessor | None = None,
        **kwargs,
    ) -> "SequenceDataset":
        """
        Create dataset from FASTA files.
        
        Args:
            file_paths: Dict mapping class names to FASTA file paths.
            seq_length: Sequence/fragment length.
            fragments_per_class: Number of fragments per class (None for all).
            preprocessor: Sequence preprocessor.
            **kwargs: Additional arguments for SequenceDataset.
        
        Returns:
            SequenceDataset instance.
        """
        from Bio import SeqIO
        
        preprocessor = preprocessor or SequencePreprocessor()
        
        sequences = []
        labels = []
        class_to_idx = {name: i for i, name in enumerate(file_paths.keys())}
        
        for class_name, file_path in file_paths.items():
            class_idx = class_to_idx[class_name]
            class_sequences = []
            
            # Read sequences
            for record in SeqIO.parse(file_path, "fasta"):
                seq_str = str(record.seq)
                cleaned, stats = preprocessor.process(seq_str)
                
                if cleaned is not None:
                    # Fragment if longer than seq_length
                    if len(cleaned) >= seq_length:
                        for fragment, _, _, _ in preprocessor.get_fragments_with_quality(
                            cleaned, seq_length
                        ):
                            class_sequences.append(fragment)
            
            # Sample if needed
            if fragments_per_class and len(class_sequences) > fragments_per_class:
                class_sequences = random.sample(class_sequences, fragments_per_class)
            
            sequences.extend(class_sequences)
            labels.extend([class_idx] * len(class_sequences))
        
        return cls(sequences, labels, seq_length=seq_length, **kwargs)


class HDF5SequenceDataset(Dataset):
    """
    Memory-efficient dataset that reads from HDF5 files.
    
    Supports:
    - Lazy loading (doesn't load entire dataset into memory)
    - Caching for repeated access
    - Memory mapping for large datasets
    """
    
    def __init__(
        self,
        hdf5_path: str | Path,
        sequences_key: str = "sequences",
        labels_key: str = "labels",
        cache_size: int = 10000,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        """
        Initialize HDF5 dataset.
        
        Args:
            hdf5_path: Path to HDF5 file.
            sequences_key: Key for sequences dataset in HDF5.
            labels_key: Key for labels dataset in HDF5.
            cache_size: Number of samples to cache in memory.
            transform: Optional transform function.
        """
        self.hdf5_path = Path(hdf5_path)
        self.sequences_key = sequences_key
        self.labels_key = labels_key
        self.transform = transform
        
        # Get dataset info without loading
        with h5py.File(self.hdf5_path, "r") as f:
            self._length = len(f[sequences_key])
            self._shape = f[sequences_key].shape
            self._dtype = f[sequences_key].dtype
            
            # Load metadata
            self.metadata = dict(f.attrs)
        
        # Simple LRU cache
        self._cache: dict[int, tuple[np.ndarray, int]] = {}
        self._cache_size = cache_size
        self._cache_order: list[int] = []
        
        # File handle (lazy initialization)
        self._file: h5py.File | None = None
    
    def __len__(self) -> int:
        return self._length
    
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Get a single sample."""
        # Check cache
        if idx in self._cache:
            sequence, label = self._cache[idx]
        else:
            # Load from file
            if self._file is None:
                self._file = h5py.File(self.hdf5_path, "r")
            
            sequence = self._file[self.sequences_key][idx]
            label = self._file[self.labels_key][idx]
            
            # Update cache
            self._update_cache(idx, (sequence, label))
        
        # Apply transform
        if self.transform is not None:
            sequence = self.transform(sequence)
        
        return (
            torch.from_numpy(sequence.astype(np.float32)),
            torch.tensor(label, dtype=torch.long),
        )
    
    def _update_cache(self, idx: int, data: tuple[np.ndarray, int]) -> None:
        """Update LRU cache."""
        if len(self._cache) >= self._cache_size:
            # Remove oldest entry
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]
        
        self._cache[idx] = data
        self._cache_order.append(idx)
    
    def __del__(self):
        """Close file handle."""
        if self._file is not None:
            self._file.close()
    
    @property
    def seq_length(self) -> int:
        """Get sequence length."""
        return self._shape[1]
    
    @property
    def num_classes(self) -> int:
        """Get number of classes from metadata."""
        return self.metadata.get("num_classes", 3)
    
    def get_class_weights(self) -> Tensor:
        """
        Calculate class weights for imbalanced data.
        
        Returns:
            Tensor of class weights.
        """
        with h5py.File(self.hdf5_path, "r") as f:
            labels = f[self.labels_key][:]
        
        unique, counts = np.unique(labels, return_counts=True)
        weights = 1.0 / counts
        weights = weights / weights.sum() * len(unique)
        
        return torch.tensor(weights, dtype=torch.float32)


class InMemoryHDF5Dataset(Dataset):
    """
    Dataset that loads entire HDF5 file into memory.
    
    Faster than HDF5SequenceDataset for smaller datasets that fit in RAM.
    """
    
    def __init__(
        self,
        hdf5_path: str | Path,
        sequences_key: str = "sequences",
        labels_key: str = "labels",
        augmentation: Callable | None = None,
    ):
        """
        Initialize in-memory dataset.
        
        Args:
            hdf5_path: Path to HDF5 file.
            sequences_key: Key for sequences.
            labels_key: Key for labels.
            augmentation: Optional augmentation function.
        """
        self.augmentation = augmentation
        
        # Load entire dataset
        with h5py.File(hdf5_path, "r") as f:
            self.sequences = torch.from_numpy(f[sequences_key][:].astype(np.float32))
            self.labels = torch.from_numpy(f[labels_key][:].astype(np.int64))
            self.metadata = dict(f.attrs)
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Get a single sample."""
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        if self.augmentation is not None:
            sequence = self.augmentation(sequence)
        
        return sequence, label
    
    @property
    def seq_length(self) -> int:
        return self.sequences.shape[1]
    
    @property
    def num_classes(self) -> int:
        return self.metadata.get("num_classes", 3)
    
    def get_class_distribution(self) -> dict[int, int]:
        """Get class distribution."""
        unique, counts = torch.unique(self.labels, return_counts=True)
        return {int(c): int(n) for c, n in zip(unique, counts)}
    
    def get_class_weights(self) -> Tensor:
        """Calculate class weights."""
        unique, counts = torch.unique(self.labels, return_counts=True)
        weights = 1.0 / counts.float()
        weights = weights / weights.sum() * len(unique)
        return weights


class MixUpDataset(Dataset):
    """
    Dataset wrapper that applies MixUp augmentation.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        alpha: float = 0.2,
        num_classes: int = 3,
    ):
        """
        Initialize MixUp dataset.
        
        Args:
            dataset: Base dataset.
            alpha: MixUp alpha parameter.
            num_classes: Number of classes.
        """
        self.dataset = dataset
        self.alpha = alpha
        self.num_classes = num_classes
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Get mixed sample."""
        x1, y1 = self.dataset[idx]
        
        # Get random second sample
        idx2 = random.randint(0, len(self.dataset) - 1)
        x2, y2 = self.dataset[idx2]
        
        # Sample lambda
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Mix inputs
        x_mixed = lam * x1 + (1 - lam) * x2
        
        # Mix labels (one-hot)
        y1_onehot = torch.zeros(self.num_classes)
        y1_onehot[y1] = 1.0
        
        y2_onehot = torch.zeros(self.num_classes)
        y2_onehot[y2] = 1.0
        
        y_mixed = lam * y1_onehot + (1 - lam) * y2_onehot
        
        return x_mixed, y_mixed
