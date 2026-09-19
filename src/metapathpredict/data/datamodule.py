"""
DataModule for managing data loading and preprocessing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

try:
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from metapathpredict.config import Settings, DataConfig
from metapathpredict.data.dataset import (
    SequenceDataset,
    HDF5SequenceDataset,
    InMemoryHDF5Dataset,
)
from metapathpredict.data.augmentation import SequenceAugmentation

logger = logging.getLogger(__name__)


# Splits whose float32 data fits under this are held in RAM. Reading one sample from the
# HDF5 file decompresses its whole 1024-sample gzip chunk (~8MB for 500bp fragments), so
# shuffled random access from disk was slow enough to leave the GPU idle. Larger datasets
# still stream from disk.
IN_MEMORY_LIMIT_BYTES = 6 * 1024**3


def _open_split(path: Path) -> Dataset:
    import h5py

    with h5py.File(path, "r") as f:
        nbytes = int(np.prod(f["sequences"].shape)) * 4
    if nbytes <= IN_MEMORY_LIMIT_BYTES:
        logger.info(f"Loading {path.name} into memory ({nbytes / 1024**3:.2f} GB)")
        return InMemoryHDF5Dataset(str(path))
    logger.info(f"Streaming {path.name} from disk ({nbytes / 1024**3:.2f} GB > in-memory limit)")
    return HDF5SequenceDataset(str(path))


class SequenceDataModule:
    """
    DataModule for sequence classification.
    
    Handles:
    - Dataset creation and splitting
    - DataLoader configuration
    - Augmentation setup
    - Class weight computation
    
    Example:
        ```python
        datamodule = SequenceDataModule.from_hdf5(
            train_path="data/train.hdf5",
            val_path="data/val.hdf5",
            batch_size=64,
        )
        
        for batch in datamodule.train_dataloader():
            x, y = batch
            ...
        ```
    """
    
    def __init__(
        self,
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
        test_dataset: Dataset | None = None,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        drop_last: bool = True,
        prefetch_factor: int = 2,
    ):
        """
        Initialize DataModule.
        
        Args:
            train_dataset: Training dataset.
            val_dataset: Validation dataset.
            test_dataset: Test dataset.
            batch_size: Batch size for all loaders.
            num_workers: Number of data loading workers.
            pin_memory: Whether to pin memory for GPU transfer.
            persistent_workers: Keep workers alive between epochs.
            drop_last: Drop last incomplete batch.
            prefetch_factor: Number of batches to prefetch per worker.
        """
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        # ML-005: Disable pin_memory on CPU (useless and causes warning)
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.persistent_workers = persistent_workers and num_workers > 0
        self.drop_last = drop_last
        self.prefetch_factor = prefetch_factor if num_workers > 0 else None
        
        self._train_loader: DataLoader | None = None
        self._val_loader: DataLoader | None = None
        self._test_loader: DataLoader | None = None
    
    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        if self.train_dataset is None:
            raise ValueError("Training dataset not set")
        
        if self._train_loader is None:
            self._train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                drop_last=self.drop_last,
                prefetch_factor=self.prefetch_factor,
            )
        
        return self._train_loader
    
    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        if self.val_dataset is None:
            raise ValueError("Validation dataset not set")
        
        if self._val_loader is None:
            self._val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size * 2,  # Can use larger batch for eval
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                drop_last=False,
                prefetch_factor=self.prefetch_factor,
            )
        
        return self._val_loader
    
    def test_dataloader(self) -> DataLoader:
        """Get test dataloader."""
        if self.test_dataset is None:
            raise ValueError("Test dataset not set")
        
        if self._test_loader is None:
            self._test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size * 2,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                drop_last=False,
                prefetch_factor=self.prefetch_factor,
            )
        
        return self._test_loader
    
    def get_class_weights(self, device: str | torch.device = "cpu") -> torch.Tensor:
        """
        Compute class weights from training dataset.
        
        Args:
            device: Device to place weights on.
        
        Returns:
            Tensor of class weights.
        """
        if self.train_dataset is None:
            raise ValueError("Training dataset not set")
        
        if hasattr(self.train_dataset, "get_class_weights"):
            return self.train_dataset.get_class_weights().to(device)
        
        # Fallback: compute from labels
        labels = []
        for _, label in self.train_dataset:
            if isinstance(label, torch.Tensor):
                labels.append(label.item())
            else:
                labels.append(label)
        
        labels = torch.tensor(labels)
        unique, counts = torch.unique(labels, return_counts=True)
        weights = 1.0 / counts.float()
        weights = weights / weights.sum() * len(unique)
        
        return weights.to(device)
    
    @classmethod
    def from_hdf5(
        cls,
        train_path: str | Path,
        val_path: str | Path | None = None,
        test_path: str | Path | None = None,
        in_memory: bool = True,
        augmentation: SequenceAugmentation | None = None,
        **kwargs,
    ) -> "SequenceDataModule":
        """
        Create DataModule from HDF5 files.
        
        Args:
            train_path: Path to training HDF5 file.
            val_path: Path to validation HDF5 file.
            test_path: Path to test HDF5 file.
            in_memory: Whether to load datasets into memory.
            augmentation: Augmentation to apply to training data.
            **kwargs: Additional arguments for DataModule.
        
        Returns:
            SequenceDataModule instance.
        """
        DatasetClass = InMemoryHDF5Dataset if in_memory else HDF5SequenceDataset
        
        train_dataset = DatasetClass(
            train_path,
            augmentation=augmentation,
        )
        
        val_dataset = None
        if val_path and Path(val_path).exists():
            val_dataset = DatasetClass(val_path)
        
        test_dataset = None
        if test_path and Path(test_path).exists():
            test_dataset = DatasetClass(test_path)
        
        return cls(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            **kwargs,
        )
    
    @classmethod
    def from_single_hdf5(
        cls,
        hdf5_path: str | Path,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        in_memory: bool = True,
        stratified: bool = True,
        **kwargs,
    ) -> "SequenceDataModule":
        """
        Create DataModule from a single HDF5 file with automatic splitting.
        
        Args:
            hdf5_path: Path to HDF5 file.
            train_ratio: Fraction for training.
            val_ratio: Fraction for validation.
            test_ratio: Fraction for testing.
            seed: Random seed for splitting.
            in_memory: Whether to load into memory.
            stratified: Whether to use stratified split (preserves class proportions).
            **kwargs: Additional arguments.
        
        Returns:
            SequenceDataModule instance.
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01
        
        DatasetClass = InMemoryHDF5Dataset if in_memory else HDF5SequenceDataset
        full_dataset = DatasetClass(hdf5_path)
        
        total_size = len(full_dataset)
        
        # Try stratified split if sklearn available and requested
        if stratified and HAS_SKLEARN:
            try:
                # Extract all labels for stratification
                all_labels = []
                for i in range(total_size):
                    _, label = full_dataset[i]
                    if isinstance(label, torch.Tensor):
                        all_labels.append(label.item())
                    else:
                        all_labels.append(int(label))
                
                all_labels = np.array(all_labels)
                indices = np.arange(total_size)
                
                # First split: train vs (val + test)
                train_idx, temp_idx = train_test_split(
                    indices,
                    test_size=(val_ratio + test_ratio),
                    stratify=all_labels,
                    random_state=seed,
                )
                
                # Second split: val vs test
                temp_labels = all_labels[temp_idx]
                relative_test_ratio = test_ratio / (val_ratio + test_ratio)
                val_idx, test_idx = train_test_split(
                    temp_idx,
                    test_size=relative_test_ratio,
                    stratify=temp_labels,
                    random_state=seed,
                )
                
                train_dataset = Subset(full_dataset, train_idx.tolist())
                val_dataset = Subset(full_dataset, val_idx.tolist())
                test_dataset = Subset(full_dataset, test_idx.tolist())
                
                logger.info(f"Using stratified split: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
                
                return cls(
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    test_dataset=test_dataset,
                    **kwargs,
                )
            except Exception as e:
                logger.warning(f"Stratified split failed ({e}), falling back to random split")
        
        # Fallback to random split
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        test_size = total_size - train_size - val_size
        
        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=generator,
        )
        
        return cls(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            **kwargs,
        )
    
    @classmethod
    def from_config(cls, config: Settings) -> "SequenceDataModule":
        """
        Create DataModule from Settings configuration.

        If pre-split files (train/val/test) exist, uses them directly.
        Otherwise falls back to loading a single train file and auto-splitting.

        Args:
            config: Settings object.

        Returns:
            SequenceDataModule instance.
        """
        data_config = config.data
        paths = config.paths
        training_config = config.training

        frag = data_config.default_fragment_size
        train_path = paths.datasets_dir / f"encoded_train_{frag}.hdf5"
        val_path = paths.datasets_dir / f"encoded_val_{frag}.hdf5"
        test_path = paths.datasets_dir / f"encoded_test_{frag}.hdf5"

        if not train_path.exists():
            raise FileNotFoundError(f"Training dataset not found: {train_path}")

        # If all three split files exist, use them directly
        if val_path.exists() and test_path.exists():
            train_dataset = _open_split(train_path)
            val_dataset = _open_split(val_path)
            test_dataset = _open_split(test_path)

            logger.info(
                f"Using pre-split data: train={len(train_dataset)}, "
                f"val={len(val_dataset)}, test={len(test_dataset)}"
            )

            return cls(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                batch_size=training_config.batch_size,
                num_workers=data_config.num_workers,
            )

        # Fallback: single file with auto-split
        return cls.from_single_hdf5(
            hdf5_path=train_path,
            train_ratio=data_config.train_ratio,
            val_ratio=data_config.val_ratio,
            test_ratio=data_config.test_ratio,
            in_memory=True,
            batch_size=training_config.batch_size,
            num_workers=data_config.num_workers,
        )
    
    @property
    def num_train_samples(self) -> int:
        """Number of training samples."""
        return len(self.train_dataset) if self.train_dataset else 0
    
    @property
    def num_val_samples(self) -> int:
        """Number of validation samples."""
        return len(self.val_dataset) if self.val_dataset else 0
    
    @property
    def num_test_samples(self) -> int:
        """Number of test samples."""
        return len(self.test_dataset) if self.test_dataset else 0
    
    def get_sample_batch(self, split: Literal["train", "val", "test"] = "train") -> tuple:
        """
        Get a sample batch for debugging.
        
        Args:
            split: Which split to sample from.
        
        Returns:
            Single batch tuple.
        """
        if split == "train":
            loader = self.train_dataloader()
        elif split == "val":
            loader = self.val_dataloader()
        else:
            loader = self.test_dataloader()
        
        return next(iter(loader))
