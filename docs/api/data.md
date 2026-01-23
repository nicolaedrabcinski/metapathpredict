# API Reference: Data

Data loading and preprocessing module.

## SequenceDataset

```python
class SequenceDataset(Dataset):
    """PyTorch Dataset for DNA sequences.
    
    Args:
        hdf5_path: Path to HDF5 file with encoded sequences.
        transform: Optional transform to apply to sequences.
        augmenter: Optional augmenter for data augmentation.
    
    Attributes:
        sequences: Encoded sequences tensor (N, 4, L).
        labels: Class labels tensor (N,).
        seq_ids: Sequence identifiers.
    
    Example:
        >>> dataset = SequenceDataset("train.hdf5")
        >>> seq, label = dataset[0]
        >>> seq.shape
        torch.Size([4, 500])
    """
    
    def __init__(
        self,
        hdf5_path: str,
        transform: Optional[Callable] = None,
        augmenter: Optional[SequenceAugmenter] = None,
    ):
        pass
    
    def __len__(self) -> int:
        """Return number of sequences."""
        pass
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get sequence and label by index."""
        pass
```

## SequenceDataModule

```python
class SequenceDataModule:
    """Data module for managing train/val/test splits.
    
    Args:
        data_dir: Directory containing HDF5 files.
        batch_size: Batch size for data loaders.
        num_workers: Number of data loading workers.
        pin_memory: Pin memory for GPU transfer.
        augmenter: Optional augmenter for training data.
    
    Example:
        >>> datamodule = SequenceDataModule("data/datasets/unified")
        >>> datamodule.setup()
        >>> train_loader = datamodule.train_dataloader()
    """
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        augmenter: Optional[SequenceAugmenter] = None,
    ):
        pass
    
    def setup(self, stage: Optional[str] = None):
        """Set up datasets for train/val/test."""
        pass
    
    def train_dataloader(self) -> DataLoader:
        """Return training data loader."""
        pass
    
    def val_dataloader(self) -> DataLoader:
        """Return validation data loader."""
        pass
    
    def test_dataloader(self) -> DataLoader:
        """Return test data loader."""
        pass
```

## SequenceAugmenter

```python
class SequenceAugmenter:
    """Augmentation transforms for DNA sequences.
    
    Args:
        crop_ratio: Ratio of sequence to keep when cropping (0.5-1.0).
        mask_ratio: Ratio of positions to mask (0.0-0.3).
        noise_std: Standard deviation of Gaussian noise.
        reverse_complement: Enable reverse complement augmentation.
    
    Example:
        >>> augmenter = SequenceAugmenter(crop_ratio=0.9)
        >>> augmented = augmenter(sequence)
    """
    
    def __init__(
        self,
        crop_ratio: float = 0.9,
        mask_ratio: float = 0.1,
        noise_std: float = 0.1,
        reverse_complement: bool = False,
    ):
        pass
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random augmentation."""
        pass
    
    @staticmethod
    def random_crop(x: torch.Tensor, crop_ratio: float) -> torch.Tensor:
        """Random crop and pad back to original length."""
        pass
    
    @staticmethod
    def random_mask(x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
        """Randomly mask positions with zeros."""
        pass
    
    @staticmethod
    def gaussian_noise(x: torch.Tensor, std: float) -> torch.Tensor:
        """Add Gaussian noise."""
        pass
    
    @staticmethod
    def reverse_complement(x: torch.Tensor) -> torch.Tensor:
        """Compute reverse complement of DNA sequence."""
        pass
```

## Preprocessing Functions

### encode_sequence

```python
def encode_sequence(
    sequence: str,
    max_length: int = 500,
    encoding: str = "onehot",
) -> torch.Tensor:
    """Encode DNA sequence to tensor.
    
    Args:
        sequence: DNA sequence string (A, T, G, C, N).
        max_length: Maximum sequence length.
        encoding: Encoding type ("onehot" or "kmer").
    
    Returns:
        Encoded tensor of shape (4, max_length) for onehot.
    
    Example:
        >>> tensor = encode_sequence("ATGC", max_length=100)
        >>> tensor.shape
        torch.Size([4, 100])
    """
    pass
```

### prepare_dataset

```python
def prepare_dataset(
    input_dir: str,
    output_dir: str,
    max_length: int = 500,
    min_length: int = 100,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, int]:
    """Prepare dataset from FASTA files.
    
    Args:
        input_dir: Directory with input FASTA files.
        output_dir: Directory for output HDF5 files.
        max_length: Maximum sequence length.
        min_length: Minimum sequence length.
        train_ratio: Training set ratio.
        val_ratio: Validation set ratio.
        test_ratio: Test set ratio.
        seed: Random seed.
    
    Returns:
        Dictionary with dataset statistics.
    
    Example:
        >>> stats = prepare_dataset("data/input", "data/output")
        >>> print(stats["train_samples"])
    """
    pass
```

### parse_fasta

```python
def parse_fasta(path: str) -> Iterator[Tuple[str, str]]:
    """Parse FASTA file.
    
    Args:
        path: Path to FASTA file.
    
    Yields:
        Tuples of (sequence_id, sequence).
    
    Example:
        >>> for seq_id, seq in parse_fasta("sequences.fasta"):
        ...     print(f"{seq_id}: {len(seq)}bp")
    """
    pass
```

### analyze_fasta

```python
def analyze_fasta(path: str) -> Dict[str, Any]:
    """Analyze FASTA file statistics.
    
    Args:
        path: Path to FASTA file.
    
    Returns:
        Dictionary with statistics:
        - count: Number of sequences
        - min_length: Minimum sequence length
        - max_length: Maximum sequence length
        - mean_length: Mean sequence length
        - gc_content: Average GC content
    
    Example:
        >>> stats = analyze_fasta("sequences.fasta")
        >>> print(f"Sequences: {stats['count']}")
    """
    pass
```

## Utility Functions

### balance_dataset

```python
def balance_dataset(
    sequences: torch.Tensor,
    labels: torch.Tensor,
    strategy: str = "undersample",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Balance dataset classes.
    
    Args:
        sequences: Sequence tensor.
        labels: Label tensor.
        strategy: "undersample" or "oversample".
    
    Returns:
        Balanced (sequences, labels) tuple.
    """
    pass
```

### filter_sequences

```python
def filter_sequences(
    sequences: List[str],
    max_n_ratio: float = 0.1,
    min_length: int = 100,
    max_length: int = 10000,
) -> List[str]:
    """Filter sequences by quality criteria.
    
    Args:
        sequences: List of sequences.
        max_n_ratio: Maximum ratio of N characters.
        min_length: Minimum sequence length.
        max_length: Maximum sequence length.
    
    Returns:
        Filtered list of sequences.
    """
    pass
```

### validate_dataset

```python
def validate_dataset(data_dir: str) -> Tuple[bool, List[str]]:
    """Validate dataset integrity.
    
    Args:
        data_dir: Dataset directory.
    
    Returns:
        Tuple of (is_valid, list_of_issues).
    """
    pass
```

## HDF5 Structure

Output HDF5 files have the following structure:

```
encoded_train.hdf5
├── sequences    # (N, 4, max_length) float32
├── labels       # (N,) int64
└── seq_ids      # (N,) string
```

## Example Usage

```python
from metapathpredict.data import (
    prepare_dataset,
    SequenceDataModule,
    SequenceAugmenter,
)

# Prepare data
prepare_dataset(
    input_dir="data/input",
    output_dir="data/output",
    max_length=500,
)

# Create augmenter
augmenter = SequenceAugmenter(
    crop_ratio=0.9,
    mask_ratio=0.1,
)

# Create data module
datamodule = SequenceDataModule(
    data_dir="data/output",
    batch_size=32,
    augmenter=augmenter,
)
datamodule.setup()

# Get data loaders
train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()

# Iterate
for batch, labels in train_loader:
    print(batch.shape)  # (32, 4, 500)
    break
```
