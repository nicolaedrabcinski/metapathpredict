# Data Preparation

This guide covers preparing your data for training with MetaPathPredict.

## Input Format

MetaPathPredict expects FASTA files with DNA sequences:

```
>seq_id_1 description
ATGCGATCGATCGATCGATCGATCGATCG...
>seq_id_2 description
GCTAGCTAGCTAGCTAGCTAGCTAGCTAG...
```

## Directory Structure

Organize your data by class:

```
data/input/
├── bacteria.fasta      # Bacterial sequences
├── viruses.fasta       # Viral sequences
└── eucaryotic.fasta    # Eukaryotic sequences
```

## Preparing Datasets

### Using CLI

```bash
metapathpredict prepare \
    --input data/input \
    --output data/datasets/unified \
    --max-length 500 \
    --min-length 100 \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1
```

### Using Python

```python
from metapathpredict.data import prepare_dataset

prepare_dataset(
    input_dir="data/input",
    output_dir="data/datasets/unified",
    max_length=500,
    min_length=100,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
)
```

## Output Format

The preparation creates HDF5 files:

```
data/datasets/unified/
├── encoded_train_500.hdf5   # Training data
├── encoded_val_500.hdf5     # Validation data
├── encoded_test_500.hdf5    # Test data
└── metadata.json            # Dataset metadata
```

### HDF5 Structure

```python
import h5py

with h5py.File("encoded_train_500.hdf5", "r") as f:
    sequences = f["sequences"][:]  # (N, 4, max_length)
    labels = f["labels"][:]        # (N,)
    seq_ids = f["seq_ids"][:]      # (N,) string IDs
```

## Encoding

### One-Hot Encoding

Default encoding (4 channels for A, T, G, C):

```python
# A = [1, 0, 0, 0]
# T = [0, 1, 0, 0]
# G = [0, 0, 1, 0]
# C = [0, 0, 0, 1]
# N = [0.25, 0.25, 0.25, 0.25]  # Ambiguous
```

### K-mer Encoding

Alternative encoding using k-mers:

```bash
metapathpredict prepare \
    --encoding kmer \
    --kmer-size 3
```

## Sequence Length

### Padding/Truncation

Sequences are padded or truncated to `max_length`:

```python
from metapathpredict.data import encode_sequence

# Pad short sequences
seq_short = "ATGC" * 10  # 40bp
encoded = encode_sequence(seq_short, max_length=500)  # (4, 500)

# Truncate long sequences
seq_long = "ATGC" * 200  # 800bp
encoded = encode_sequence(seq_long, max_length=500)  # (4, 500)
```

### Length Distribution

Check your data's length distribution:

```python
from metapathpredict.data import analyze_fasta

stats = analyze_fasta("data/input/bacteria.fasta")
print(f"Min length: {stats['min_length']}")
print(f"Max length: {stats['max_length']}")
print(f"Mean length: {stats['mean_length']:.1f}")
print(f"Sequences: {stats['count']}")
```

## Class Balancing

### Undersampling

```python
from metapathpredict.data import balance_dataset

# Balance to smallest class
balanced_data = balance_dataset(
    sequences,
    labels,
    strategy="undersample",
)
```

### Oversampling

```python
# Balance by oversampling minority classes
balanced_data = balance_dataset(
    sequences,
    labels,
    strategy="oversample",
)
```

### Class Weights

For imbalanced training:

```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    "balanced",
    classes=np.unique(labels),
    y=labels,
)

# Use in loss function
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
```

## Quality Filtering

### Remove Low-Quality Sequences

```python
from metapathpredict.data import filter_sequences

# Remove sequences with too many Ns
filtered = filter_sequences(
    sequences,
    max_n_ratio=0.1,  # Max 10% N's
    min_length=100,
    max_length=10000,
)
```

### Remove Duplicates

```bash
metapathpredict prepare \
    --input data/input \
    --remove-duplicates
```

## Data Augmentation

### Available Augmentations

| Augmentation | Description | Use Case |
|--------------|-------------|----------|
| Random Crop | Crop portion of sequence | General |
| Random Mask | Mask random positions | Contrastive |
| Reverse Complement | DNA reverse complement | Biology |
| Gaussian Noise | Add noise to encoding | Robustness |

### Configuration

```yaml
data:
  augmentation: true
  aug_crop_ratio: 0.9    # Keep 90% when cropping
  aug_mask_ratio: 0.1    # Mask 10% of positions
  aug_noise_std: 0.1     # Noise standard deviation
```

### Runtime Augmentation

```python
from metapathpredict.data import SequenceDataset, SequenceAugmenter

augmenter = SequenceAugmenter(
    crop_ratio=0.9,
    mask_ratio=0.1,
)

dataset = SequenceDataset(
    hdf5_path="encoded_train.hdf5",
    augmenter=augmenter,  # Applied during loading
)
```

## Large Datasets

### Memory-Mapped Loading

For datasets larger than RAM:

```python
from metapathpredict.data import LazySequenceDataset

# Memory-efficient loading
dataset = LazySequenceDataset(
    hdf5_path="large_dataset.hdf5",
    chunk_size=1000,
)
```

### Chunked Processing

```python
from metapathpredict.data import process_fasta_chunked

for chunk in process_fasta_chunked("large.fasta", chunk_size=10000):
    # Process chunk
    encoded = encode_sequences(chunk)
    save_to_hdf5(encoded, "output.hdf5", append=True)
```

## Validation

### Check Dataset

```python
from metapathpredict.data import validate_dataset

is_valid, issues = validate_dataset("data/datasets/unified")

if not is_valid:
    for issue in issues:
        print(f"Issue: {issue}")
```

### Expected Output

```
✓ Training set: 48000 samples
✓ Validation set: 6000 samples
✓ Test set: 6000 samples
✓ Classes: ['bacteria', 'eucaryotic', 'virus']
✓ Class distribution balanced
✓ No NaN values
✓ Encoding valid
```

## Next Steps

- [Training Guide](training.md) - Train your model
- [Configuration](../getting-started/configuration.md) - Data config options
- [API Reference](../api/data.md) - Data API documentation
