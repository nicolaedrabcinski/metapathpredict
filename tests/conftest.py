"""
pytest configuration and fixtures.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import numpy as np


# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def seed():
    """Set random seeds for reproducibility."""
    seed_value = 42
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    return seed_value


@pytest.fixture
def device():
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_sequence():
    """Generate a sample DNA sequence."""
    return "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT" * 25  # 1000 nt


@pytest.fixture
def sample_sequences():
    """Generate multiple sample sequences."""
    bases = ["A", "C", "G", "T"]
    sequences = []
    for i in range(10):
        seq = "".join(np.random.choice(bases, size=1000))
        sequences.append(seq)
    return sequences


@pytest.fixture
def sample_batch():
    """Generate a sample batch of one-hot encoded sequences."""
    return torch.randn(8, 4, 1000)


@pytest.fixture
def sample_labels():
    """Generate sample labels for 3-class classification."""
    return torch.randint(0, 3, (8,))


@pytest.fixture
def tmp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config_dict():
    """Generate a sample configuration dictionary."""
    return {
        "paths": {
            "data_dir": "data",
            "weights_dir": "data/weights",
        },
        "data": {
            "fragment_size": 1000,
            "batch_size": 64,
            "train_split": 0.8,
            "val_split": 0.1,
            "test_split": 0.1,
        },
        "model": {
            "model_type": "multi_scale_cnn",
            "hidden_channels": 64,
            "num_classes": 3,
            "dropout": 0.2,
        },
        "training": {
            "epochs": 100,
            "learning_rate": 0.001,
            "optimizer": "adamw",
            "scheduler": "cosine_warmup",
        },
    }


@pytest.fixture
def sample_fasta_content():
    """Generate sample FASTA file content."""
    return """>seq1
ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT
>seq2
TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA
>seq3
AAAACCCCGGGGTTTTAAAACCCCGGGGTTTTAAAACCCC
"""


@pytest.fixture
def sample_fasta_file(tmp_dir, sample_fasta_content):
    """Create a temporary FASTA file."""
    fasta_path = tmp_dir / "test.fasta"
    fasta_path.write_text(sample_fasta_content)
    return fasta_path


@pytest.fixture
def sample_hdf5_file(tmp_dir):
    """Create a temporary HDF5 file with encoded sequences."""
    import h5py
    
    hdf5_path = tmp_dir / "test.hdf5"
    
    with h5py.File(hdf5_path, "w") as f:
        # Create random one-hot encoded sequences
        sequences = np.random.randint(0, 2, size=(100, 1000, 4)).astype(np.float32)
        labels = np.random.randint(0, 3, size=(100,))
        
        f.create_dataset("sequences", data=sequences)
        f.create_dataset("labels", data=labels)
    
    return hdf5_path


# Markers for conditional tests
def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on available resources."""
    skip_gpu = pytest.mark.skip(reason="GPU not available")
    
    for item in items:
        # Skip GPU tests if CUDA not available
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_gpu)
