"""
Sequence preprocessing utilities for MetaPathPredict.

This module provides unified preprocessing functions for DNA sequences,
including validation, cleaning, fragmentation, and encoding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

# Valid nucleotides for DNA sequences
VALID_NUCLEOTIDES = frozenset("ACGT")
NUCLEOTIDE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
IDX_TO_NUCLEOTIDE = {0: "A", 1: "C", 2: "G", 3: "T"}
COMPLEMENT_MAP = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}


@dataclass
class SequenceStats:
    """Statistics about a processed sequence."""
    
    original_length: int
    cleaned_length: int
    valid_nt_count: int
    n_count: int
    gc_content: float
    is_valid: bool
    
    @property
    def valid_ratio(self) -> float:
        """Ratio of valid nucleotides."""
        return self.valid_nt_count / self.cleaned_length if self.cleaned_length > 0 else 0.0
    
    @property
    def n_ratio(self) -> float:
        """Ratio of N (unknown) nucleotides."""
        return self.n_count / self.cleaned_length if self.cleaned_length > 0 else 0.0


class OneHotEncoder:
    """
    One-hot encoder for DNA sequences.
    
    Encodes nucleotides as:
        A -> [1, 0, 0, 0]
        C -> [0, 1, 0, 0]
        G -> [0, 0, 1, 0]
        T -> [0, 0, 0, 1]
        N/other -> [0.25, 0.25, 0.25, 0.25] (uniform distribution - Bayesian approach)
    
    The uniform encoding for N represents maximum uncertainty about the nucleotide,
    which is biologically correct as N means "unknown" not "absent".
    """
    
    def __init__(self, include_n_channel: bool = False, n_encoding: str = "uniform"):
        """
        Initialize encoder.
        
        Args:
            include_n_channel: If True, add 5th channel for N nucleotides.
            n_encoding: How to encode N nucleotides:
                - "uniform": [0.25, 0.25, 0.25, 0.25] (default, recommended)
                - "zeros": [0, 0, 0, 0] (legacy gap encoding)
                - "channel": use 5th channel (requires include_n_channel=True)
        """
        self.include_n_channel = include_n_channel
        self.n_encoding = n_encoding
        self.num_channels = 5 if include_n_channel else 4
    
    def encode(self, sequence: str) -> NDArray[np.float32]:
        """
        Encode a DNA sequence to one-hot representation.
        
        Args:
            sequence: DNA sequence string.
        
        Returns:
            One-hot encoded array of shape (seq_len, num_channels).
        """
        seq_upper = sequence.upper()
        encoded = np.zeros((len(seq_upper), self.num_channels), dtype=np.float32)
        
        for i, nt in enumerate(seq_upper):
            if nt in NUCLEOTIDE_TO_IDX:
                encoded[i, NUCLEOTIDE_TO_IDX[nt]] = 1.0
            elif nt == "N" or nt not in NUCLEOTIDE_TO_IDX:
                # Handle unknown nucleotides (N, R, Y, W, S, M, K, etc.)
                if self.include_n_channel and self.n_encoding == "channel":
                    encoded[i, 4] = 1.0
                elif self.n_encoding == "uniform":
                    # Uniform distribution - biologically correct for unknown
                    encoded[i, :4] = 0.25
                # else: zeros (legacy gap encoding)
        
        return encoded
    
    def encode_batch(self, sequences: Sequence[str]) -> NDArray[np.float32]:
        """
        Encode multiple sequences (must be same length).
        
        Args:
            sequences: List of DNA sequences of equal length.
        
        Returns:
            Batch of one-hot encoded arrays, shape (batch, seq_len, num_channels).
        """
        if not sequences:
            raise ValueError("Empty sequence list")
        
        seq_len = len(sequences[0])
        if not all(len(s) == seq_len for s in sequences):
            raise ValueError("All sequences must have the same length")
        
        batch = np.zeros((len(sequences), seq_len, self.num_channels), dtype=np.float32)
        
        for i, seq in enumerate(sequences):
            batch[i] = self.encode(seq)
        
        return batch
    
    def decode(self, encoded: NDArray[np.float32], threshold: float = 0.5) -> str:
        """
        Decode one-hot representation back to sequence string.
        
        Args:
            encoded: One-hot encoded array of shape (seq_len, num_channels).
            threshold: Threshold for considering a position as valid.
        
        Returns:
            Decoded DNA sequence string.
        """
        sequence = []
        
        for pos in encoded:
            max_idx = np.argmax(pos)
            if pos[max_idx] >= threshold and max_idx < 4:
                sequence.append(IDX_TO_NUCLEOTIDE[max_idx])
            else:
                sequence.append("N")
        
        return "".join(sequence)


class SequencePreprocessor:
    """
    Preprocessor for DNA sequences with validation, cleaning, and fragmentation.
    """
    
    def __init__(
        self,
        min_length: int = 100,
        min_valid_ratio: float = 0.8,
        max_n_ratio: float = 0.1,
        replace_invalid_with: str = "N",
    ):
        """
        Initialize preprocessor.
        
        Args:
            min_length: Minimum sequence length to accept.
            min_valid_ratio: Minimum ratio of valid nucleotides (ACGT).
            max_n_ratio: Maximum ratio of N nucleotides allowed.
            replace_invalid_with: Character to replace invalid nucleotides with.
        """
        self.min_length = min_length
        self.min_valid_ratio = min_valid_ratio
        self.max_n_ratio = max_n_ratio
        self.replace_invalid_with = replace_invalid_with
        
        # Regex for valid nucleotides
        self._valid_pattern = re.compile(r"[ACGT]", re.IGNORECASE)
        self._invalid_pattern = re.compile(r"[^ACGTN]", re.IGNORECASE)
    
    def clean(self, sequence: str) -> str:
        """
        Clean a DNA sequence by replacing invalid characters.
        
        Args:
            sequence: Raw DNA sequence.
        
        Returns:
            Cleaned sequence with invalid characters replaced.
        """
        seq_upper = sequence.upper()
        return self._invalid_pattern.sub(self.replace_invalid_with, seq_upper)
    
    def validate(self, sequence: str) -> tuple[bool, SequenceStats]:
        """
        Validate a DNA sequence.
        
        Args:
            sequence: DNA sequence to validate.
        
        Returns:
            Tuple of (is_valid, stats).
        """
        seq_upper = sequence.upper()
        length = len(seq_upper)
        
        # Count nucleotides
        valid_count = sum(1 for nt in seq_upper if nt in VALID_NUCLEOTIDES)
        n_count = seq_upper.count("N")
        g_count = seq_upper.count("G")
        c_count = seq_upper.count("C")
        
        gc_content = (g_count + c_count) / length if length > 0 else 0.0
        
        # Check validity
        is_valid = (
            length >= self.min_length
            and (valid_count / length if length > 0 else 0) >= self.min_valid_ratio
            and (n_count / length if length > 0 else 0) <= self.max_n_ratio
        )
        
        stats = SequenceStats(
            original_length=len(sequence),
            cleaned_length=length,
            valid_nt_count=valid_count,
            n_count=n_count,
            gc_content=gc_content,
            is_valid=is_valid,
        )
        
        return is_valid, stats
    
    def process(self, sequence: str) -> tuple[str | None, SequenceStats]:
        """
        Clean and validate a sequence.
        
        Args:
            sequence: Raw DNA sequence.
        
        Returns:
            Tuple of (cleaned_sequence or None if invalid, stats).
        """
        cleaned = self.clean(sequence)
        is_valid, stats = self.validate(cleaned)
        
        return (cleaned if is_valid else None, stats)
    
    def fragment(
        self,
        sequence: str,
        fragment_size: int,
        step_size: int | None = None,
        pad_last: bool = True,
        min_fragment_ratio: float = 0.95,
    ) -> Iterator[tuple[str, int, int]]:
        """
        Fragment a sequence using sliding window.
        
        Args:
            sequence: DNA sequence to fragment.
            fragment_size: Size of each fragment.
            step_size: Step size for sliding window. If None, uses fragment_size // 2.
            pad_last: Whether to pad the last fragment if shorter.
            min_fragment_ratio: Minimum ratio of fragment_size for last fragment.
        
        Yields:
            Tuples of (fragment, start_position, end_position).
        """
        if step_size is None:
            step_size = fragment_size // 2
        
        seq_len = len(sequence)
        
        if seq_len < fragment_size:
            if pad_last and seq_len >= int(fragment_size * min_fragment_ratio):
                # Pad short sequence
                padded = sequence + "N" * (fragment_size - seq_len)
                yield padded, 0, seq_len
            return
        
        # Generate fragments
        start = 0
        while start < seq_len:
            end = min(start + fragment_size, seq_len)
            fragment = sequence[start:end]
            
            if len(fragment) == fragment_size:
                yield fragment, start, end
            elif pad_last and len(fragment) >= int(fragment_size * min_fragment_ratio):
                # Pad last fragment
                padded = fragment + "N" * (fragment_size - len(fragment))
                yield padded, start, seq_len
            
            start += step_size
            
            # Avoid duplicate last fragment
            if end >= seq_len:
                break
    
    def get_fragments_with_quality(
        self,
        sequence: str,
        fragment_size: int,
        step_size: int | None = None,
        max_fragment_n_ratio: float = 0.1,
    ) -> list[tuple[str, int, int, float]]:
        """
        Get fragments with quality filtering.
        
        Args:
            sequence: DNA sequence.
            fragment_size: Fragment size.
            step_size: Step size for sliding window.
            max_fragment_n_ratio: Maximum N ratio allowed per fragment.
        
        Returns:
            List of (fragment, start, end, quality_score) tuples.
        """
        results = []
        
        for fragment, start, end in self.fragment(sequence, fragment_size, step_size):
            n_count = fragment.count("N")
            n_ratio = n_count / len(fragment)
            
            if n_ratio <= max_fragment_n_ratio:
                # Quality score based on valid nucleotide ratio
                quality = 1.0 - n_ratio
                results.append((fragment, start, end, quality))
        
        return results


def reverse_complement(sequence: str) -> str:
    """
    Get the reverse complement of a DNA sequence.
    
    Args:
        sequence: DNA sequence.
    
    Returns:
        Reverse complement sequence.
    """
    return "".join(COMPLEMENT_MAP.get(nt, "N") for nt in reversed(sequence.upper()))


def calculate_gc_content(sequence: str) -> float:
    """
    Calculate GC content of a sequence.
    
    Args:
        sequence: DNA sequence.
    
    Returns:
        GC content as a ratio (0-1).
    """
    seq_upper = sequence.upper()
    gc_count = seq_upper.count("G") + seq_upper.count("C")
    total = sum(1 for nt in seq_upper if nt in VALID_NUCLEOTIDES)
    return gc_count / total if total > 0 else 0.0


def sequence_to_kmers(sequence: str, k: int, stride: int = 1) -> list[str]:
    """
    Convert sequence to k-mers.
    
    Args:
        sequence: DNA sequence.
        k: K-mer size.
        stride: Stride for k-mer extraction.
    
    Returns:
        List of k-mers.
    """
    return [sequence[i:i + k] for i in range(0, len(sequence) - k + 1, stride)]


def batch_encode_sequences(
    sequences: list[str],
    max_length: int | None = None,
    padding: str = "right",
    truncation: str = "right",
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """
    Batch encode sequences with padding/truncation.
    
    Args:
        sequences: List of DNA sequences.
        max_length: Maximum length (uses max in batch if None).
        padding: Padding side ("right" or "left").
        truncation: Truncation side ("right" or "left").
    
    Returns:
        Tuple of (encoded_batch, lengths).
    """
    encoder = OneHotEncoder()
    
    if max_length is None:
        max_length = max(len(s) for s in sequences)
    
    batch = np.zeros((len(sequences), max_length, 4), dtype=np.float32)
    lengths = np.zeros(len(sequences), dtype=np.int64)
    
    for i, seq in enumerate(sequences):
        # Truncate if needed
        if len(seq) > max_length:
            if truncation == "right":
                seq = seq[:max_length]
            else:
                seq = seq[-max_length:]
        
        encoded = encoder.encode(seq)
        seq_len = len(seq)
        lengths[i] = seq_len
        
        # Apply padding
        if padding == "right":
            batch[i, :seq_len] = encoded
        else:
            batch[i, max_length - seq_len:] = encoded
    
    return batch, lengths


# ============================================================
# Standalone utility functions (for backwards compatibility)
# ============================================================

def clean_sequence(sequence: str, replace_invalid_with: str = "N") -> str:
    """
    Clean a DNA sequence by replacing invalid characters.
    
    Args:
        sequence: Raw DNA sequence.
        replace_invalid_with: Character to replace invalid nucleotides with.
    
    Returns:
        Cleaned sequence with invalid characters replaced.
    """
    seq_upper = sequence.upper()
    invalid_pattern = re.compile(r"[^ACGTN]", re.IGNORECASE)
    return invalid_pattern.sub(replace_invalid_with, seq_upper)


def validate_sequence(
    sequence: str,
    min_length: int = 100,
    min_valid_ratio: float = 0.8,
    max_n_ratio: float = 0.1,
) -> tuple[bool, SequenceStats]:
    """
    Validate a DNA sequence.
    
    Args:
        sequence: DNA sequence to validate.
        min_length: Minimum sequence length to accept.
        min_valid_ratio: Minimum ratio of valid nucleotides (ACGT).
        max_n_ratio: Maximum ratio of N nucleotides allowed.
    
    Returns:
        Tuple of (is_valid, stats).
    """
    preprocessor = SequencePreprocessor(
        min_length=min_length,
        min_valid_ratio=min_valid_ratio,
        max_n_ratio=max_n_ratio,
    )
    return preprocessor.validate(sequence)


def fragment_sequence(
    sequence: str,
    fragment_size: int,
    step_size: int | None = None,
    pad_last: bool = True,
) -> list[str]:
    """
    Fragment a sequence into fixed-size pieces.
    
    Args:
        sequence: DNA sequence to fragment.
        fragment_size: Size of each fragment.
        step_size: Step size for sliding window. If None, uses fragment_size // 2.
        pad_last: Whether to pad the last fragment if shorter.
    
    Returns:
        List of fragments.
    """
    preprocessor = SequencePreprocessor()
    return [frag for frag, _, _ in preprocessor.fragment(sequence, fragment_size, step_size, pad_last)]


def get_reverse_complement(sequence: str) -> str:
    """
    Get reverse complement of a DNA sequence.
    
    Alias for reverse_complement() for backwards compatibility.
    
    Args:
        sequence: DNA sequence.
    
    Returns:
        Reverse complement sequence.
    """
    return reverse_complement(sequence)
