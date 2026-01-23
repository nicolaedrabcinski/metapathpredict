"""
Unit tests for preprocessing module.
"""

import numpy as np
import pytest

from metapathpredict.data.preprocessing import (
    COMPLEMENT_MAP,
    NUCLEOTIDE_TO_IDX,
    OneHotEncoder,
    SequencePreprocessor,
    SequenceStats,
    clean_sequence,
    fragment_sequence,
    get_reverse_complement,
    validate_sequence,
)


class TestConstants:
    """Tests for module constants."""

    def test_nucleotide_mapping(self):
        """Test nucleotide to index mapping."""
        assert NUCLEOTIDE_TO_IDX["A"] == 0
        assert NUCLEOTIDE_TO_IDX["C"] == 1
        assert NUCLEOTIDE_TO_IDX["G"] == 2
        assert NUCLEOTIDE_TO_IDX["T"] == 3
        assert len(NUCLEOTIDE_TO_IDX) == 4

    def test_complement_map(self):
        """Test complement nucleotide mapping."""
        assert COMPLEMENT_MAP["A"] == "T"
        assert COMPLEMENT_MAP["T"] == "A"
        assert COMPLEMENT_MAP["C"] == "G"
        assert COMPLEMENT_MAP["G"] == "C"
        assert COMPLEMENT_MAP["N"] == "N"


class TestSequenceStats:
    """Tests for SequenceStats dataclass."""

    def test_valid_ratio(self):
        """Test valid_ratio property."""
        stats = SequenceStats(
            original_length=100,
            cleaned_length=100,
            valid_nt_count=90,
            n_count=10,
            gc_content=0.5,
            is_valid=True,
        )
        assert stats.valid_ratio == 0.9

    def test_n_ratio(self):
        """Test n_ratio property."""
        stats = SequenceStats(
            original_length=100,
            cleaned_length=100,
            valid_nt_count=80,
            n_count=20,
            gc_content=0.5,
            is_valid=True,
        )
        assert stats.n_ratio == 0.2

    def test_zero_length_handling(self):
        """Test handling of zero-length sequences."""
        stats = SequenceStats(
            original_length=0,
            cleaned_length=0,
            valid_nt_count=0,
            n_count=0,
            gc_content=0.0,
            is_valid=False,
        )
        assert stats.valid_ratio == 0.0
        assert stats.n_ratio == 0.0


class TestOneHotEncoder:
    """Tests for OneHotEncoder."""

    def test_encode_simple_sequence(self):
        """Test encoding simple sequence."""
        encoder = OneHotEncoder()
        encoded = encoder.encode("ACGT")
        
        expected = np.array([
            [1, 0, 0, 0],  # A
            [0, 1, 0, 0],  # C
            [0, 0, 1, 0],  # G
            [0, 0, 0, 1],  # T
        ], dtype=np.float32)
        
        np.testing.assert_array_equal(encoded, expected)

    def test_encode_with_n(self):
        """Test encoding sequence with N - uses uniform distribution by default (BIO-002 fix)."""
        encoder = OneHotEncoder()  # default n_encoding="uniform"
        encoded = encoder.encode("ANG")
        
        assert encoded[0, 0] == 1.0  # A
        # N -> uniform [0.25, 0.25, 0.25, 0.25] (biologically correct for unknown)
        np.testing.assert_array_almost_equal(encoded[1], [0.25, 0.25, 0.25, 0.25])
        assert encoded[2, 2] == 1.0  # G
    
    def test_encode_with_n_zeros(self):
        """Test legacy zero encoding for N."""
        encoder = OneHotEncoder(n_encoding="zeros")
        encoded = encoder.encode("ANG")
        
        assert encoded[0, 0] == 1.0  # A
        assert encoded[1].sum() == 0.0  # N -> all zeros (legacy)
        assert encoded[2, 2] == 1.0  # G

    def test_encode_with_n_channel(self):
        """Test encoding with dedicated N channel."""
        encoder = OneHotEncoder(include_n_channel=True, n_encoding="channel")
        encoded = encoder.encode("ANT")
        
        assert encoded.shape == (3, 5)
        assert encoded[0, 0] == 1.0  # A
        assert encoded[1, 4] == 1.0  # N in 5th channel
        assert encoded[2, 3] == 1.0  # T

    def test_encode_lowercase(self):
        """Test encoding handles lowercase."""
        encoder = OneHotEncoder()
        encoded_lower = encoder.encode("acgt")
        encoded_upper = encoder.encode("ACGT")
        
        np.testing.assert_array_equal(encoded_lower, encoded_upper)

    def test_encode_batch(self):
        """Test batch encoding."""
        encoder = OneHotEncoder()
        sequences = ["ACGT", "TGCA", "AAAA"]
        encoded = encoder.encode_batch(sequences)
        
        assert encoded.shape == (3, 4, 4)
        
        # Check first sequence
        assert encoded[0, 0, 0] == 1.0  # A
        assert encoded[0, 1, 1] == 1.0  # C
        
        # Check second sequence
        assert encoded[1, 0, 3] == 1.0  # T
        assert encoded[1, 1, 2] == 1.0  # G

    def test_encode_batch_different_lengths_raises(self):
        """Test that batch encoding requires same length sequences."""
        encoder = OneHotEncoder()
        sequences = ["ACGT", "ACG"]  # Different lengths
        
        with pytest.raises(ValueError, match="same length"):
            encoder.encode_batch(sequences)

    def test_encode_batch_empty_raises(self):
        """Test that empty batch raises error."""
        encoder = OneHotEncoder()
        
        with pytest.raises(ValueError, match="Empty"):
            encoder.encode_batch([])

    def test_decode(self):
        """Test decoding back to sequence."""
        encoder = OneHotEncoder()
        original = "ACGT"
        encoded = encoder.encode(original)
        decoded = encoder.decode(encoded)
        
        assert decoded == original

    def test_decode_ambiguous(self):
        """Test decoding ambiguous positions."""
        encoder = OneHotEncoder()
        # Create ambiguous encoding (all zeros)
        ambiguous = np.zeros((3, 4), dtype=np.float32)
        decoded = encoder.decode(ambiguous)
        
        assert decoded == "NNN"


class TestCleanSequence:
    """Tests for clean_sequence function."""

    def test_clean_whitespace(self):
        """Test replacing whitespace with N."""
        result = clean_sequence("A C G T")
        # Whitespace is replaced with N (invalid character)
        assert result == "ANCNGNT"

    def test_clean_newlines(self):
        """Test replacing newlines with N."""
        result = clean_sequence("ACG\nTAC\nGTA")
        # Newlines are replaced with N (invalid character)
        assert result == "ACGNTACNGTA"

    def test_uppercase(self):
        """Test conversion to uppercase."""
        result = clean_sequence("acgt")
        assert result == "ACGT"

    def test_invalid_characters(self):
        """Test handling of invalid characters."""
        result = clean_sequence("ACGT123XYZ", replace_invalid_with="N")
        # Numbers and X, Y, Z should be replaced with N
        assert "1" not in result
        assert "X" not in result


class TestValidateSequence:
    """Tests for validate_sequence function."""

    def test_valid_sequence(self):
        """Test validation of valid sequence."""
        # Use min_length=10 since sequence is only 12 chars (default is 100)
        is_valid, stats = validate_sequence("ACGTACGTACGT", min_length=10)
        assert is_valid is True
        assert stats.is_valid is True

    def test_sequence_too_short(self):
        """Test rejection of too-short sequences."""
        is_valid, stats = validate_sequence("ACG", min_length=10)
        assert is_valid is False
        assert stats.is_valid is False

    def test_too_many_n(self):
        """Test rejection of sequences with too many Ns."""
        # 50% N nucleotides
        is_valid, stats = validate_sequence("ACNNNNNNGT", max_n_ratio=0.1)
        assert is_valid is False

    def test_gc_content_calculation(self):
        """Test GC content calculation."""
        _, stats = validate_sequence("GCGCGCGC")  # 100% GC
        assert stats.gc_content == 1.0
        
        _, stats = validate_sequence("ATATAT")  # 0% GC
        assert stats.gc_content == 0.0
        
        _, stats = validate_sequence("ACGT")  # 50% GC
        assert stats.gc_content == 0.5


class TestGetReverseComplement:
    """Tests for get_reverse_complement function."""

    def test_simple_reverse_complement(self):
        """Test basic reverse complement."""
        assert get_reverse_complement("ACGT") == "ACGT"  # Palindrome
        assert get_reverse_complement("AAA") == "TTT"
        assert get_reverse_complement("GGG") == "CCC"

    def test_reverse_complement_order(self):
        """Test that sequence is reversed."""
        assert get_reverse_complement("AACC") == "GGTT"
        assert get_reverse_complement("ATCG") == "CGAT"

    def test_reverse_complement_with_n(self):
        """Test reverse complement handles N."""
        assert get_reverse_complement("ANC") == "GNT"


class TestFragmentSequence:
    """Tests for fragment_sequence function."""

    def test_fragment_exact_multiple(self):
        """Test fragmentation when sequence is exact multiple of fragment size."""
        sequence = "A" * 100
        fragments = fragment_sequence(sequence, fragment_size=50, step_size=50)
        
        assert len(fragments) == 2
        assert all(len(f) == 50 for f in fragments)

    def test_fragment_with_overlap(self):
        """Test fragmentation with overlapping windows."""
        sequence = "A" * 100
        fragments = fragment_sequence(sequence, fragment_size=50, step_size=25)
        
        # With step_size=25, we get: 0-50, 25-75, 50-100
        assert len(fragments) == 3

    def test_fragment_min_length(self):
        """Test that short remaining fragments are padded by default."""
        sequence = "A" * 80
        # fragment_sequence pads short fragments by default (pad_last=True)
        fragments = fragment_sequence(
            sequence, 
            fragment_size=50, 
            step_size=50,
        )
        
        # First full fragment (50) + padded last fragment (30+20N)
        assert len(fragments) >= 1
        assert all(len(f) == 50 for f in fragments)

    def test_fragment_sequence_too_short(self):
        """Test handling of sequence shorter than fragment size."""
        sequence = "ACGT"
        fragments = list(fragment_sequence(sequence, fragment_size=100))
        
        assert len(fragments) == 0

    def test_fragment_with_reverse_complement(self):
        """Test that fragments can be combined with reverse complement."""
        sequence = "AAACCC"
        fragments = fragment_sequence(
            sequence, 
            fragment_size=6, 
            step_size=6,
        )
        
        # Basic fragmentation returns only original
        assert len(fragments) == 1
        assert "AAACCC" in fragments
        
        # Reverse complement should be done separately
        rc = get_reverse_complement(sequence)
        assert rc == "GGGTTT"


class TestSequencePreprocessor:
    """Tests for SequencePreprocessor class."""

    def test_preprocessor_initialization(self):
        """Test preprocessor initialization with custom params."""
        preprocessor = SequencePreprocessor(
            min_length=500,
            min_valid_ratio=0.9,
            max_n_ratio=0.05,
        )
        
        assert preprocessor.min_length == 500
        assert preprocessor.min_valid_ratio == 0.9
        assert preprocessor.max_n_ratio == 0.05

    def test_process_valid_sequence(self):
        """Test processing a valid sequence."""
        preprocessor = SequencePreprocessor(min_length=10)
        
        cleaned_seq, stats = preprocessor.process("ACGTACGTACGT")
        
        assert cleaned_seq is not None
        assert cleaned_seq == "ACGTACGTACGT"
        assert stats.is_valid is True

    def test_process_invalid_sequence(self):
        """Test processing an invalid sequence."""
        preprocessor = SequencePreprocessor(min_length=100)
        
        cleaned_seq, stats = preprocessor.process("ACGT")  # Too short
        
        # Returns None for invalid sequences
        assert cleaned_seq is None
        assert stats.is_valid is False

    def test_process_batch(self):
        """Test batch processing using process method in a loop."""
        preprocessor = SequencePreprocessor(min_length=5)
        
        sequences = ["ACGTACGT", "TGCATGCA", "AC"]  # Last one invalid
        results = [preprocessor.process(seq) for seq in sequences]
        
        valid_results = [seq for seq, stats in results if stats.is_valid]
        assert len(valid_results) == 2


class TestIntegration:
    """Integration tests for preprocessing pipeline."""

    def test_full_pipeline(self):
        """Test complete preprocessing pipeline."""
        # Simulate real sequence
        sequence = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT" * 25  # 1000 nt
        
        # Clean
        cleaned = clean_sequence(sequence)
        
        # Validate
        is_valid, stats = validate_sequence(cleaned, min_length=500)
        assert is_valid
        
        # Fragment
        fragments = fragment_sequence(cleaned, fragment_size=500, step_size=250)
        assert len(fragments) > 0
        
        # Encode
        encoder = OneHotEncoder()
        for fragment in fragments:
            encoded = encoder.encode(fragment)
            assert encoded.shape == (500, 4)

    def test_pipeline_with_noisy_data(self):
        """Test pipeline handles real-world noisy data."""
        # Sequence with noise
        noisy_sequence = """
        >header_ignored
        ACGT ACGT ACGT NNNN
        acgt acgt RYWM
        ACGT ACGT ACGT ACGT
        """
        
        cleaned = clean_sequence(noisy_sequence, replace_invalid_with="N")
        
        # Should have cleaned the sequence
        assert " " not in cleaned
        assert "\n" not in cleaned
        assert ">" not in cleaned
