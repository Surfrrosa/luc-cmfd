"""Unit tests for RLE encoding/decoding."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest
from rle import rle_encode, rle_decode, validate_submission_row


def test_empty_mask_returns_authentic():
    """Test that an empty mask returns 'authentic'."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    result = rle_encode(mask)
    assert result == "authentic", f"Expected 'authentic', got {result}"


def test_round_trip_simple():
    """Test round-trip encoding and decoding of a simple mask."""
    # Create simple mask with one region
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 3:7] = 1

    # Encode
    rle = rle_encode(mask)

    # Decode
    decoded = rle_decode(rle, mask.shape)

    # Check match
    assert np.array_equal(mask, decoded), "Round-trip failed for simple mask"


def test_round_trip_multiple_regions():
    """Test round-trip with multiple disconnected regions."""
    mask = np.zeros((20, 20), dtype=np.uint8)

    # Region 1
    mask[2:5, 2:5] = 1

    # Region 2
    mask[10:15, 10:15] = 1

    # Region 3
    mask[5:8, 15:18] = 1

    # Encode and decode
    rle = rle_encode(mask)
    decoded = rle_decode(rle, mask.shape)

    assert np.array_equal(mask, decoded), "Round-trip failed for multiple regions"


def test_round_trip_random_masks():
    """Test round-trip on random binary masks."""
    np.random.seed(42)

    for _ in range(10):
        h, w = np.random.randint(50, 200), np.random.randint(50, 200)
        mask = (np.random.rand(h, w) > 0.7).astype(np.uint8)

        rle = rle_encode(mask)

        # Skip if authentic
        if rle == "authentic":
            assert mask.sum() == 0
            continue

        decoded = rle_decode(rle, mask.shape)
        assert np.array_equal(mask, decoded), f"Round-trip failed for random mask {h}x{w}"


def test_single_pixel():
    """Test encoding a single pixel."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[5, 5] = 1

    rle = rle_encode(mask)
    decoded = rle_decode(rle, mask.shape)

    assert np.array_equal(mask, decoded), "Single pixel encoding failed"


def test_full_mask():
    """Test encoding a fully filled mask."""
    mask = np.ones((20, 20), dtype=np.uint8)

    rle = rle_encode(mask)
    decoded = rle_decode(rle, mask.shape)

    assert np.array_equal(mask, decoded), "Full mask encoding failed"


def test_horizontal_stripe():
    """Test encoding horizontal stripe."""
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[10, :] = 1

    rle = rle_encode(mask)
    decoded = rle_decode(rle, mask.shape)

    assert np.array_equal(mask, decoded), "Horizontal stripe encoding failed"


def test_vertical_stripe():
    """Test encoding vertical stripe."""
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[:, 10] = 1

    rle = rle_encode(mask)
    decoded = rle_decode(rle, mask.shape)

    assert np.array_equal(mask, decoded), "Vertical stripe encoding failed"


def test_checkerboard():
    """Test encoding checkerboard pattern."""
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[::2, ::2] = 1
    mask[1::2, 1::2] = 1

    rle = rle_encode(mask)
    decoded = rle_decode(rle, mask.shape)

    assert np.array_equal(mask, decoded), "Checkerboard encoding failed"


def test_decode_authentic():
    """Test decoding 'authentic' string."""
    decoded = rle_decode("authentic", (50, 50))
    expected = np.zeros((50, 50), dtype=np.uint8)

    assert np.array_equal(decoded, expected), "Decoding 'authentic' failed"


def test_rle_format():
    """Test that RLE format is correct (list string with brackets)."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[5, 5] = 1

    rle = rle_encode(mask)

    # Should be a string starting with [ and ending with ]
    assert isinstance(rle, str), "RLE should be a string"
    assert rle.startswith('['), "RLE should start with ["
    assert rle.endswith(']'), "RLE should end with ]"


def test_validate_submission_row_valid():
    """Test validation of valid submission rows."""
    # Authentic row
    row1 = {'case_id': 'image_001', 'annotation': 'authentic'}
    assert validate_submission_row(row1), "Valid authentic row should pass"

    # RLE row
    row2 = {'case_id': 'image_002', 'annotation': '[1, 10, 50, 20]'}
    assert validate_submission_row(row2), "Valid RLE row should pass"


def test_validate_submission_row_invalid():
    """Test validation catches invalid rows."""
    # Missing case_id
    row1 = {'annotation': 'authentic'}
    assert not validate_submission_row(row1), "Missing case_id should fail"

    # Missing annotation
    row2 = {'case_id': 'image_001'}
    assert not validate_submission_row(row2), "Missing annotation should fail"

    # Invalid RLE format
    row3 = {'case_id': 'image_001', 'annotation': '1, 10, 50, 20'}  # Missing brackets
    assert not validate_submission_row(row3), "Invalid RLE format should fail"


def test_consistency_across_shapes():
    """Test that encoding is consistent for different shapes."""
    # Create same logical pattern in different sized arrays
    for size in [10, 20, 50, 100]:
        mask = np.zeros((size, size), dtype=np.uint8)
        # Always put a square in the middle
        center = size // 2
        mask[center-2:center+2, center-2:center+2] = 1

        rle = rle_encode(mask)
        decoded = rle_decode(rle, mask.shape)

        assert np.array_equal(mask, decoded), f"Consistency failed for size {size}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
