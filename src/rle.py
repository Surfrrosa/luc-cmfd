"""Run-length encoding (RLE) utilities for mask serialization.

The RLE format uses 1-indexed flat indexing (row-major order).
Empty masks are encoded as the literal string "authentic".
"""

from typing import Union
import numpy as np


def rle_encode(mask: np.ndarray) -> str:
    """
    Encode a binary mask as run-length encoding.

    Args:
        mask: 2D binary mask of shape (H, W) with values {0, 1}

    Returns:
        If mask is empty (all zeros): returns literal string "authentic"
        Otherwise: returns RLE string "[start length start length ...]"
                  with 1-indexed flat indexing in row-major order
    """
    # Ensure mask is binary
    assert mask.ndim == 2, f"Mask must be 2D, got shape {mask.shape}"
    assert set(np.unique(mask)).issubset({0, 1}), "Mask must be binary {0, 1}"

    # Check if mask is empty
    if not mask.any():
        return "authentic"

    # Flatten in row-major (C) order and convert to 1-indexed
    flat = mask.flatten(order='C')

    # Find runs of 1s
    # Pad with 0 to detect edges
    padded = np.concatenate([[0], flat, [0]])
    diff = np.diff(padded)

    # Start positions (0->1 transitions) - these are 0-indexed
    starts = np.where(diff == 1)[0]
    # End positions (1->0 transitions) - these are 0-indexed
    ends = np.where(diff == -1)[0]

    # Convert to 1-indexed starts and compute lengths
    starts_1idx = starts + 1  # Convert to 1-indexed
    lengths = ends - starts

    # Interleave starts and lengths
    rle_pairs = np.stack([starts_1idx, lengths], axis=1).flatten()

    # Format as string "[start length start length ...]"
    rle_str = str(rle_pairs.tolist())

    return rle_str


def rle_decode(rle: str, shape: tuple[int, int]) -> np.ndarray:
    """
    Decode run-length encoding back to a binary mask.

    Args:
        rle: RLE string (either "authentic" or "[start length ...]")
        shape: Output mask shape (H, W)

    Returns:
        Binary mask of shape (H, W) with values {0, 1}
    """
    h, w = shape
    mask = np.zeros(h * w, dtype=np.uint8)

    # Handle authentic case
    if rle == "authentic":
        return mask.reshape(shape, order='C')

    # Parse RLE string
    # Remove brackets and parse as list
    rle = rle.strip()
    if rle.startswith('[') and rle.endswith(']'):
        rle = rle[1:-1]

    if not rle:  # Empty RLE
        return mask.reshape(shape, order='C')

    # Parse pairs of (start, length)
    values = [int(x.strip()) for x in rle.split(',')]

    # Process pairs
    for i in range(0, len(values), 2):
        start_1idx = values[i]
        length = values[i + 1]

        # Convert from 1-indexed to 0-indexed
        start_0idx = start_1idx - 1

        # Set the run to 1
        mask[start_0idx:start_0idx + length] = 1

    return mask.reshape(shape, order='C')


def mask_to_submission_format(case_id: str, mask: np.ndarray) -> dict:
    """
    Convert a case_id and mask to submission format.

    Args:
        case_id: Image identifier
        mask: Binary mask (H, W)

    Returns:
        Dictionary with 'case_id' and 'annotation' keys
    """
    return {
        'case_id': case_id,
        'annotation': rle_encode(mask)
    }


def validate_submission_row(row: dict) -> bool:
    """
    Validate a single submission row.

    Args:
        row: Dictionary with 'case_id' and 'annotation' keys

    Returns:
        True if valid, False otherwise
    """
    if 'case_id' not in row or 'annotation' not in row:
        return False

    if not isinstance(row['case_id'], str):
        return False

    annotation = row['annotation']
    if annotation == "authentic":
        return True

    # Check if it's a valid RLE string
    if not isinstance(annotation, str):
        return False

    annotation = annotation.strip()
    if not (annotation.startswith('[') and annotation.endswith(']')):
        return False

    return True
