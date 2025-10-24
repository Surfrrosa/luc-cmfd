"""Unit tests for self-correlation computation."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import pytest
from corr import (
    build_pyramid_feats,
    self_corr,
    extract_match_pairs,
    create_exclusion_mask
)


def test_build_pyramid_basic():
    """Test basic pyramid building."""
    # Create random features
    feats = torch.randn(1, 128, 32, 32)

    pyramid = build_pyramid_feats(feats, levels=[1, 2, 4])

    assert len(pyramid) == 3, "Should have 3 pyramid levels"
    assert pyramid[0].shape == (1, 128, 32, 32), "Level 1 should match input"
    assert pyramid[1].shape == (1, 128, 16, 16), "Level 2 should be downsampled 2x"
    assert pyramid[2].shape == (1, 128, 8, 8), "Level 4 should be downsampled 4x"


def test_pyramid_normalization():
    """Test that pyramid features are L2 normalized."""
    feats = torch.randn(1, 64, 16, 16)

    pyramid = build_pyramid_feats(feats, levels=[1])

    # Check L2 norm along channel dimension
    norms = torch.norm(pyramid[0], p=2, dim=1)

    # Should be close to 1 for all spatial locations
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
        "Features should be L2 normalized"


def test_self_corr_basic():
    """Test basic self-correlation computation."""
    # Create features with a duplicated pattern
    feats = torch.randn(1, 64, 32, 32)

    result = self_corr(
        feats,
        patch=8,
        stride=4,
        top_k=5
    )

    assert 'corr_map' in result, "Should return correlation map"
    assert 'offset_map' in result, "Should return offset map"
    assert 'query_coords' in result, "Should return query coordinates"

    corr_map = result['corr_map']
    offset_map = result['offset_map']

    # Check shapes
    assert corr_map.dim() == 4, "Corr map should be 4D"
    assert corr_map.shape[1] == 5, "Should have top_k=5 channels"

    assert offset_map.dim() == 5, "Offset map should be 5D"
    assert offset_map.shape[1] == 5, "Should have top_k=5 channels"
    assert offset_map.shape[2] == 2, "Should have 2 offset dimensions (dy, dx)"


def test_self_corr_with_duplicated_patch():
    """Test that self-correlation finds duplicated patches."""
    np.random.seed(42)
    torch.manual_seed(42)

    # Create image with a clear duplicated region
    # Use simpler features for more controlled test
    feats = torch.zeros(1, 32, 40, 40)

    # Create a distinctive pattern
    pattern = torch.randn(1, 32, 8, 8)

    # Place pattern at two locations
    feats[:, :, 5:13, 5:13] = pattern
    feats[:, :, 25:33, 25:33] = pattern  # Duplicate

    # Normalize
    feats = torch.nn.functional.normalize(feats, p=2, dim=1)

    result = self_corr(
        feats,
        patch=8,
        stride=4,
        top_k=3,
        exclude_self_radius=1  # Small exclusion for this test
    )

    corr_map = result['corr_map'][0]  # (k, H, W)

    # The duplicated regions should have high correlation
    # Check that max correlation is high
    max_corr = corr_map.max().item()

    assert max_corr > 0.5, f"Max correlation should be high for duplicated patches, got {max_corr}"


def test_exclusion_mask():
    """Test that exclusion mask correctly excludes self-matches."""
    mask = create_exclusion_mask(
        h=10, w=10,
        exclude_radius=1,
        device=torch.device('cpu')
    )

    # Check shape
    assert mask.shape == (100, 100), "Mask should be (N, N) where N=h*w"

    # Diagonal should be True (self-matches excluded)
    assert torch.all(torch.diag(mask)), "Diagonal should be excluded"

    # Check a specific point
    # Point at (5, 5) -> index 55
    # Its neighbors within radius 1 should be excluded
    idx = 5 * 10 + 5
    neighbors = [
        4 * 10 + 5,  # (4, 5)
        6 * 10 + 5,  # (6, 5)
        5 * 10 + 4,  # (5, 4)
        5 * 10 + 6,  # (5, 6)
    ]

    for neighbor_idx in neighbors:
        assert mask[idx, neighbor_idx], f"Neighbor {neighbor_idx} should be excluded"


def test_extract_match_pairs():
    """Test extraction of point correspondences."""
    # Create synthetic correlation result
    corr_map = torch.rand(1, 5, 16, 16)  # (B, k, H, W)
    offset_map = torch.randn(1, 5, 2, 16, 16)  # (B, k, 2, H, W)

    query_coords = np.zeros((16, 16, 2))
    for i in range(16):
        for j in range(16):
            query_coords[i, j] = [i, j]

    corr_result = {
        'corr_map': corr_map,
        'offset_map': offset_map,
        'query_coords': query_coords
    }

    query_pts, target_pts, scores = extract_match_pairs(
        corr_result,
        score_threshold=0.5,
        max_pairs=100
    )

    # Check shapes
    assert query_pts.ndim == 2 and query_pts.shape[1] == 2, "Query points should be (N, 2)"
    assert target_pts.ndim == 2 and target_pts.shape[1] == 2, "Target points should be (N, 2)"
    assert scores.ndim == 1, "Scores should be 1D"
    assert len(query_pts) == len(target_pts) == len(scores), "All should have same length"

    # Should respect max_pairs
    assert len(query_pts) <= 100, "Should not exceed max_pairs"

    # All scores should be >= threshold
    assert np.all(scores >= 0.5), "All scores should be above threshold"


def test_self_corr_deterministic():
    """Test that self-correlation is deterministic."""
    torch.manual_seed(42)

    feats = torch.randn(1, 64, 24, 24)

    result1 = self_corr(feats, patch=8, stride=4, top_k=5)
    result2 = self_corr(feats, patch=8, stride=4, top_k=5)

    # Results should be identical
    assert torch.allclose(result1['corr_map'], result2['corr_map']), \
        "Self-correlation should be deterministic"


def test_self_corr_different_params():
    """Test self-correlation with different parameter settings."""
    feats = torch.randn(1, 64, 32, 32)

    # Test different patch sizes
    for patch in [8, 12, 16]:
        result = self_corr(feats, patch=patch, stride=4, top_k=5)
        assert result['corr_map'].shape[1] == 5, f"Should have top_k=5 for patch={patch}"

    # Test different strides
    for stride in [4, 8]:
        result = self_corr(feats, patch=12, stride=stride, top_k=5)
        h_out = (32 - 12) // stride + 1
        assert result['corr_map'].shape[2] == h_out, f"Incorrect output height for stride={stride}"

    # Test different top_k
    for top_k in [3, 5, 7]:
        result = self_corr(feats, patch=12, stride=4, top_k=top_k)
        assert result['corr_map'].shape[1] == top_k, f"Should have top_k={top_k}"


def test_corr_map_values_in_range():
    """Test that correlation map values are in valid range."""
    feats = torch.randn(1, 64, 32, 32)
    feats = torch.nn.functional.normalize(feats, p=2, dim=1)

    result = self_corr(feats, patch=8, stride=4, top_k=5)
    corr_map = result['corr_map']

    # Correlation of normalized vectors should be in [-1, 1]
    assert corr_map.max() <= 1.0 + 1e-5, "Max correlation should be <= 1"
    assert corr_map.min() >= -1.0 - 1e-5, "Min correlation should be >= -1"


def test_query_coords_shape():
    """Test query coordinates have correct shape."""
    feats = torch.randn(1, 64, 32, 32)

    result = self_corr(feats, patch=8, stride=4, top_k=5)
    query_coords = result['query_coords']

    h_out = (32 - 8) // 4 + 1
    w_out = (32 - 8) // 4 + 1

    assert query_coords.shape == (h_out, w_out, 2), \
        f"Query coords should have shape ({h_out}, {w_out}, 2)"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
