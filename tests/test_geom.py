"""Unit tests for geometric verification and RANSAC."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest
from geom import (
    estimate_transform_ransac,
    mask_from_inliers,
    compute_transform_params
)


def create_synthetic_correspondence(
    n_points: int = 50,
    rotation_deg: float = 15,
    scale: float = 1.1,
    translation: tuple = (10, 20),
    noise_std: float = 0.5,
    outlier_ratio: float = 0.1
):
    """
    Create synthetic point correspondences with known transform.

    Args:
        n_points: Number of points
        rotation_deg: Rotation angle in degrees
        scale: Scale factor
        translation: Translation (tx, ty)
        noise_std: Gaussian noise std
        outlier_ratio: Ratio of outlier points

    Returns:
        Tuple of (source_pts, target_pts, true_inliers)
    """
    # Generate random source points
    src_pts = np.random.rand(n_points, 2) * 100

    # Apply similarity transform
    theta = np.radians(rotation_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # Transformation matrix
    A = scale * np.array([
        [cos_t, -sin_t],
        [sin_t, cos_t]
    ])

    t = np.array(translation)

    # Transform points
    tgt_pts = (A @ src_pts.T).T + t

    # Add noise to inliers
    tgt_pts += np.random.randn(n_points, 2) * noise_std

    # Add outliers
    n_outliers = int(n_points * outlier_ratio)
    outlier_indices = np.random.choice(n_points, n_outliers, replace=False)

    true_inliers = np.ones(n_points, dtype=bool)
    true_inliers[outlier_indices] = False

    # Replace outliers with random points
    tgt_pts[outlier_indices] = np.random.rand(n_outliers, 2) * 100

    return src_pts, tgt_pts, true_inliers


def test_ransac_similarity_pure_inliers():
    """Test RANSAC with pure inliers (no outliers)."""
    np.random.seed(42)

    src_pts, tgt_pts, _ = create_synthetic_correspondence(
        n_points=50,
        rotation_deg=10,
        scale=1.0,
        translation=(5, 10),
        noise_std=0.3,
        outlier_ratio=0.0  # No outliers
    )

    result = estimate_transform_ransac(
        src_pts, tgt_pts,
        model='similarity',
        inlier_thresh_px=2.0
    )

    # Should find most points as inliers
    assert result['inlier_ratio'] >= 0.9, f"Inlier ratio too low: {result['inlier_ratio']}"
    assert result['n_inliers'] >= 45, f"Too few inliers: {result['n_inliers']}"


def test_ransac_similarity_with_outliers():
    """Test RANSAC with outliers."""
    np.random.seed(42)

    src_pts, tgt_pts, true_inliers = create_synthetic_correspondence(
        n_points=100,
        rotation_deg=15,
        scale=1.05,
        translation=(10, 15),
        noise_std=0.5,
        outlier_ratio=0.2
    )

    result = estimate_transform_ransac(
        src_pts, tgt_pts,
        model='similarity',
        inlier_thresh_px=2.0
    )

    # Should identify at least 70% of points as inliers (most true inliers)
    assert result['inlier_ratio'] >= 0.7, f"Inlier ratio too low: {result['inlier_ratio']}"

    # Check overlap with true inliers
    overlap = np.sum(result['inliers'] & true_inliers)
    expected_inliers = np.sum(true_inliers)
    overlap_ratio = overlap / expected_inliers

    assert overlap_ratio >= 0.85, f"Poor inlier detection: {overlap_ratio:.2f}"


def test_ransac_affine():
    """Test RANSAC with affine transform."""
    np.random.seed(42)

    # Create affine transform (with shear)
    src_pts = np.random.rand(50, 2) * 100

    # Affine matrix
    A = np.array([
        [1.1, 0.2],
        [0.1, 1.0]
    ])
    t = np.array([5, 10])

    tgt_pts = (A @ src_pts.T).T + t
    tgt_pts += np.random.randn(50, 2) * 0.5

    result = estimate_transform_ransac(
        src_pts, tgt_pts,
        model='affine',
        inlier_thresh_px=2.0
    )

    assert result['inlier_ratio'] >= 0.9, f"Affine inlier ratio too low: {result['inlier_ratio']}"


def test_ransac_rotation_range():
    """Test RANSAC with rotations in [-20, 20] degree range."""
    np.random.seed(42)

    for angle in [-20, -10, 0, 10, 20]:
        src_pts, tgt_pts, _ = create_synthetic_correspondence(
            n_points=50,
            rotation_deg=angle,
            scale=1.0,
            translation=(0, 0),
            noise_std=0.5,
            outlier_ratio=0.1
        )

        result = estimate_transform_ransac(
            src_pts, tgt_pts,
            model='similarity',
            inlier_thresh_px=2.0
        )

        assert result['inlier_ratio'] >= 0.8, \
            f"Failed for rotation {angle}°: inlier_ratio={result['inlier_ratio']}"


def test_ransac_scale_range():
    """Test RANSAC with scales in [0.8, 1.2] range."""
    np.random.seed(42)

    for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
        src_pts, tgt_pts, _ = create_synthetic_correspondence(
            n_points=50,
            rotation_deg=0,
            scale=scale,
            translation=(0, 0),
            noise_std=0.5,
            outlier_ratio=0.1
        )

        result = estimate_transform_ransac(
            src_pts, tgt_pts,
            model='similarity',
            inlier_thresh_px=2.0
        )

        assert result['inlier_ratio'] >= 0.8, \
            f"Failed for scale {scale}: inlier_ratio={result['inlier_ratio']}"


def test_ransac_degenerate_cases():
    """Test RANSAC handles degenerate cases gracefully."""
    # Too few points
    src_pts = np.random.rand(2, 2)
    tgt_pts = np.random.rand(2, 2)

    result = estimate_transform_ransac(src_pts, tgt_pts, model='similarity')

    assert result['n_inliers'] == 0, "Should return 0 inliers for too few points"
    assert result['transform'] is None, "Should return None transform for degenerate case"


def test_mask_from_inliers():
    """Test creating mask from inlier points."""
    # Create some inlier points
    pts_q = np.array([
        [10, 10],
        [20, 20],
        [30, 30],
        [40, 40]
    ], dtype=float)

    inliers = np.array([True, True, False, True])

    mask = mask_from_inliers(
        inliers, pts_q,
        shape=(50, 50),
        growth=0
    )

    # Check that inlier positions are marked
    assert mask[10, 10] == 1, "Inlier point should be marked"
    assert mask[20, 20] == 1, "Inlier point should be marked"
    assert mask[30, 30] == 0, "Non-inlier point should not be marked"
    assert mask[40, 40] == 1, "Inlier point should be marked"


def test_mask_from_inliers_with_growth():
    """Test mask creation with region growing."""
    pts_q = np.array([[25, 25]], dtype=float)
    inliers = np.array([True])

    mask = mask_from_inliers(
        inliers, pts_q,
        shape=(50, 50),
        growth=5
    )

    # Check that region has grown around the point
    assert mask.sum() > 1, "Mask should have grown around inlier point"

    # Center should definitely be marked
    assert mask[25, 25] == 1, "Center inlier point should be marked"


def test_compute_transform_params():
    """Test extraction of transform parameters."""
    # Create a known similarity transform
    angle = 15  # degrees
    scale = 1.1
    tx, ty = 10, 20

    theta = np.radians(angle)
    transform_matrix = np.array([
        [scale * np.cos(theta), -scale * np.sin(theta), tx],
        [scale * np.sin(theta), scale * np.cos(theta), ty],
        [0, 0, 1]
    ])

    params = compute_transform_params(transform_matrix)

    # Check parameters are approximately correct
    assert abs(params['rotation_deg'] - angle) < 1, \
        f"Rotation mismatch: expected {angle}, got {params['rotation_deg']}"

    assert abs(params['scale'] - scale) < 0.1, \
        f"Scale mismatch: expected {scale}, got {params['scale']}"

    assert np.allclose(params['translation'], [tx, ty], atol=0.1), \
        f"Translation mismatch: expected [{tx}, {ty}], got {params['translation']}"


def test_ransac_high_inlier_rate():
    """Test that RANSAC achieves >90% inlier detection on clean synthetic data."""
    np.random.seed(42)

    src_pts, tgt_pts, true_inliers = create_synthetic_correspondence(
        n_points=100,
        rotation_deg=10,
        scale=0.95,
        translation=(15, 25),
        noise_std=0.3,
        outlier_ratio=0.05  # Very few outliers
    )

    result = estimate_transform_ransac(
        src_pts, tgt_pts,
        model='similarity',
        inlier_thresh_px=1.5
    )

    # Should achieve high inlier rate
    assert result['inlier_ratio'] >= 0.9, \
        f"Expected ≥90% inliers, got {result['inlier_ratio']:.2%}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
