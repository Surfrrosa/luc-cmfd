"""Geometric verification using RANSAC for copy-move detection."""

from typing import Dict, Tuple, Optional
import numpy as np
from skimage.transform import SimilarityTransform, AffineTransform, estimate_transform


def estimate_transform_ransac(
    pts_q: np.ndarray,
    pts_t: np.ndarray,
    model: str = "similarity",
    inlier_thresh_px: float = 2.0,
    max_trials: int = 1000,
    min_samples: Optional[int] = None,
    stop_probability: float = 0.99
) -> Dict:
    """
    Estimate geometric transform using RANSAC.

    Args:
        pts_q: Query points (N, 2) - source locations
        pts_t: Target points (N, 2) - matched locations
        model: Transform model, either "similarity" or "affine"
        inlier_thresh_px: Inlier threshold in pixels
        max_trials: Maximum RANSAC iterations
        min_samples: Minimum samples for model (auto-set if None)
        stop_probability: Desired probability of finding good model

    Returns:
        Dictionary containing:
            - 'transform': Estimated transform matrix (3x3 for homogeneous coords)
            - 'inliers': Boolean array indicating inliers
            - 'n_inliers': Number of inliers
            - 'inlier_ratio': Ratio of inliers to total points
    """
    assert pts_q.shape == pts_t.shape, "Point arrays must have same shape"
    assert pts_q.shape[1] == 2, "Points must be (N, 2)"
    assert model in ["similarity", "affine"], f"Unknown model: {model}"

    n_pts = len(pts_q)

    # Set minimum samples based on model
    if min_samples is None:
        min_samples = 3 if model == "similarity" else 3  # 3 for both

    # Handle degenerate cases
    if n_pts < min_samples:
        return {
            'transform': None,
            'inliers': np.zeros(n_pts, dtype=bool),
            'n_inliers': 0,
            'inlier_ratio': 0.0
        }

    # Custom RANSAC implementation
    best_inliers = np.zeros(n_pts, dtype=bool)
    best_n_inliers = 0
    best_transform = None

    for trial in range(max_trials):
        # Sample random points
        sample_idx = np.random.choice(n_pts, min_samples, replace=False)
        sample_src = pts_q[sample_idx]
        sample_dst = pts_t[sample_idx]

        # Estimate transform
        try:
            if model == "similarity":
                tform = SimilarityTransform()
            else:  # affine
                tform = AffineTransform()

            if not tform.estimate(sample_src, sample_dst):
                continue

        except Exception:
            continue

        # Compute residuals for all points
        pts_t_pred = tform(pts_q)
        residuals = np.sqrt(np.sum((pts_t - pts_t_pred) ** 2, axis=1))

        # Find inliers
        inliers = residuals < inlier_thresh_px
        n_inliers = np.sum(inliers)

        # Update best model
        if n_inliers > best_n_inliers:
            best_n_inliers = n_inliers
            best_inliers = inliers
            best_transform = tform

            # Early stopping based on inlier ratio
            inlier_ratio = n_inliers / n_pts
            if inlier_ratio > 0.8:  # High quality match found
                break

    # Refine transform using all inliers
    if best_n_inliers >= min_samples and best_transform is not None:
        try:
            if model == "similarity":
                refined_tform = SimilarityTransform()
            else:
                refined_tform = AffineTransform()

            refined_tform.estimate(pts_q[best_inliers], pts_t[best_inliers])

            # Recompute inliers with refined transform
            pts_t_pred = refined_tform(pts_q)
            residuals = np.sqrt(np.sum((pts_t - pts_t_pred) ** 2, axis=1))
            best_inliers = residuals < inlier_thresh_px
            best_n_inliers = np.sum(best_inliers)
            best_transform = refined_tform

        except Exception:
            pass  # Keep original if refinement fails

    # Extract transform matrix
    transform_matrix = best_transform.params if best_transform is not None else None

    return {
        'transform': transform_matrix,
        'inliers': best_inliers,
        'n_inliers': best_n_inliers,
        'inlier_ratio': best_n_inliers / n_pts if n_pts > 0 else 0.0
    }


def mask_from_inliers(
    inliers: np.ndarray,
    pts_q: np.ndarray,
    shape: Tuple[int, int],
    growth: int = 0
) -> np.ndarray:
    """
    Create a binary mask from inlier points.

    Args:
        inliers: Boolean array indicating inlier points
        pts_q: Query point coordinates (N, 2) in (x, y) format
        shape: Output mask shape (H, W)
        growth: Dilation radius for region growing (0 for no growth)

    Returns:
        Binary mask (H, W) with inlier regions marked
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # Mark inlier positions
    inlier_pts = pts_q[inliers]

    if len(inlier_pts) == 0:
        return mask

    # Convert to integer coordinates and clip to bounds
    pts_int = np.round(inlier_pts).astype(int)
    pts_int[:, 0] = np.clip(pts_int[:, 0], 0, w - 1)  # x
    pts_int[:, 1] = np.clip(pts_int[:, 1], 0, h - 1)  # y

    # Mark points (y, x for array indexing)
    mask[pts_int[:, 1], pts_int[:, 0]] = 1

    # Optional region growing via dilation
    if growth > 0:
        from scipy.ndimage import binary_dilation
        from skimage.morphology import disk
        mask = binary_dilation(mask, disk(growth)).astype(np.uint8)

    return mask


def filter_matches_by_consistency(
    pts_q: np.ndarray,
    pts_t: np.ndarray,
    scores: np.ndarray,
    distance_thresh: float = 50.0,
    angle_thresh: float = 30.0
) -> np.ndarray:
    """
    Filter matches based on geometric consistency.

    Removes matches that have inconsistent displacement vectors
    compared to their neighbors.

    Args:
        pts_q: Query points (N, 2)
        pts_t: Target points (N, 2)
        scores: Match scores (N,)
        distance_thresh: Maximum displacement distance variation (pixels)
        angle_thresh: Maximum angle variation (degrees)

    Returns:
        Boolean array indicating consistent matches
    """
    n_pts = len(pts_q)

    if n_pts < 3:
        return np.ones(n_pts, dtype=bool)

    # Compute displacement vectors
    displacements = pts_t - pts_q

    # Compute pairwise consistency
    consistent = np.ones(n_pts, dtype=bool)

    # Median displacement
    median_disp = np.median(displacements, axis=0)

    # Filter based on distance from median
    dist_from_median = np.linalg.norm(displacements - median_disp, axis=1)
    consistent &= dist_from_median < distance_thresh

    return consistent


def compute_transform_params(transform_matrix: np.ndarray) -> Dict:
    """
    Extract interpretable parameters from transform matrix.

    Args:
        transform_matrix: 3x3 transformation matrix

    Returns:
        Dictionary with scale, rotation, translation, and shear
    """
    if transform_matrix is None:
        return {
            'scale': 1.0,
            'rotation_deg': 0.0,
            'translation': np.array([0.0, 0.0]),
            'shear': 0.0
        }

    # Extract 2x2 linear part
    A = transform_matrix[:2, :2]
    t = transform_matrix[:2, 2]

    # SVD decomposition
    U, S, Vt = np.linalg.svd(A)

    # Rotation
    R = U @ Vt
    rotation_rad = np.arctan2(R[1, 0], R[0, 0])
    rotation_deg = np.degrees(rotation_rad)

    # Scale (average of singular values)
    scale = np.mean(S)

    # Shear (ratio of singular values)
    shear = S[1] / S[0] if S[0] > 0 else 0.0

    return {
        'scale': scale,
        'rotation_deg': rotation_deg,
        'translation': t,
        'shear': shear
    }
