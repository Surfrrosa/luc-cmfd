"""Keypoint-based proposals for copy-move detection.

Uses SIFT/ORB to find rotation/scale-invariant correspondences within an image.
Provides high-quality seeds for RANSAC geometric verification.
"""

from typing import Dict, Optional
import numpy as np
import cv2


def kp_proposals(
    img_gray: np.ndarray,
    method: str = 'ORB',
    max_kp: int = 1500,
    top_matches: int = 300,
    ratio_test: float = 0.75,
    min_distance: float = 2.0
) -> Optional[Dict[str, np.ndarray]]:
    """
    Detect and match keypoints within a single image for copy-move detection.

    Args:
        img_gray: Grayscale image (H, W) uint8
        method: 'SIFT' or 'ORB'
        max_kp: Maximum number of keypoints to detect
        top_matches: Keep top N matches
        ratio_test: Lowe's ratio test threshold
        min_distance: Minimum spatial distance to avoid trivial matches

    Returns:
        Dictionary with 'qxy' and 'txy' arrays (N, 2) or None if insufficient matches
    """
    if img_gray.ndim != 2:
        raise ValueError(f"Expected grayscale image, got shape {img_gray.shape}")

    # Create detector
    if method.upper() == 'SIFT':
        if hasattr(cv2, 'SIFT_create'):
            detector = cv2.SIFT_create(nfeatures=max_kp)
        elif hasattr(cv2, 'xfeatures2d'):
            detector = cv2.xfeatures2d.SIFT_create(nfeatures=max_kp)
        else:
            # Fallback to ORB if SIFT not available
            print("Warning: SIFT not available, falling back to ORB")
            detector = cv2.ORB_create(nfeatures=max_kp)
            method = 'ORB'
    elif method.upper() == 'ORB':
        detector = cv2.ORB_create(nfeatures=max_kp)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'SIFT' or 'ORB'")

    # Detect and describe
    keypoints, descriptors = detector.detectAndCompute(img_gray, None)

    if descriptors is None or len(keypoints) < 8:
        return None

    # Match to itself
    if method.upper() == 'SIFT':
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # K-NN matching (k=2 for ratio test)
    matches = matcher.knnMatch(descriptors, descriptors, k=2)

    # Filter matches
    good_matches = []

    for match_pair in matches:
        if len(match_pair) < 2:
            continue

        m, n = match_pair

        # Skip identical matches
        if m.trainIdx == m.queryIdx:
            continue

        # Lowe's ratio test
        if m.distance < ratio_test * n.distance:
            q_pt = keypoints[m.queryIdx].pt
            t_pt = keypoints[m.trainIdx].pt

            # Spatial distance check (avoid trivial matches)
            spatial_dist = np.sqrt((q_pt[0] - t_pt[0])**2 + (q_pt[1] - t_pt[1])**2)

            if spatial_dist > min_distance:
                good_matches.append((q_pt, t_pt, m.distance))

    if len(good_matches) < 8:
        return None

    # Sort by distance and keep top matches
    good_matches = sorted(good_matches, key=lambda x: x[2])[:top_matches]

    # Extract coordinates
    qxy = np.array([m[0] for m in good_matches], dtype=np.float32)
    txy = np.array([m[1] for m in good_matches], dtype=np.float32)

    return {
        'qxy': qxy,
        'txy': txy,
        'n_matches': len(good_matches)
    }


def kp_proposals_rgb(
    img_rgb: np.ndarray,
    **kwargs
) -> Optional[Dict[str, np.ndarray]]:
    """
    Wrapper for RGB images.

    Args:
        img_rgb: RGB image (H, W, 3) uint8
        **kwargs: Arguments for kp_proposals

    Returns:
        Keypoint proposals or None
    """
    if img_rgb.ndim == 3:
        # Convert to grayscale
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    elif img_rgb.ndim == 2:
        img_gray = img_rgb
    else:
        raise ValueError(f"Expected 2D or 3D image, got shape {img_rgb.shape}")

    return kp_proposals(img_gray, **kwargs)


def visualize_matches(
    img: np.ndarray,
    qxy: np.ndarray,
    txy: np.ndarray,
    n_show: int = 50
) -> np.ndarray:
    """
    Visualize keypoint matches on image.

    Args:
        img: RGB image (H, W, 3)
        qxy: Query points (N, 2)
        txy: Target points (N, 2)
        n_show: Number of matches to show

    Returns:
        Visualization image
    """
    vis = img.copy()

    n = min(n_show, len(qxy))

    for i in range(n):
        q = tuple(qxy[i].astype(int))
        t = tuple(txy[i].astype(int))

        # Draw line
        cv2.line(vis, q, t, (0, 255, 0), 1)

        # Draw points
        cv2.circle(vis, q, 3, (255, 0, 0), -1)
        cv2.circle(vis, t, 3, (0, 0, 255), -1)

    return vis
