"""Post-processing utilities for mask refinement."""

from typing import Dict, Optional, Tuple
import numpy as np
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, binary_opening, disk


def clean_mask(
    prob: np.ndarray,
    thr: float = 0.5,
    min_area: int = 24,
    morph: Optional[Dict[str, int]] = None
) -> np.ndarray:
    """
    Clean and refine a probability mask.

    Args:
        prob: Probability map (H, W) with values in [0, 1]
        thr: Threshold for binarization
        min_area: Minimum connected component area (pixels). Components smaller
                 than this will be removed.
        morph: Dictionary with morphological operation parameters:
               - 'close': kernel size for binary closing (0 to skip)
               - 'open': kernel size for binary opening (0 to skip)

    Returns:
        Binary mask (H, W) with values {0, 1}
    """
    # Binarize
    binary = (prob >= thr).astype(np.uint8)

    # Apply morphological operations if specified
    if morph is not None:
        close_size = morph.get('close', 0)
        open_size = morph.get('open', 0)

        if close_size > 0:
            binary = binary_closing(binary, disk(close_size)).astype(np.uint8)

        if open_size > 0:
            binary = binary_opening(binary, disk(open_size)).astype(np.uint8)

    # Connected components analysis
    labeled = label(binary, connectivity=2)

    # Remove small components
    if min_area > 0:
        binary = remove_small_components(labeled, min_area)
    else:
        binary = (labeled > 0).astype(np.uint8)

    return binary


def remove_small_components(labeled: np.ndarray, min_area: int) -> np.ndarray:
    """
    Remove connected components smaller than min_area.

    Args:
        labeled: Labeled image from skimage.measure.label
        min_area: Minimum area threshold

    Returns:
        Binary mask with small components removed
    """
    # Count pixels per label
    unique, counts = np.unique(labeled, return_counts=True)

    # Create mask of labels to keep (excluding background 0)
    keep_labels = unique[(counts >= min_area) & (unique != 0)]

    # Build output mask
    mask = np.isin(labeled, keep_labels).astype(np.uint8)

    return mask


def fill_holes(mask: np.ndarray, max_hole_area: int = 100) -> np.ndarray:
    """
    Fill small holes in binary mask.

    Args:
        mask: Binary mask (H, W)
        max_hole_area: Maximum hole size to fill (pixels)

    Returns:
        Binary mask with holes filled
    """
    # Find holes (connected components of background)
    inverted = 1 - mask
    labeled_holes = label(inverted, connectivity=2)

    # Fill holes smaller than threshold
    filled = mask.copy()
    unique, counts = np.unique(labeled_holes, return_counts=True)

    for hole_label, hole_size in zip(unique, counts):
        if hole_label == 0:  # Skip background
            continue
        if hole_size <= max_hole_area:
            filled[labeled_holes == hole_label] = 1

    return filled


def merge_nearby_components(
    mask: np.ndarray,
    distance_threshold: int = 10
) -> np.ndarray:
    """
    Merge connected components that are close to each other.

    Args:
        mask: Binary mask (H, W)
        distance_threshold: Maximum distance for merging (pixels)

    Returns:
        Binary mask with nearby components merged
    """
    # Dilate to merge nearby components
    if distance_threshold > 0:
        structure = disk(distance_threshold)
        merged = binary_closing(mask, structure).astype(np.uint8)
        return merged

    return mask


def smooth_boundary(mask: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Smooth mask boundaries using Gaussian blur.

    Args:
        mask: Binary mask (H, W)
        sigma: Gaussian blur sigma

    Returns:
        Smoothed binary mask
    """
    # Apply Gaussian blur and re-threshold
    blurred = ndimage.gaussian_filter(mask.astype(float), sigma=sigma)
    smoothed = (blurred >= 0.5).astype(np.uint8)

    return smoothed


def largest_component(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected component.

    Args:
        mask: Binary mask (H, W)

    Returns:
        Binary mask with only largest component
    """
    labeled = label(mask, connectivity=2)

    if labeled.max() == 0:  # No components
        return mask

    # Find largest component
    unique, counts = np.unique(labeled, return_counts=True)
    # Exclude background (label 0)
    foreground_labels = unique[unique != 0]
    foreground_counts = counts[unique != 0]

    if len(foreground_labels) == 0:
        return np.zeros_like(mask)

    largest_label = foreground_labels[np.argmax(foreground_counts)]

    # Keep only largest
    result = (labeled == largest_label).astype(np.uint8)

    return result


def detect_periodicity(image: np.ndarray, threshold: float = 0.3) -> Tuple[bool, float]:
    """
    Detect periodic textures using FFT radial profile.

    Args:
        image: Grayscale image (H, W)
        threshold: Periodicity strength threshold

    Returns:
        (is_periodic, strength) tuple
    """
    # Apply FFT
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    # Compute radial profile
    h, w = magnitude.shape
    center = (h // 2, w // 2)
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)

    # Average magnitude per radius
    max_r = min(center)
    radial_profile = np.zeros(max_r)
    for radius in range(max_r):
        mask = (r == radius)
        if mask.sum() > 0:
            radial_profile[radius] = magnitude[mask].mean()

    # Detect peaks (excluding DC and very low frequencies)
    if len(radial_profile) < 10:
        return False, 0.0

    profile = radial_profile[5:]  # Skip low frequencies
    if profile.max() == 0:
        return False, 0.0

    # Normalize
    profile_norm = profile / profile.max()

    # Detect strong peaks
    peak_strength = (profile_norm > 0.5).sum() / len(profile_norm)

    is_periodic = peak_strength > threshold
    return is_periodic, peak_strength


def score_component(
    component_mask: np.ndarray,
    corr_map: Optional[np.ndarray] = None,
    inlier_ratio: Optional[float] = None,
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 0.0,
    plausibility_prior: float = 1.0
) -> float:
    """
    Score a connected component based on multiple factors.

    Args:
        component_mask: Binary mask of single component (H, W)
        corr_map: Correlation map (H, W), optional
        inlier_ratio: RANSAC inlier ratio, optional
        alpha: Weight for correlation score
        beta: Weight for inlier ratio
        gamma: Weight for plausibility prior
        plausibility_prior: Domain-specific plausibility (e.g., band-shaped for gels)

    Returns:
        Component score in [0, 1]
    """
    score = 0.0
    total_weight = 0.0

    # Correlation score
    if corr_map is not None and alpha > 0:
        mean_corr = corr_map[component_mask > 0].mean() if component_mask.sum() > 0 else 0.0
        score += alpha * mean_corr
        total_weight += alpha

    # Inlier ratio score
    if inlier_ratio is not None and beta > 0:
        score += beta * inlier_ratio
        total_weight += beta

    # Plausibility prior
    if gamma > 0:
        score += gamma * plausibility_prior
        total_weight += gamma

    # Normalize
    if total_weight > 0:
        score /= total_weight

    return score


def component_aware_pruning(
    mask: np.ndarray,
    corr_map: Optional[np.ndarray] = None,
    min_score: float = 0.3,
    min_area: int = 24,
    max_area: Optional[int] = None,
    size_adaptive: bool = True
) -> np.ndarray:
    """
    Prune components based on quality scores and size.

    Args:
        mask: Binary mask (H, W)
        corr_map: Correlation map (H, W), optional
        min_score: Minimum component score to keep
        min_area: Minimum component area (pixels)
        max_area: Maximum component area (None for no limit)
        size_adaptive: Use size-adaptive thresholds (keep small high-quality components)

    Returns:
        Pruned binary mask
    """
    labeled = label(mask, connectivity=2)

    if labeled.max() == 0:
        return mask

    # Analyze each component
    kept_labels = []

    for region in regionprops(labeled):
        component_label = region.label
        component_area = region.area

        # Size filtering
        if component_area < min_area:
            continue
        if max_area is not None and component_area > max_area:
            continue

        # Extract component mask
        component_mask = (labeled == component_label).astype(np.uint8)

        # Compute score
        score = score_component(
            component_mask,
            corr_map=corr_map,
            alpha=1.0 if corr_map is not None else 0.0,
            beta=0.0,  # No inlier ratio available here
            gamma=0.0
        )

        # Size-adaptive threshold
        if size_adaptive:
            # Smaller components need higher scores
            area_factor = np.clip(component_area / 100.0, 0.5, 1.5)
            effective_threshold = min_score / area_factor
        else:
            effective_threshold = min_score

        # Keep if score exceeds threshold
        if score >= effective_threshold:
            kept_labels.append(component_label)

    # Build output mask
    if len(kept_labels) == 0:
        return np.zeros_like(mask)

    result = np.isin(labeled, kept_labels).astype(np.uint8)
    return result


def is_band_shaped(component_mask: np.ndarray, aspect_ratio_threshold: float = 3.0) -> bool:
    """
    Check if component is band-shaped (for gel forensics).

    Args:
        component_mask: Binary mask of single component
        aspect_ratio_threshold: Minimum aspect ratio to be considered band-shaped

    Returns:
        True if band-shaped
    """
    regions = regionprops(label(component_mask))

    if len(regions) == 0:
        return False

    region = regions[0]

    # Check aspect ratio of bounding box
    minr, minc, maxr, maxc = region.bbox
    height = maxr - minr
    width = maxc - minc

    if height == 0 or width == 0:
        return False

    aspect_ratio = max(height / width, width / height)

    return aspect_ratio >= aspect_ratio_threshold
