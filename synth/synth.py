"""Synthetic copy-move forgery generator for biomedical images."""

from typing import Tuple, Optional, List
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.transform import rotate as sk_rotate


def make_copy_move(
    img: np.ndarray,
    max_rois: int = 2,
    min_roi_size: int = 40,
    max_roi_size: int = 120,
    rotation_range: Tuple[float, float] = (-20, 20),
    scale_range: Tuple[float, float] = (0.8, 1.2),
    elastic_alpha: float = 20.0,
    elastic_sigma: float = 4.0,
    add_noise: bool = True,
    alpha_range: Tuple[float, float] = (0.85, 1.0)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic copy-move forgery in a biomedical image.

    Args:
        img: Input image (H, W, 3) in range [0, 1]
        max_rois: Maximum number of ROIs to copy-move
        min_roi_size: Minimum ROI dimension
        max_roi_size: Maximum ROI dimension
        rotation_range: Rotation angle range in degrees
        scale_range: Scale factor range
        elastic_alpha: Elastic deformation strength
        elastic_sigma: Elastic deformation smoothness
        add_noise: Whether to add noise to pasted region
        alpha_range: Blending alpha range

    Returns:
        Tuple of (forged_image, mask):
            - forged_image: Image with copy-move forgery (H, W, 3)
            - mask: Binary mask of forged regions (H, W)
    """
    h, w = img.shape[:2]
    forged = img.copy()
    mask = np.zeros((h, w), dtype=np.uint8)

    # Number of ROIs to copy
    n_rois = np.random.randint(1, max_rois + 1)

    for _ in range(n_rois):
        # Random ROI size
        roi_h = np.random.randint(min_roi_size, max_roi_size)
        roi_w = np.random.randint(min_roi_size, max_roi_size)

        # Source location
        src_y = np.random.randint(0, max(1, h - roi_h))
        src_x = np.random.randint(0, max(1, w - roi_w))

        # Extract source ROI
        src_roi = forged[src_y:src_y + roi_h, src_x:src_x + roi_w].copy()

        # Apply transformations
        transformed_roi, transform_mask = transform_roi(
            src_roi,
            rotation_range=rotation_range,
            scale_range=scale_range,
            elastic_alpha=elastic_alpha,
            elastic_sigma=elastic_sigma,
            add_noise=add_noise
        )

        # Target location (ensure no overlap with source)
        max_attempts = 20
        for _ in range(max_attempts):
            tgt_y = np.random.randint(0, max(1, h - transformed_roi.shape[0]))
            tgt_x = np.random.randint(0, max(1, w - transformed_roi.shape[1]))

            # Check overlap with source
            if not regions_overlap(
                (src_y, src_x, src_y + roi_h, src_x + roi_w),
                (tgt_y, tgt_x, tgt_y + transformed_roi.shape[0], tgt_x + transformed_roi.shape[1])
            ):
                break

        # Paste transformed ROI
        th, tw = transformed_roi.shape[:2]
        if tgt_y + th <= h and tgt_x + tw <= w:
            # Blending alpha
            alpha = np.random.uniform(alpha_range[0], alpha_range[1])

            # Apply alpha blending
            forged[tgt_y:tgt_y + th, tgt_x:tgt_x + tw] = (
                alpha * transformed_roi +
                (1 - alpha) * forged[tgt_y:tgt_y + th, tgt_x:tgt_x + tw]
            )

            # Update mask
            mask[tgt_y:tgt_y + th, tgt_x:tgt_x + tw] = np.maximum(
                mask[tgt_y:tgt_y + th, tgt_x:tgt_x + tw],
                transform_mask
            )

    return forged, mask


def transform_roi(
    roi: np.ndarray,
    rotation_range: Tuple[float, float],
    scale_range: Tuple[float, float],
    elastic_alpha: float,
    elastic_sigma: float,
    add_noise: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply transformations to ROI.

    Args:
        roi: ROI image (H, W, 3)
        rotation_range: Rotation angle range
        scale_range: Scale range
        elastic_alpha: Elastic deformation strength
        elastic_sigma: Elastic deformation smoothness
        add_noise: Whether to add noise

    Returns:
        Tuple of (transformed_roi, mask)
    """
    transformed = roi.copy()

    # Rotation
    if np.random.rand() > 0.3:
        angle = np.random.uniform(rotation_range[0], rotation_range[1])
        transformed = sk_rotate(
            transformed, angle, resize=True,
            mode='constant', cval=0, preserve_range=True
        )

    # Scale
    if np.random.rand() > 0.3:
        scale = np.random.uniform(scale_range[0], scale_range[1])
        new_h = int(transformed.shape[0] * scale)
        new_w = int(transformed.shape[1] * scale)
        transformed = cv2.resize(transformed, (new_w, new_h))

    # Elastic deformation
    if np.random.rand() > 0.5:
        transformed = elastic_transform(
            transformed,
            alpha=elastic_alpha,
            sigma=elastic_sigma
        )

    # Adjust brightness/contrast
    if np.random.rand() > 0.4:
        # Brightness
        brightness_delta = np.random.uniform(-0.1, 0.1)
        transformed = np.clip(transformed + brightness_delta, 0, 1)

        # Contrast
        contrast_factor = np.random.uniform(0.9, 1.1)
        transformed = np.clip((transformed - 0.5) * contrast_factor + 0.5, 0, 1)

    # Add noise
    if add_noise and np.random.rand() > 0.3:
        noise_type = np.random.choice(['gaussian', 'poisson'])

        if noise_type == 'gaussian':
            sigma = np.random.uniform(0.01, 0.03)
            noise = np.random.normal(0, sigma, transformed.shape)
            transformed = np.clip(transformed + noise, 0, 1)
        else:  # poisson
            # Scale to higher range for poisson, then normalize back
            scaled = transformed * 255
            noisy = np.random.poisson(scaled) / 255
            transformed = np.clip(noisy, 0, 1)

    # Blur or sharpen
    if np.random.rand() > 0.5:
        if np.random.rand() > 0.5:
            # Blur
            sigma = np.random.uniform(0.5, 1.5)
            transformed = gaussian_filter(transformed, sigma=(sigma, sigma, 0))
        else:
            # Sharpen
            blurred = gaussian_filter(transformed, sigma=(1, 1, 0))
            transformed = np.clip(transformed + 0.3 * (transformed - blurred), 0, 1)

    # Create binary mask for the transformed ROI
    # Use alpha channel or create from non-zero pixels
    if transformed.shape[2] == 4:
        mask = (transformed[..., 3] > 0.5).astype(np.uint8)
        transformed = transformed[..., :3]
    else:
        # Create mask from non-black pixels
        mask = (transformed.sum(axis=2) > 0.01).astype(np.uint8)

    return transformed.astype(np.float32), mask


def elastic_transform(
    image: np.ndarray,
    alpha: float,
    sigma: float
) -> np.ndarray:
    """
    Apply elastic deformation to image.

    Args:
        image: Input image (H, W, C)
        alpha: Deformation strength
        sigma: Smoothness of deformation

    Returns:
        Deformed image
    """
    h, w = image.shape[:2]

    # Random displacement fields
    dx = gaussian_filter(
        (np.random.rand(h, w) * 2 - 1),
        sigma, mode='constant', cval=0
    ) * alpha

    dy = gaussian_filter(
        (np.random.rand(h, w) * 2 - 1),
        sigma, mode='constant', cval=0
    ) * alpha

    # Meshgrid
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    # Apply displacement
    indices = [
        np.clip(y + dy, 0, h - 1),
        np.clip(x + dx, 0, w - 1)
    ]

    # Apply to each channel
    if image.ndim == 3:
        result = np.zeros_like(image)
        for c in range(image.shape[2]):
            result[..., c] = map_coordinates(
                image[..., c], indices, order=1, mode='reflect'
            )
    else:
        result = map_coordinates(image, indices, order=1, mode='reflect')

    return result


def regions_overlap(
    box1: Tuple[int, int, int, int],
    box2: Tuple[int, int, int, int]
) -> bool:
    """
    Check if two bounding boxes overlap.

    Args:
        box1: (y0, x0, y1, x1)
        box2: (y0, x0, y1, x1)

    Returns:
        True if boxes overlap
    """
    y0_1, x0_1, y1_1, x1_1 = box1
    y0_2, x0_2, y1_2, x1_2 = box2

    return not (x1_1 < x0_2 or x1_2 < x0_1 or y1_1 < y0_2 or y1_2 < y0_1)


def add_biomedical_artifacts(
    img: np.ndarray,
    add_scale_bar: bool = True,
    add_text: bool = True
) -> np.ndarray:
    """
    Add common biomedical image artifacts (scale bars, text overlays).

    Args:
        img: Input image (H, W, 3)
        add_scale_bar: Whether to add scale bar
        add_text: Whether to add text overlay

    Returns:
        Image with artifacts
    """
    result = img.copy()
    h, w = img.shape[:2]

    # Scale bar
    if add_scale_bar and np.random.rand() > 0.5:
        bar_h = int(h * 0.02)
        bar_w = int(w * 0.15)
        bar_y = int(h * 0.92)
        bar_x = int(w * 0.05)

        # White bar
        result[bar_y:bar_y + bar_h, bar_x:bar_x + bar_w] = 1.0

    # Text overlay (simulate labels)
    if add_text and np.random.rand() > 0.5:
        # Create simple text-like rectangles
        text_h = int(h * 0.04)
        text_w = int(w * 0.1)
        text_y = int(h * 0.05)
        text_x = int(w * 0.85)

        result[text_y:text_y + text_h, text_x:text_x + text_w] = 0.9

    return result


def create_authentic_sample(
    img: np.ndarray,
    add_artifacts: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create an authentic (non-forged) sample.

    Args:
        img: Input image (H, W, 3)
        add_artifacts: Whether to add biomedical artifacts

    Returns:
        Tuple of (image, empty_mask)
    """
    result = img.copy()

    if add_artifacts:
        result = add_biomedical_artifacts(result)

    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    return result, mask
