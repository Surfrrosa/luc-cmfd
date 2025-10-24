"""Self-correlation computation for copy-move detection."""

from typing import List, Tuple, Dict
import torch
import torch.nn.functional as F
import numpy as np


def build_pyramid_feats(
    x: torch.Tensor,
    levels: List[int] = [1, 2, 4]
) -> List[torch.Tensor]:
    """
    Build multi-scale feature pyramid with L2 normalization.

    Args:
        x: Feature tensor (B, C, H, W)
        levels: Downsampling factors (1 = original scale)

    Returns:
        List of normalized feature tensors at different scales
    """
    pyramid = []

    for scale in levels:
        if scale == 1:
            feats = x
        else:
            # Downsample using average pooling
            feats = F.avg_pool2d(x, kernel_size=scale, stride=scale)

        # L2 normalize across channel dimension
        feats = F.normalize(feats, p=2, dim=1)

        pyramid.append(feats)

    return pyramid


def self_corr(
    feats: torch.Tensor,
    patch: int = 12,
    stride: int = 4,
    top_k: int = 5,
    exclude_self_radius: int = 2
) -> Dict[str, torch.Tensor]:
    """
    Compute self-correlation within a single image to detect duplicated regions.

    Args:
        feats: Normalized feature tensor (B, C, H, W)
        patch: Patch size for query (must be divisible by stride for simplicity)
        stride: Stride for extracting query locations
        top_k: Number of top matches to keep per query
        exclude_self_radius: Exclude matches within this radius from query

    Returns:
        Dictionary containing:
            - 'corr_map': Correlation map (B, top_k, H_out, W_out)
            - 'offset_map': Offset map (B, top_k, 2, H_out, W_out) with (dy, dx)
            - 'query_coords': Query grid coordinates (H_out, W_out, 2)
    """
    B, C, H, W = feats.shape

    # Extract all patches as potential targets (dense)
    # Using unfold to create overlapping patches
    patches = F.unfold(feats, kernel_size=patch, stride=stride, padding=0)
    # patches: (B, C*patch*patch, N_patches)

    N_patches = patches.shape[2]
    H_out = (H - patch) // stride + 1
    W_out = (W - patch) // stride + 1

    # Reshape: (B, C, patch*patch, N_patches)
    patches = patches.view(B, C, patch * patch, N_patches)

    # Average pool each patch to get patch descriptor
    patch_desc = patches.mean(dim=2)  # (B, C, N_patches)

    # Normalize
    patch_desc = F.normalize(patch_desc, p=2, dim=1)

    # Compute self-correlation matrix: (B, N_patches, N_patches)
    corr_matrix = torch.bmm(patch_desc.transpose(1, 2), patch_desc)  # (B, N, N)

    # Create mask to exclude self and nearby matches
    mask = create_exclusion_mask(
        H_out, W_out,
        exclude_radius=exclude_self_radius,
        device=feats.device
    )  # (N, N)

    # Apply mask (set excluded correlations to -inf)
    corr_matrix = corr_matrix.masked_fill(mask.unsqueeze(0), -1e9)

    # Get top-k matches for each query
    top_scores, top_indices = torch.topk(corr_matrix, k=top_k, dim=2)  # (B, N, k)

    # Compute offsets
    # Convert flat indices to 2D coordinates
    query_y, query_x = np.meshgrid(
        np.arange(H_out), np.arange(W_out), indexing='ij'
    )
    query_coords = np.stack([query_y, query_x], axis=-1)  # (H_out, W_out, 2)

    # Flatten query coords
    query_flat = torch.from_numpy(query_coords.reshape(-1, 2)).to(feats.device)  # (N, 2)

    # Target coords from indices
    target_indices = top_indices  # (B, N, k)
    target_y = target_indices // W_out
    target_x = target_indices % W_out
    target_coords = torch.stack([target_y, target_x], dim=-1)  # (B, N, k, 2)

    # Compute offsets: target - query
    offsets = target_coords - query_flat.unsqueeze(0).unsqueeze(2)  # (B, N, k, 2)

    # Reshape outputs to spatial format
    corr_map = top_scores.view(B, H_out, W_out, top_k).permute(0, 3, 1, 2)  # (B, k, H, W)
    offset_map = offsets.view(B, H_out, W_out, top_k, 2).permute(0, 3, 4, 1, 2)  # (B, k, 2, H, W)

    return {
        'corr_map': corr_map,
        'offset_map': offset_map,
        'query_coords': query_coords
    }


def create_exclusion_mask(
    h: int,
    w: int,
    exclude_radius: int,
    device: torch.device
) -> torch.Tensor:
    """
    Create mask to exclude self-matches and nearby patches.

    Args:
        h: Height of patch grid
        w: Width of patch grid
        exclude_radius: Radius for exclusion (in grid units)
        device: Target device

    Returns:
        Boolean mask (N, N) where N = h * w
    """
    n = h * w

    # Create coordinate grids
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    coords = np.stack([y.flatten(), x.flatten()], axis=1)  # (N, 2)

    # Compute pairwise distances
    dist_matrix = np.sqrt(
        (coords[:, None, 0] - coords[None, :, 0]) ** 2 +
        (coords[:, None, 1] - coords[None, :, 1]) ** 2
    )  # (N, N)

    # Create mask for distances <= exclude_radius
    mask = dist_matrix <= exclude_radius

    return torch.from_numpy(mask).to(device)


def extract_match_pairs(
    corr_result: Dict[str, torch.Tensor],
    score_threshold: float = 0.7,
    max_pairs: int = 1000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract point correspondences from correlation results.

    Args:
        corr_result: Output from self_corr()
        score_threshold: Minimum correlation score
        max_pairs: Maximum number of pairs to return

    Returns:
        Tuple of (query_pts, target_pts, scores):
            - query_pts: (N, 2) array of query coordinates (x, y)
            - target_pts: (N, 2) array of target coordinates (x, y)
            - scores: (N,) array of correlation scores
    """
    corr_map = corr_result['corr_map'][0]  # (k, H, W)
    offset_map = corr_result['offset_map'][0]  # (k, 2, H, W)
    query_coords = corr_result['query_coords']  # (H, W, 2)

    k, h, w = corr_map.shape

    # Flatten
    corr_flat = corr_map.reshape(k, -1).cpu().numpy()  # (k, H*W)
    offset_flat = offset_map.reshape(k, 2, -1).cpu().numpy()  # (k, 2, H*W)

    # Get query coordinates for each spatial location
    query_y, query_x = query_coords[..., 0], query_coords[..., 1]
    query_y_flat = np.tile(query_y.flatten(), k)  # (k * H*W,)
    query_x_flat = np.tile(query_x.flatten(), k)

    # Replicate for k matches per location
    corr_all = corr_flat.flatten()  # (k * H*W,)
    offset_y = offset_flat[:, 0, :].flatten()  # (k * H*W,)
    offset_x = offset_flat[:, 1, :].flatten()

    # Compute target coordinates
    target_y = query_y_flat + offset_y
    target_x = query_x_flat + offset_x

    # Filter by score
    valid = corr_all >= score_threshold
    query_x_valid = query_x_flat[valid]
    query_y_valid = query_y_flat[valid]
    target_x_valid = target_x[valid]
    target_y_valid = target_y[valid]
    scores_valid = corr_all[valid]

    # Sort by score and take top max_pairs
    if len(scores_valid) > max_pairs:
        top_idx = np.argsort(scores_valid)[-max_pairs:]
        query_x_valid = query_x_valid[top_idx]
        query_y_valid = query_y_valid[top_idx]
        target_x_valid = target_x_valid[top_idx]
        target_y_valid = target_y_valid[top_idx]
        scores_valid = scores_valid[top_idx]

    # Stack into (N, 2) format with (x, y) order
    query_pts = np.stack([query_x_valid, query_y_valid], axis=1)
    target_pts = np.stack([target_x_valid, target_y_valid], axis=1)

    return query_pts, target_pts, scores_valid


def aggregate_multiscale_corr(
    pyramid_results: List[Dict[str, torch.Tensor]],
    target_size: Tuple[int, int],
    weights: List[float] = None
) -> torch.Tensor:
    """
    Aggregate correlation maps from multiple scales.

    Args:
        pyramid_results: List of correlation results from different scales
        target_size: Target spatial size (H, W)
        weights: Optional weights for each scale (default: equal)

    Returns:
        Aggregated correlation map (B, k, H, W)
    """
    if weights is None:
        weights = [1.0] * len(pyramid_results)

    weights = torch.tensor(weights, device=pyramid_results[0]['corr_map'].device)
    weights = weights / weights.sum()

    # Resize and aggregate
    agg_corr = None

    for i, result in enumerate(pyramid_results):
        corr = result['corr_map']  # (B, k, H_i, W_i)

        # Resize to target size
        corr_resized = F.interpolate(
            corr, size=target_size, mode='bilinear', align_corners=False
        )

        # Weighted sum
        if agg_corr is None:
            agg_corr = weights[i] * corr_resized
        else:
            agg_corr += weights[i] * corr_resized

    return agg_corr
