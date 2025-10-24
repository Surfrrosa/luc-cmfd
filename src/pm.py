"""Deep PatchMatch for Copy-Move Forgery Detection.

Based on "Image Copy-Move Forgery Detection via Deep PatchMatch
and Pairwise Ranking Learning" (2024).

Implements multi-scale differentiable PatchMatch with pairwise ranking
for robust source-target correspondence discovery.
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F


def build_feature_pyramid(
    feats: torch.Tensor,
    scales: int = 3
) -> list[torch.Tensor]:
    """
    Build feature pyramid by progressive downsampling.

    Args:
        feats: Features (B, C, H, W)
        scales: Number of pyramid levels

    Returns:
        List of features from coarse to fine
    """
    pyramid = []
    current = feats

    for _ in range(scales):
        pyramid.append(current)
        # Downsample by 2x
        current = F.avg_pool2d(current, kernel_size=2, stride=2)

    # Return coarse-to-fine
    return pyramid[::-1]


def normalize_features(feats: torch.Tensor) -> torch.Tensor:
    """L2 normalize features along channel dimension."""
    return F.normalize(feats, dim=1, p=2)


def random_search(
    query_feats: torch.Tensor,
    target_feats: torch.Tensor,
    current_offset: torch.Tensor,
    search_radius: int = 32,
    num_samples: int = 8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Random search step of PatchMatch.

    Args:
        query_feats: Query features (B, C, H, W)
        target_feats: Target features (B, C, H, W)
        current_offset: Current best offsets (B, H, W, 2)
        search_radius: Maximum search radius
        num_samples: Number of random samples per position

    Returns:
        best_offset: Updated offsets (B, H, W, 2)
        best_score: Matching scores (B, H, W)
    """
    B, C, H, W = query_feats.shape
    device = query_feats.device

    # Generate random offsets
    random_offsets = torch.rand(B, H, W, num_samples, 2, device=device) * 2 - 1
    random_offsets = random_offsets * search_radius
    random_offsets = random_offsets.long()

    # Add to current offset
    candidate_offsets = current_offset.unsqueeze(3) + random_offsets  # (B, H, W, num_samples, 2)

    # Evaluate all candidates
    best_offset = current_offset.clone()
    best_score = compute_matching_score(query_feats, target_feats, current_offset)

    for i in range(num_samples):
        offset = candidate_offsets[..., i, :]
        score = compute_matching_score(query_feats, target_feats, offset)

        # Update if better
        mask = score > best_score
        best_offset[mask] = offset[mask]
        best_score[mask] = score[mask]

    return best_offset, best_score


def propagation_step(
    query_feats: torch.Tensor,
    target_feats: torch.Tensor,
    current_offset: torch.Tensor,
    current_score: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Propagation step: check neighbors' offsets.

    Args:
        query_feats: Query features (B, C, H, W)
        target_feats: Target features (B, C, H, W)
        current_offset: Current offsets (B, H, W, 2)
        current_score: Current scores (B, H, W)

    Returns:
        Updated offsets and scores
    """
    B, C, H, W = query_feats.shape

    best_offset = current_offset.clone()
    best_score = current_score.clone()

    # Check 4-connected neighbors
    for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        # Shift offset field
        shifted_offset = torch.roll(current_offset, shifts=(dy, dx), dims=(1, 2))

        # Evaluate at neighbor's offset
        score = compute_matching_score(query_feats, target_feats, shifted_offset)

        # Update if better
        mask = score > best_score
        best_offset[mask] = shifted_offset[mask]
        best_score[mask] = score[mask]

    return best_offset, best_score


def compute_matching_score(
    query_feats: torch.Tensor,
    target_feats: torch.Tensor,
    offset: torch.Tensor,
    exclusion_radius: int = 5
) -> torch.Tensor:
    """
    Compute normalized dot product similarity with exclusion.

    Args:
        query_feats: Query features (B, C, H, W)
        target_feats: Target features (B, C, H, W)
        offset: Flow field (B, H, W, 2) in pixels
        exclusion_radius: Exclude matches within this radius (avoid trivial matches)

    Returns:
        scores: Matching scores (B, H, W)
    """
    B, C, H, W = query_feats.shape
    device = query_feats.device

    # Create grid
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    grid = torch.stack([x_grid, y_grid], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

    # Target coordinates
    target_coords = grid + offset  # (B, H, W, 2)

    # Normalize to [-1, 1] for grid_sample
    target_coords_norm = target_coords.clone()
    target_coords_norm[..., 0] = 2.0 * target_coords[..., 0] / (W - 1) - 1.0
    target_coords_norm[..., 1] = 2.0 * target_coords[..., 1] / (H - 1) - 1.0

    # Sample target features
    sampled_feats = F.grid_sample(
        target_feats,
        target_coords_norm,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )

    # Compute similarity (normalized dot product)
    scores = (query_feats * sampled_feats).sum(dim=1)  # (B, H, W)

    # Exclude trivial matches (same position ± radius)
    dist = offset.pow(2).sum(dim=-1).sqrt()
    exclusion_mask = dist <= exclusion_radius
    scores[exclusion_mask] = -1.0

    return scores


def patchmatch_iteration(
    query_feats: torch.Tensor,
    target_feats: torch.Tensor,
    offset: torch.Tensor,
    score: torch.Tensor,
    search_radius: int = 32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Single PatchMatch iteration: propagate + random search.

    Args:
        query_feats: Normalized query features (B, C, H, W)
        target_feats: Normalized target features (B, C, H, W)
        offset: Current offsets (B, H, W, 2)
        score: Current scores (B, H, W)
        search_radius: Random search radius

    Returns:
        Updated offset and score
    """
    # Propagation
    offset, score = propagation_step(query_feats, target_feats, offset, score)

    # Random search
    offset, score = random_search(query_feats, target_feats, offset, search_radius)

    return offset, score


def pm_proposals(
    feats: torch.Tensor,
    iters: int = 3,
    topk: int = 50,
    score_thr: float = 0.2,
    use_pyramid: bool = True,
    exclusion_radius: int = 5
) -> Dict[str, torch.Tensor]:
    """
    Deep PatchMatch for copy-move correspondence discovery.

    Args:
        feats: Input features (B, C, H, W), assumed L2-normalized
        iters: Number of PatchMatch iterations per scale
        topk: Number of top correspondences to return
        score_thr: Minimum score threshold
        use_pyramid: Use multi-scale pyramid
        exclusion_radius: Exclude matches within this radius

    Returns:
        Dictionary with:
            'qxy': Query positions (B, N, 2) in [x, y] format
            'txy': Target positions (B, N, 2)
            'score': Matching scores (B, N)
            'offset': Full offset field (B, H, W, 2)
    """
    B, C, H, W = feats.shape
    device = feats.device

    # Normalize features
    feats = normalize_features(feats)

    if use_pyramid:
        # Multi-scale PatchMatch
        pyramid = build_feature_pyramid(feats, scales=3)

        # Initialize at coarsest scale
        query_feats = pyramid[0]
        target_feats = query_feats  # Self-matching
        h, w = query_feats.shape[2:]

        # Random initialization
        offset = torch.rand(B, h, w, 2, device=device) * 2 - 1
        offset = offset * torch.tensor([w, h], device=device).view(1, 1, 1, 2)
        offset = offset.long().float()

        score = compute_matching_score(query_feats, target_feats, offset, exclusion_radius)

        # Coarse-to-fine refinement
        for scale_idx, scale_feats in enumerate(pyramid):
            query_feats = scale_feats
            target_feats = scale_feats

            # Upsample offset from previous scale
            if scale_idx > 0:
                offset = F.interpolate(
                    offset.permute(0, 3, 1, 2),
                    size=(query_feats.shape[2], query_feats.shape[3]),
                    mode='bilinear',
                    align_corners=True
                ).permute(0, 2, 3, 1)
                offset = offset * 2  # Scale offset values

                score = compute_matching_score(query_feats, target_feats, offset, exclusion_radius)

            # PatchMatch iterations at this scale
            search_radius = max(32 // (2 ** scale_idx), 8)
            for _ in range(iters):
                offset, score = patchmatch_iteration(
                    query_feats, target_feats, offset, score, search_radius
                )
    else:
        # Single-scale PatchMatch
        query_feats = feats
        target_feats = feats

        # Random initialization
        offset = torch.rand(B, H, W, 2, device=device) * 2 - 1
        offset = offset * torch.tensor([W, H], device=device).view(1, 1, 1, 2)
        offset = offset.long().float()

        score = compute_matching_score(query_feats, target_feats, offset, exclusion_radius)

        # PatchMatch iterations
        for _ in range(iters):
            offset, score = patchmatch_iteration(query_feats, target_feats, offset, score)

    # Extract top-k correspondences
    B, H, W = score.shape

    # Flatten and get top-k
    score_flat = score.view(B, -1)
    topk_vals, topk_idx = torch.topk(score_flat, k=min(topk, H * W), dim=1)

    # Filter by threshold
    valid_mask = topk_vals > score_thr

    results = []
    for b in range(B):
        valid_k = valid_mask[b].sum().item()
        if valid_k == 0:
            # No valid matches
            results.append({
                'qxy': torch.zeros(0, 2, device=device),
                'txy': torch.zeros(0, 2, device=device),
                'score': torch.zeros(0, device=device)
            })
            continue

        # Get valid indices
        batch_idx = topk_idx[b, :valid_k]

        # Convert to (y, x) coordinates
        y_coords = batch_idx // W
        x_coords = batch_idx % W

        # Query positions
        qxy = torch.stack([x_coords, y_coords], dim=1).float()  # (N, 2)

        # Target positions (query + offset)
        batch_offset = offset[b]  # (H, W, 2)
        txy = qxy.clone()
        for i, (y, x) in enumerate(zip(y_coords, x_coords)):
            txy[i] = qxy[i] + batch_offset[y, x]

        results.append({
            'qxy': qxy,
            'txy': txy,
            'score': topk_vals[b, :valid_k]
        })

    # Batch results
    if B == 1:
        return {
            'qxy': results[0]['qxy'],
            'txy': results[0]['txy'],
            'score': results[0]['score'],
            'offset': offset[0]
        }
    else:
        return {
            'qxy': [r['qxy'] for r in results],
            'txy': [r['txy'] for r in results],
            'score': [r['score'] for r in results],
            'offset': offset
        }


def pm_proposals_np(
    feats: np.ndarray,
    iters: int = 3,
    topk: int = 50,
    score_thr: float = 0.2
) -> Dict[str, np.ndarray]:
    """
    NumPy wrapper for pm_proposals.

    Args:
        feats: Features (H, W, C) or (C, H, W) numpy array
        iters: PatchMatch iterations
        topk: Number of top correspondences
        score_thr: Score threshold

    Returns:
        Dictionary with numpy arrays
    """
    # Convert to tensor
    if feats.ndim == 3:
        if feats.shape[-1] < feats.shape[0]:  # (C, H, W)
            feats_t = torch.from_numpy(feats).unsqueeze(0).float()
        else:  # (H, W, C)
            feats_t = torch.from_numpy(feats).permute(2, 0, 1).unsqueeze(0).float()
    else:
        raise ValueError(f"Expected 3D array, got shape {feats.shape}")

    # Run PatchMatch
    results = pm_proposals(feats_t, iters=iters, topk=topk, score_thr=score_thr)

    # Convert back to numpy
    return {
        'qxy': results['qxy'].cpu().numpy(),
        'txy': results['txy'].cpu().numpy(),
        'score': results['score'].cpu().numpy()
    }
