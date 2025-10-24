"""Inference pipeline for copy-move forgery detection."""

import time
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

from .model import CMFDNet
from .dataset import CMFDDataset, collate_fn
from .corr import self_corr, extract_match_pairs
from .geom import estimate_transform_ransac, mask_from_inliers
from .post import clean_mask, component_aware_pruning, detect_periodicity
from .rle import rle_encode
from .utils import setup_logger, memory_stats

# Optional imports for new features
try:
    from .kp import kp_proposals_rgb
    HAS_KP = True
except ImportError:
    HAS_KP = False

try:
    from .pm import pm_proposals
    HAS_PM = True
except ImportError:
    HAS_PM = False


def infer_image(
    model: CMFDNet,
    image: torch.Tensor,
    config: Dict,
    device: torch.device,
    use_tta: bool = False
) -> np.ndarray:
    """
    Inference on a single image with full pipeline.

    Args:
        model: CMFD model
        image: Input image tensor (3, H, W)
        config: Configuration dictionary
        device: Device
        use_tta: Whether to use test-time augmentation

    Returns:
        Binary mask (H, W) as numpy array
    """
    model.eval()

    with torch.no_grad():
        # Add batch dimension
        x = image.unsqueeze(0).to(device)

        # Base prediction
        output = model.forward(x, return_features=True)
        logits = output['logits']  # (1, 1, H, W)

        # TTA
        if use_tta:
            tta_mode = config.get('tta', 'none')

            if tta_mode in ['flip', 'flip+rot90']:
                # Horizontal flip
                x_flip = torch.flip(x, dims=[3])
                output_flip = model.forward(x_flip)
                logits_flip = torch.flip(output_flip['logits'], dims=[3])

                # Average
                logits = (logits + logits_flip) / 2

            if tta_mode == 'flip+rot90':
                # 90-degree rotations
                for k in [1, 2, 3]:
                    x_rot = torch.rot90(x, k, dims=[2, 3])
                    output_rot = model.forward(x_rot)
                    logits_rot = torch.rot90(output_rot['logits'], -k, dims=[2, 3])
                    logits = logits + logits_rot

                logits = logits / 4  # Average over 4 augmentations

        # Convert to probability
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()  # (H, W)

        # Additional geometric verification using correlation
        corr_map = output['corr_map'][0]  # (k, H, W)
        features = output['features'][0]  # (C, H', W')

        # Extract correspondences
        corr_result = {
            'corr_map': output['corr_map'],
            'offset_map': torch.zeros_like(output['corr_map'][:, :, :2]),  # Placeholder
            'query_coords': np.zeros((corr_map.shape[1], corr_map.shape[2], 2))
        }

        # Get point matches
        try:
            query_pts, target_pts, scores = extract_match_pairs(
                corr_result,
                score_threshold=config.get('corr_threshold', 0.7),
                max_pairs=1000
            )

            # RANSAC geometric verification
            if len(query_pts) > 3:
                ransac_result = estimate_transform_ransac(
                    query_pts, target_pts,
                    model=config.get('ransac_model', 'similarity'),
                    inlier_thresh_px=config.get('inlier_thresh_px', 2.0)
                )

                # Create inlier mask
                if ransac_result['n_inliers'] > 3:
                    inlier_mask = mask_from_inliers(
                        ransac_result['inliers'],
                        query_pts,
                        shape=prob.shape,
                        growth=config.get('inlier_growth', 5)
                    )

                    # Combine with decoder prediction
                    prob = np.maximum(prob, inlier_mask * 0.5)
        except Exception as e:
            # Fallback to decoder only if geometric verification fails
            pass

        # Keypoint proposals (if enabled)
        kp_mask = None
        if config.get('keypoints', {}).get('enable', False) and HAS_KP:
            try:
                # Convert image to numpy RGB
                img_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

                kp_result = kp_proposals_rgb(
                    img_np,
                    method=config.get('keypoints', {}).get('method', 'ORB'),
                    max_kp=config.get('keypoints', {}).get('max_kp', 1500),
                    top_matches=config.get('keypoints', {}).get('top_matches', 300)
                )

                if kp_result is not None and len(kp_result['qxy']) >= 8:
                    # Run RANSAC on keypoint matches
                    kp_ransac = estimate_transform_ransac(
                        kp_result['qxy'],
                        kp_result['txy'],
                        model=config.get('geom', {}).get('model', 'affine'),
                        inlier_thresh_px=config.get('geom', {}).get('inlier_thresh_px', 3.0)
                    )

                    if kp_ransac['n_inliers'] > 8:
                        kp_mask = mask_from_inliers(
                            kp_ransac['inliers'],
                            kp_result['qxy'],
                            shape=prob.shape,
                            growth=config.get('geom', {}).get('inlier_growth', 5)
                        )
            except Exception as e:
                pass  # Fallback gracefully

        # PatchMatch proposals (if enabled)
        pm_mask = None
        if config.get('patchmatch', {}).get('enable', False) and HAS_PM:
            try:
                # Run PatchMatch on features
                pm_result = pm_proposals(
                    features.unsqueeze(0),  # Add batch dim
                    iters=config.get('patchmatch', {}).get('iters', 3),
                    topk=config.get('patchmatch', {}).get('topk', 50),
                    score_thr=config.get('patchmatch', {}).get('score_thr', 0.2),
                    use_pyramid=config.get('patchmatch', {}).get('use_pyramid', True)
                )

                if isinstance(pm_result['qxy'], torch.Tensor):
                    qxy = pm_result['qxy'].cpu().numpy()
                    txy = pm_result['txy'].cpu().numpy()
                else:
                    qxy, txy = pm_result['qxy'], pm_result['txy']

                if len(qxy) >= 8:
                    # Scale coordinates from feature space to image space
                    h_scale = prob.shape[0] / features.shape[1]
                    w_scale = prob.shape[1] / features.shape[2]
                    qxy_scaled = qxy * np.array([w_scale, h_scale])
                    txy_scaled = txy * np.array([w_scale, h_scale])

                    pm_ransac = estimate_transform_ransac(
                        qxy_scaled,
                        txy_scaled,
                        model=config.get('geom', {}).get('model', 'affine'),
                        inlier_thresh_px=config.get('geom', {}).get('inlier_thresh_px', 3.0)
                    )

                    if pm_ransac['n_inliers'] > 8:
                        pm_mask = mask_from_inliers(
                            pm_ransac['inliers'],
                            qxy_scaled,
                            shape=prob.shape,
                            growth=config.get('geom', {}).get('inlier_growth', 5)
                        )
            except Exception as e:
                pass  # Fallback gracefully

        # Fuse multiple predictions
        if kp_mask is not None:
            prob = np.maximum(prob, kp_mask * 0.6)
        if pm_mask is not None:
            prob = np.maximum(prob, pm_mask * 0.7)

        # Clamp probabilities to valid range for numerical stability
        prob = np.clip(prob, 0.0, 1.0)

        # Post-processing with component-aware pruning
        use_component_pruning = config.get('post', {}).get('component_pruning', {}).get('enable', False)

        if use_component_pruning:
            # Initial threshold
            mask = (prob >= config.get('post', {}).get('thr', 0.5)).astype(np.uint8)

            # Apply morphological operations first
            morph_cfg = config.get('post', {}).get('morph', {'close': 3, 'open': 0})
            if morph_cfg.get('close', 0) > 0:
                from skimage.morphology import binary_closing, disk
                mask = binary_closing(mask, disk(morph_cfg['close'])).astype(np.uint8)
            if morph_cfg.get('open', 0) > 0:
                from skimage.morphology import binary_opening, disk
                mask = binary_opening(mask, disk(morph_cfg['open'])).astype(np.uint8)

            # Component-aware pruning
            mask = component_aware_pruning(
                mask,
                corr_map=corr_map.max(dim=0)[0].cpu().numpy(),  # Max over top-k channels
                min_score=config.get('post', {}).get('component_pruning', {}).get('min_score', 0.3),
                min_area=config.get('post', {}).get('min_area', 24),
                size_adaptive=config.get('post', {}).get('component_pruning', {}).get('size_adaptive', True)
            )
        else:
            # Standard clean_mask
            mask = clean_mask(
                prob,
                thr=config.get('post', {}).get('thr', 0.5),
                min_area=config.get('post', {}).get('min_area', 24),
                morph=config.get('post', {}).get('morph', {'close': 3, 'open': 0})
            )

    return mask


def batch_inference(
    model: CMFDNet,
    dataloader: DataLoader,
    config: Dict,
    device: torch.device,
    logger=None
) -> List[Dict]:
    """
    Run inference on a dataset.

    Args:
        model: CMFD model
        dataloader: Data loader
        config: Configuration
        device: Device
        logger: Logger instance

    Returns:
        List of results with case_id and mask
    """
    model.eval()
    results = []

    total_time = 0
    total_images = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            images = batch['image'].to(device)
            case_ids = batch['case_id']
            original_sizes = batch['original_size']

            batch_start = time.perf_counter()

            # Process each image in batch
            for i, (img, case_id, orig_size) in enumerate(
                zip(images, case_ids, original_sizes)
            ):
                # Inference
                mask = infer_image(
                    model, img, config, device,
                    use_tta=(config.get('tta', 'none') != 'none')
                )

                # Resize to original size if needed
                if mask.shape != orig_size:
                    import cv2
                    mask = cv2.resize(
                        mask.astype(np.uint8),
                        (orig_size[1], orig_size[0]),  # (W, H)
                        interpolation=cv2.INTER_NEAREST
                    )

                results.append({
                    'case_id': case_id,
                    'mask': mask
                })

                total_images += 1

            batch_time = time.perf_counter() - batch_start
            total_time += batch_time

    # Log stats
    avg_time = total_time / total_images if total_images > 0 else 0
    if logger:
        logger.info(f"Processed {total_images} images in {total_time:.2f}s")
        logger.info(f"Average time per image: {avg_time * 1000:.2f}ms")

        mem = memory_stats(device)
        logger.info(f"Peak GPU memory: {mem['allocated_gb']:.2f}GB")

    return results


def create_submission(
    results: List[Dict],
    output_path: str,
    logger=None
) -> None:
    """
    Create submission file from inference results.

    Args:
        results: List of dicts with 'case_id' and 'mask'
        output_path: Path to save submission.csv
        logger: Logger instance
    """
    rows = []

    for result in results:
        case_id = result['case_id']
        mask = result['mask']

        # Encode mask to RLE
        annotation = rle_encode(mask)

        rows.append({
            'case_id': case_id,
            'annotation': annotation
        })

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Save
    df.to_csv(output_path, index=False)

    if logger:
        logger.info(f"Submission saved to {output_path}")
        logger.info(f"Total submissions: {len(df)}")

        # Count authentic vs forged
        n_authentic = (df['annotation'] == 'authentic').sum()
        n_forged = len(df) - n_authentic
        logger.info(f"Authentic: {n_authentic}, Forged: {n_forged}")


def main(
    weights_path: str,
    image_dir: str,
    output_path: str,
    config_path: Optional[str] = None,
    batch_size: int = 4,
    num_workers: int = 4
):
    """
    Main inference script.

    Args:
        weights_path: Path to model weights
        image_dir: Directory containing images
        output_path: Path to save submission.csv
        config_path: Path to config YAML (optional)
        batch_size: Batch size
        num_workers: Number of data loader workers
    """
    # Setup
    logger = setup_logger()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"Device: {device}")
    logger.info(f"Image directory: {image_dir}")
    logger.info(f"Output: {output_path}")

    # Load config
    if config_path:
        from .utils import load_config
        config = load_config(config_path)
    else:
        # Default config
        config = {
            'backbone': 'dinov2_vits14',
            'freeze_backbone': True,
            'patch': 12,
            'stride': 4,
            'top_k': 5,
            'ransac_model': 'similarity',
            'inlier_thresh_px': 1.5,
            'tta': 'flip',
            'post': {
                'thr': 0.5,
                'min_area': 24,
                'morph': {'close': 3, 'open': 0}
            }
        }

    logger.info(f"Config: {config}")

    # Create model
    logger.info("Loading model...")
    model = CMFDNet(
        backbone=config['backbone'],
        freeze_backbone=config['freeze_backbone'],
        patch=config['patch'],
        stride=config['stride'],
        top_k=config['top_k']
    )

    # Load weights
    if Path(weights_path).exists():
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded weights from {weights_path}")
    else:
        logger.warning(f"Weights not found at {weights_path}, using random init")

    model = model.to(device)
    model.eval()

    # Create dataset
    logger.info("Creating dataset...")
    dataset = CMFDDataset(
        image_dir=image_dir,
        normalize=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    logger.info(f"Dataset size: {len(dataset)}")

    # Run inference
    logger.info("Running inference...")
    results = batch_inference(model, dataloader, config, device, logger)

    # Create submission
    logger.info("Creating submission...")
    create_submission(results, output_path, logger)

    logger.info("Done!")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--output', type=str, default='submission.csv')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()

    main(
        weights_path=args.weights,
        image_dir=args.image_dir,
        output_path=args.output,
        config_path=args.config,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
