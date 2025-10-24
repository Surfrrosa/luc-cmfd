"""Ablation study runner for CMFD hyperparameter tuning.

Runs a grid search over specified parameters and logs results.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import itertools
import time
import hashlib
import pandas as pd
import numpy as np
import torch
from typing import Dict, List

from utils import set_all_seeds, setup_logger
from model import CMFDNet
from dataset import CMFDDataset, collate_fn
from torch.utils.data import DataLoader


def compute_config_hash(config: Dict) -> str:
    """Compute hash of configuration for unique identification."""
    config_str = str(sorted(config.items()))
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def evaluate_config(
    config: Dict,
    model: CMFDNet,
    val_loader: DataLoader,
    device: torch.device
) -> Dict:
    """
    Evaluate a single configuration.

    Args:
        config: Configuration dictionary
        model: CMFD model
        val_loader: Validation data loader
        device: Device

    Returns:
        Dictionary with metrics
    """
    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_time = 0
    n_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            batch_start = time.perf_counter()

            # Forward pass
            output = model(images)
            logits = output['logits']

            # Apply threshold
            pred_masks = (torch.sigmoid(logits) > config.get('post', {}).get('thr', 0.5)).float()

            batch_time = time.perf_counter() - batch_start

            # Compute metrics
            tp = (pred_masks * masks).sum().item()
            fp = (pred_masks * (1 - masks)).sum().item()
            fn = ((1 - pred_masks) * masks).sum().item()

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_time += batch_time
            n_samples += len(images)

    # Compute F1
    precision = total_tp / (total_tp + total_fp + 1e-7)
    recall = total_tp / (total_tp + total_fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)

    # Average time per image (ms)
    avg_time_ms = (total_time / n_samples) * 1000 if n_samples > 0 else 0

    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'runtime_ms': avg_time_ms
    }


def should_promote_config(baseline_f1: float, baseline_time: float,
                          new_f1: float, new_time: float) -> bool:
    """
    Decide if new config should be promoted.

    Promotion rule: ΔF1 >= +0.01 with ≤ +10% runtime increase.

    Args:
        baseline_f1: Baseline F1 score
        baseline_time: Baseline runtime (ms)
        new_f1: New F1 score
        new_time: New runtime (ms)

    Returns:
        True if config should be promoted
    """
    f1_improvement = new_f1 - baseline_f1
    time_increase_pct = ((new_time - baseline_time) / baseline_time) * 100

    if f1_improvement >= 0.01 and time_increase_pct <= 10:
        return True

    return False


def run_ablation_study(
    val_image_dir: str,
    val_mask_dir: str,
    output_csv: str = "ablation_results.csv",
    seed: int = 42
):
    """
    Run ablation study over hyperparameter grid.

    Args:
        val_image_dir: Validation images directory
        val_mask_dir: Validation masks directory
        output_csv: Output CSV file path
        seed: Random seed
    """
    # Setup
    set_all_seeds(seed)
    logger = setup_logger()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"Device: {device}")
    logger.info(f"Running ablation study...")

    # Load validation dataset
    val_dataset = CMFDDataset(
        image_dir=val_image_dir,
        mask_dir=val_mask_dir,
        normalize=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )

    logger.info(f"Validation set size: {len(val_dataset)}")

    # Define parameter grid
    param_grid = {
        'patch': [8, 12, 16],
        'stride': [4, 8],
        'top_k': [3, 5, 7],
        'ransac_model': ['similarity', 'affine'],
        'inlier_thresh_px': [1.5, 2.0],
        'tta': ['none', 'flip', 'flip+rot90'],
    }

    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    configs = [dict(zip(keys, v)) for v in itertools.product(*values)]

    logger.info(f"Total configurations to test: {len(configs)}")

    results = []
    baseline_f1 = None
    baseline_time = None

    # Test each configuration
    for i, config in enumerate(configs):
        logger.info(f"\nTesting config {i+1}/{len(configs)}")
        logger.info(f"Config: {config}")

        # Add default post-processing settings
        config['post'] = {
            'thr': 0.5,
            'min_area': 24,
            'morph': {'close': 3, 'open': 0}
        }

        # Create model with this config
        model = CMFDNet(
            backbone='dinov2_vits14',
            freeze_backbone=True,
            patch=config['patch'],
            stride=config['stride'],
            top_k=config['top_k']
        ).to(device)

        # Evaluate
        try:
            metrics = evaluate_config(config, model, val_loader, device)

            # Compute hash
            config_hash = compute_config_hash(config)

            # Set baseline from first config
            if baseline_f1 is None:
                baseline_f1 = metrics['f1']
                baseline_time = metrics['runtime_ms']

            # Check promotion
            promote = should_promote_config(
                baseline_f1, baseline_time,
                metrics['f1'], metrics['runtime_ms']
            )

            result_row = {
                'config_hash': config_hash,
                'patch': config['patch'],
                'stride': config['stride'],
                'top_k': config['top_k'],
                'ransac_model': config['ransac_model'],
                'inlier_thresh_px': config['inlier_thresh_px'],
                'tta': config['tta'],
                'f1': metrics['f1'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'runtime_ms': metrics['runtime_ms'],
                'promote': promote,
                'notes': 'baseline' if i == 0 else ''
            }

            results.append(result_row)

            logger.info(f"F1: {metrics['f1']:.4f}, Runtime: {metrics['runtime_ms']:.2f}ms")
            logger.info(f"Promote: {promote}")

        except Exception as e:
            logger.error(f"Error testing config: {e}")
            continue

    # Save results
    df = pd.DataFrame(results)
    df = df.sort_values('f1', ascending=False)
    df.to_csv(output_csv, index=False)

    logger.info(f"\nAblation study complete!")
    logger.info(f"Results saved to {output_csv}")
    logger.info(f"\nTop 5 configurations by F1:")
    logger.info("\n" + str(df.head(5)))

    # Count promoted configs
    n_promoted = df['promote'].sum()
    logger.info(f"\nPromoted configurations: {n_promoted}/{len(df)}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run ablation study')
    parser.add_argument('--val_images', type=str, required=True,
                       help='Validation images directory')
    parser.add_argument('--val_masks', type=str, required=True,
                       help='Validation masks directory')
    parser.add_argument('--output', type=str, default='ablation_results.csv',
                       help='Output CSV file')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    run_ablation_study(
        val_image_dir=args.val_images,
        val_mask_dir=args.val_masks,
        output_csv=args.output,
        seed=args.seed
    )
