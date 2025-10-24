#!/usr/bin/env python3
"""
Fast threshold sweep for post-processing optimization.

NO RETRAINING NEEDED - Uses saved validation predictions!

Usage:
    # First: Save validation predictions during/after training
    python scripts/sweep_threshold.py \
        --val_preds /kaggle/working/val_predictions.npz \
        --val_masks /path/to/val_masks \
        --thresholds 0.3 0.35 0.4 0.45 0.5 0.55 0.6 \
        --min_areas 8 12 16 20 24

Output: CSV with F1 scores for each combination
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, binary_opening, disk


def compute_f1(pred_mask, gt_mask):
    """Compute F1 score between prediction and ground truth."""
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()

    tp = (pred_flat * gt_flat).sum()
    fp = (pred_flat * (1 - gt_flat)).sum()
    fn = ((1 - pred_flat) * gt_flat).sum()

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)

    return f1


def apply_postprocessing(prob_mask, threshold, min_area, close_size=3, open_size=1):
    """Apply post-processing to probability mask."""

    # Threshold
    mask = (prob_mask >= threshold).astype(np.uint8)

    # Morphological operations
    if close_size > 0:
        mask = binary_closing(mask, disk(close_size)).astype(np.uint8)
    if open_size > 0:
        mask = binary_opening(mask, disk(open_size)).astype(np.uint8)

    # Min area filtering
    if min_area > 0:
        labeled = label(mask)
        for region in regionprops(labeled):
            if region.area < min_area:
                mask[labeled == region.label] = 0

    return mask


def main(args):
    print("Loading validation predictions...")
    data = np.load(args.val_preds)
    prob_masks = data['prob_masks']  # (N, H, W) float probabilities
    gt_masks = data['gt_masks']      # (N, H, W) binary ground truth
    case_ids = data['case_ids']      # (N,) image IDs

    n_images = len(prob_masks)
    print(f"Loaded {n_images} validation images")

    # Grid search
    results = []

    total_combos = len(args.thresholds) * len(args.min_areas)
    with tqdm(total=total_combos, desc="Sweeping") as pbar:
        for threshold in args.thresholds:
            for min_area in args.min_areas:

                f1_scores = []

                # Compute F1 for each image
                for i in range(n_images):
                    prob = prob_masks[i]
                    gt = gt_masks[i]

                    # Apply post-processing
                    pred = apply_postprocessing(
                        prob, threshold, min_area,
                        close_size=args.close_size,
                        open_size=args.open_size
                    )

                    # Compute F1
                    f1 = compute_f1(pred, gt)
                    f1_scores.append(f1)

                # Average F1
                mean_f1 = np.mean(f1_scores)
                std_f1 = np.std(f1_scores)

                results.append({
                    'threshold': threshold,
                    'min_area': min_area,
                    'close_size': args.close_size,
                    'open_size': args.open_size,
                    'mean_f1': mean_f1,
                    'std_f1': std_f1,
                    'n_images': n_images
                })

                pbar.update(1)
                pbar.set_postfix({'best_f1': max(r['mean_f1'] for r in results)})

    # Create DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values('mean_f1', ascending=False)

    # Save
    df.to_csv(args.output, index=False)
    print(f"\n✓ Results saved to {args.output}")

    # Show best
    print("\nTop 10 configurations:")
    print(df.head(10).to_string(index=False))

    # Best config
    best = df.iloc[0]
    print(f"\n🏆 BEST CONFIG:")
    print(f"   Threshold: {best['threshold']}")
    print(f"   Min Area: {best['min_area']}")
    print(f"   F1: {best['mean_f1']:.4f} ± {best['std_f1']:.4f}")

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sweep post-processing hyperparameters')
    parser.add_argument('--val_preds', type=str, required=True,
                       help='Path to saved validation predictions (.npz)')
    parser.add_argument('--thresholds', type=float, nargs='+',
                       default=[0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6],
                       help='Thresholds to try')
    parser.add_argument('--min_areas', type=int, nargs='+',
                       default=[8, 12, 16, 20, 24],
                       help='Min areas to try')
    parser.add_argument('--close_size', type=int, default=3,
                       help='Morphological closing size')
    parser.add_argument('--open_size', type=int, default=1,
                       help='Morphological opening size')
    parser.add_argument('--output', type=str, default='threshold_sweep.csv',
                       help='Output CSV path')

    args = parser.parse_args()
    sys.exit(main(args))
