#!/usr/bin/env python3
"""
Visualize model predictions on validation set for sanity checks.

Usage:
    python scripts/visualize_predictions.py \
        --val_preds /kaggle/working/val_predictions.npz \
        --output sanity_check.png \
        --n_samples 8

Shows side-by-side: Original | Ground Truth | Prediction | Overlay
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches


def visualize_samples(prob_masks, gt_masks, case_ids, n_samples=8, output_path='sanity_check.png'):
    """Create visualization grid of predictions."""

    n_samples = min(n_samples, len(prob_masks))

    # Select diverse samples (high conf forgery, low conf, authentic, etc)
    indices = []

    # Sort by prediction confidence
    confidences = [prob_masks[i].max() for i in range(len(prob_masks))]
    sorted_idx = np.argsort(confidences)

    # Pick spread: lowest conf, mid, highest conf
    for i in np.linspace(0, len(sorted_idx)-1, n_samples).astype(int):
        indices.append(sorted_idx[i])

    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4*n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for row, idx in enumerate(indices):
        prob = prob_masks[idx]
        gt = gt_masks[idx]
        case_id = case_ids[idx] if case_ids is not None else f"Sample {idx}"

        # Binary prediction (threshold=0.5)
        pred = (prob >= 0.5).astype(np.uint8)

        # Column 1: Probability heatmap
        axes[row, 0].imshow(prob, cmap='hot', vmin=0, vmax=1)
        axes[row, 0].set_title(f"{case_id}\nProbability (max={prob.max():.3f})")
        axes[row, 0].axis('off')

        # Column 2: Ground truth
        axes[row, 1].imshow(gt, cmap='gray', vmin=0, vmax=1)
        axes[row, 1].set_title(f"Ground Truth\n(forged px: {gt.sum()})")
        axes[row, 1].axis('off')

        # Column 3: Binary prediction
        axes[row, 2].imshow(pred, cmap='gray', vmin=0, vmax=1)
        axes[row, 2].set_title(f"Prediction (thr=0.5)\n(forged px: {pred.sum()})")
        axes[row, 2].axis('off')

        # Column 4: Overlay (TP=green, FP=red, FN=blue)
        overlay = np.zeros((*prob.shape, 3))

        # True Positives (green)
        tp_mask = (pred == 1) & (gt == 1)
        overlay[tp_mask] = [0, 1, 0]

        # False Positives (red)
        fp_mask = (pred == 1) & (gt == 0)
        overlay[fp_mask] = [1, 0, 0]

        # False Negatives (blue)
        fn_mask = (pred == 0) & (gt == 1)
        overlay[fn_mask] = [0, 0, 1]

        axes[row, 3].imshow(overlay)

        # Compute F1
        tp = tp_mask.sum()
        fp = fp_mask.sum()
        fn = fn_mask.sum()
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)

        axes[row, 3].set_title(f"Overlay (F1={f1:.3f})\nTP/FP/FN: {tp}/{fp}/{fn}")
        axes[row, 3].axis('off')

        # Add legend for first row
        if row == 0:
            legend_elements = [
                patches.Patch(facecolor='green', label='TP (correct forgery)'),
                patches.Patch(facecolor='red', label='FP (false alarm)'),
                patches.Patch(facecolor='blue', label='FN (missed forgery)')
            ]
            axes[row, 3].legend(handles=legend_elements, loc='upper right',
                              bbox_to_anchor=(1.0, -0.1), fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to {output_path}")


def main(args):
    print(f"Loading validation predictions from {args.val_preds}...")

    data = np.load(args.val_preds)
    prob_masks = data['prob_masks']
    gt_masks = data['gt_masks']
    case_ids = data.get('case_ids', None)

    print(f"Loaded {len(prob_masks)} validation samples")
    print(f"Creating visualization with {args.n_samples} samples...")

    visualize_samples(
        prob_masks=prob_masks,
        gt_masks=gt_masks,
        case_ids=case_ids,
        n_samples=args.n_samples,
        output_path=args.output
    )

    # Compute overall stats
    print("\n=== Overall Validation Stats ===")

    for threshold in [0.3, 0.4, 0.5, 0.6]:
        f1_scores = []
        for i in range(len(prob_masks)):
            pred = (prob_masks[i] >= threshold).astype(np.uint8)
            gt = gt_masks[i]

            tp = (pred * gt).sum()
            fp = (pred * (1 - gt)).sum()
            fn = ((1 - pred) * gt).sum()

            precision = tp / (tp + fp + 1e-7)
            recall = tp / (tp + fn + 1e-7)
            f1 = 2 * precision * recall / (precision + recall + 1e-7)
            f1_scores.append(f1)

        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        print(f"Threshold {threshold:.1f}: F1 = {mean_f1:.4f} ± {std_f1:.4f}")

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize validation predictions')
    parser.add_argument('--val_preds', type=str, required=True,
                       help='Path to validation predictions (.npz)')
    parser.add_argument('--output', type=str, default='sanity_check.png',
                       help='Output visualization path')
    parser.add_argument('--n_samples', type=int, default=8,
                       help='Number of samples to visualize')

    args = parser.parse_args()
    sys.exit(main(args))
