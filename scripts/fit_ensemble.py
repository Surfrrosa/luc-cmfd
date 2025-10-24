"""
Ensemble weight fitting for multi-model fusion.

Fits optimal per-pixel probability blending weights using non-negative least squares
or logistic regression to maximize F1 score on validation set.

Usage:
    python scripts/fit_ensemble.py \
        --models weights/model1.pth weights/model2.pth weights/model3.pth \
        --dev_images data/dev_subset.json \
        --output ensembles/weights.json \
        --method nnls
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import torch
from sklearn.linear_model import Ridge, LogisticRegression
from scipy.optimize import nnls
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from model import CMFDNet
from dataset import CMFDDataset
from utils import setup_logger, load_config
from infer import infer_image


def load_models(model_paths: List[str], config: Dict, device: torch.device) -> List[CMFDNet]:
    """Load multiple models from checkpoints."""
    models = []
    logger = setup_logger()
    
    for path in model_paths:
        logger.info(f"Loading model from {path}")
        model = CMFDNet(
            backbone=config['model']['backbone'],
            freeze_backbone=config['model']['freeze_backbone'],
            patch=config['correlation']['patch'],
            stride=config['correlation']['stride'],
            top_k=config['correlation']['top_k'],
            use_decoder=config['model'].get('use_decoder', True),
            use_strip_pool=config['model'].get('use_strip_pool', False)
        )
        
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        models.append(model)
    
    logger.info(f"Loaded {len(models)} models")
    return models


def collect_predictions(
    models: List[CMFDNet],
    dev_images: List[Dict],
    config: Dict,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect per-pixel predictions from all models.
    
    Returns:
        predictions: (N_models, N_pixels) array of probabilities
        targets: (N_pixels,) binary ground truth
    """
    logger = setup_logger()
    
    all_preds = []
    all_targets = []
    
    logger.info(f"Collecting predictions from {len(models)} models on {len(dev_images)} images")
    
    for img_info in tqdm(dev_images, desc="Processing images"):
        # Load image
        img_path = img_info['image']
        label = img_info['label']
        
        # Load ground truth mask if available
        if label == 'forged':
            mask_path = img_path.replace('train_images', 'train_masks').replace('.png', '.npy')
            if Path(mask_path).exists():
                gt_mask = np.load(mask_path, mmap_mode='r')
                if gt_mask.ndim == 3:
                    gt_mask = gt_mask.max(axis=0)  # Multi-channel
                gt_mask = (gt_mask > 0).astype(np.uint8)
            else:
                continue  # Skip if no mask
        else:
            # Authentic - all zeros
            continue  # Skip authentic for fitting (no signal)
        
        # Load image tensor
        from PIL import Image
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        
        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        # Collect predictions from each model
        model_probs = []
        for model in models:
            with torch.no_grad():
                x = img_tensor.unsqueeze(0).to(device)
                output = model.forward(x, return_features=False)
                logits = output['logits']  # (1, 1, H, W)
                prob = torch.sigmoid(logits).squeeze().cpu().numpy()
                
                # Resize to match gt_mask
                if prob.shape != gt_mask.shape:
                    from skimage.transform import resize
                    prob = resize(prob, gt_mask.shape, order=1, preserve_range=True, anti_aliasing=True)
                
                model_probs.append(prob.flatten())
        
        # Stack model predictions
        model_probs = np.stack(model_probs, axis=0)  # (N_models, N_pixels)
        
        all_preds.append(model_probs)
        all_targets.append(gt_mask.flatten())
    
    # Concatenate all images
    predictions = np.concatenate(all_preds, axis=1)  # (N_models, Total_pixels)
    targets = np.concatenate(all_targets)  # (Total_pixels,)
    
    logger.info(f"Collected predictions: {predictions.shape}, targets: {targets.shape}")
    logger.info(f"Positive pixels: {targets.sum()} / {len(targets)} ({targets.mean()*100:.2f}%)")
    
    return predictions, targets


def fit_weights_nnls(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Fit non-negative least squares weights."""
    logger = setup_logger()
    logger.info("Fitting NNLS weights...")
    
    # NNLS: minimize ||Ax - b||^2 subject to x >= 0
    # A: (N_pixels, N_models), b: (N_pixels,)
    A = predictions.T  # (N_pixels, N_models)
    b = targets.astype(np.float32)
    
    weights, residual = nnls(A, b)
    
    # Normalize weights to sum to 1
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        # Fallback: uniform weights
        weights = np.ones(len(weights)) / len(weights)
    
    logger.info(f"NNLS weights: {weights}")
    logger.info(f"NNLS residual: {residual:.6f}")
    
    return weights


def fit_weights_ridge(predictions: np.ndarray, targets: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Fit Ridge regression weights (L2 regularized)."""
    logger = setup_logger()
    logger.info(f"Fitting Ridge weights (alpha={alpha})...")
    
    A = predictions.T
    b = targets.astype(np.float32)
    
    model = Ridge(alpha=alpha, fit_intercept=False, positive=True)
    model.fit(A, b)
    
    weights = model.coef_
    
    # Normalize
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones(len(weights)) / len(weights)
    
    logger.info(f"Ridge weights: {weights}")
    logger.info(f"Ridge score: {model.score(A, b):.6f}")
    
    return weights


def evaluate_ensemble(
    predictions: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    threshold: float = 0.5
) -> Dict:
    """Evaluate ensemble F1 score."""
    # Weighted average
    ensemble_prob = (predictions.T @ weights).flatten()  # (N_pixels,)
    ensemble_pred = (ensemble_prob >= threshold).astype(np.uint8)
    
    # Compute F1
    tp = ((ensemble_pred == 1) & (targets == 1)).sum()
    fp = ((ensemble_pred == 1) & (targets == 0)).sum()
    fn = ((ensemble_pred == 0) & (targets == 1)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn)
    }


def evaluate_individual_models(predictions: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> List[Dict]:
    """Evaluate each model individually."""
    results = []
    for i in range(predictions.shape[0]):
        pred = (predictions[i] >= threshold).astype(np.uint8)
        
        tp = ((pred == 1) & (targets == 1)).sum()
        fp = ((pred == 1) & (targets == 0)).sum()
        fn = ((pred == 0) & (targets == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results.append({
            'model_idx': i,
            'f1': f1,
            'precision': precision,
            'recall': recall
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Fit ensemble weights')
    parser.add_argument('--models', nargs='+', required=True, help='Paths to model checkpoints')
    parser.add_argument('--dev_images', type=str, required=True, help='Dev subset JSON')
    parser.add_argument('--config', type=str, default='configs/accuracy.yaml', help='Config file')
    parser.add_argument('--output', type=str, default='ensembles/weights.json', help='Output weights JSON')
    parser.add_argument('--method', type=str, default='nnls', choices=['nnls', 'ridge'], help='Fitting method')
    parser.add_argument('--alpha', type=float, default=1.0, help='Ridge alpha (if method=ridge)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Probability threshold for F1 evaluation')
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logger()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Load config
    config = load_config(args.config)
    
    # Load dev images
    with open(args.dev_images, 'r') as f:
        dev_images = json.load(f)
    logger.info(f"Loaded {len(dev_images)} dev images")
    
    # Load models
    models = load_models(args.models, config, device)
    
    # Collect predictions
    predictions, targets = collect_predictions(models, dev_images, config, device)
    
    # Fit weights
    if args.method == 'nnls':
        weights = fit_weights_nnls(predictions, targets)
    elif args.method == 'ridge':
        weights = fit_weights_ridge(predictions, targets, alpha=args.alpha)
    else:
        raise ValueError(f"Unknown method: {args.method}")
    
    # Evaluate ensemble
    ensemble_metrics = evaluate_ensemble(predictions, targets, weights, threshold=args.threshold)
    logger.info(f"Ensemble F1: {ensemble_metrics['f1']:.4f}")
    logger.info(f"Ensemble Precision: {ensemble_metrics['precision']:.4f}")
    logger.info(f"Ensemble Recall: {ensemble_metrics['recall']:.4f}")
    
    # Evaluate individual models
    individual_metrics = evaluate_individual_models(predictions, targets, threshold=args.threshold)
    logger.info("\nIndividual model performance:")
    for i, metrics in enumerate(individual_metrics):
        logger.info(f"  Model {i} ({Path(args.models[i]).name}): F1={metrics['f1']:.4f}")
    
    # Check for improvement
    best_single_f1 = max(m['f1'] for m in individual_metrics)
    delta_f1 = ensemble_metrics['f1'] - best_single_f1
    logger.info(f"\nEnsemble gain: ΔF1 = {delta_f1:+.4f}")
    
    # Check for zero weights (collinearity warning)
    zero_weights = (weights < 0.01).sum()
    if zero_weights > 0:
        logger.warning(f"Warning: {zero_weights} models have near-zero weights (potential redundancy)")
    
    # Save weights
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'method': args.method,
        'weights': weights.tolist(),
        'model_paths': args.models,
        'ensemble_f1': ensemble_metrics['f1'],
        'best_single_f1': best_single_f1,
        'delta_f1': delta_f1,
        'individual_f1': [m['f1'] for m in individual_metrics],
        'threshold': args.threshold,
        'timestamp': str(Path(__file__).stat().st_mtime)
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"\nWeights saved to {output_path}")
    logger.info(f"Ensemble ready for inference with --ensemble_weights {output_path}")


if __name__ == '__main__':
    main()
