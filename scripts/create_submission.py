#!/usr/bin/env python3
"""
End-to-end submission pipeline for Kaggle CMFD competition.

Usage:
    python scripts/create_submission.py \
        --weights /kaggle/working/best_model.pth \
        --config configs/accuracy.yaml \
        --test_dir /kaggle/input/recodai-luc-scientific-image-forgery-detection/test_images \
        --output submission.csv \
        --dry_run  # Optional: test on first 10 images

Features:
    - Loads best checkpoint
    - Runs inference on test set
    - Applies post-processing
    - RLE encodes masks
    - Validates format
    - Writes submission.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import torch
import yaml
import pandas as pd
from tqdm import tqdm
import cv2
import numpy as np

from model import CMFDNet
from dataset import CMFDDataset
from rle import rle_encode
from post import clean_mask, component_aware_pruning, detect_periodicity
from utils import setup_logger


def load_model_and_config(weights_path, config_path, device):
    """Load trained model and config."""

    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Create model
    model_cfg = cfg.get('model', {})
    model = CMFDNet(
        backbone=model_cfg.get('backbone', 'dinov2_vits14'),
        freeze_backbone=True,
        patch=model_cfg.get('patch', 12),
        stride=model_cfg.get('stride', 4),
        top_k=model_cfg.get('top_k', 5),
        use_decoder=model_cfg.get('use_decoder', True),
        use_strip_pool=model_cfg.get('use_strip_pool', False)
    )

    # Load weights
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, cfg


def infer_single_image(model, image_path, config, device):
    """Run inference on a single image."""

    # Load image
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]

    # Convert to tensor
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(img_tensor)
        logits = output['logits']
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()

    # Post-processing
    post_cfg = config.get('post', {})

    # Threshold
    mask = (prob >= post_cfg.get('thr', 0.5)).astype(np.uint8)

    # Morphological operations
    morph_cfg = post_cfg.get('morph', {})
    if morph_cfg.get('close', 0) > 0:
        from skimage.morphology import binary_closing, disk
        mask = binary_closing(mask, disk(morph_cfg['close'])).astype(np.uint8)
    if morph_cfg.get('open', 0) > 0:
        from skimage.morphology import binary_opening, disk
        mask = binary_opening(mask, disk(morph_cfg['open'])).astype(np.uint8)

    # Component pruning
    if post_cfg.get('component_pruning', {}).get('enable', False):
        # Need correlation map for this
        # For now, skip or use basic area filtering
        pass

    # Min area filtering
    min_area = post_cfg.get('min_area', 16)
    from skimage.measure import label, regionprops
    labeled = label(mask)
    for region in regionprops(labeled):
        if region.area < min_area:
            mask[labeled == region.label] = 0

    # Resize to original size if needed
    if mask.shape != (orig_h, orig_w):
        mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    return mask


def validate_submission(df, logger):
    """Validate submission format."""

    errors = []

    # Check columns
    if list(df.columns) != ['case_id', 'annotation']:
        errors.append(f"Wrong columns: {list(df.columns)}. Expected: ['case_id', 'annotation']")

    # Check for duplicates
    if df['case_id'].duplicated().any():
        errors.append(f"Duplicate case_ids found: {df[df['case_id'].duplicated()]['case_id'].tolist()}")

    # Check annotation format
    for idx, row in df.iterrows():
        annotation = row['annotation']
        if annotation != 'authentic':
            # Should be RLE format
            if not isinstance(annotation, str):
                errors.append(f"case_id {row['case_id']}: annotation is not string")
            elif annotation == '':
                errors.append(f"case_id {row['case_id']}: annotation is empty string")

    if errors:
        logger.error("Submission validation FAILED:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    else:
        logger.info("✓ Submission validation PASSED")
        return True


def main(args):
    logger = setup_logger()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"Device: {device}")
    logger.info(f"Weights: {args.weights}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Test dir: {args.test_dir}")
    logger.info(f"Output: {args.output}")

    # Load model
    logger.info("Loading model...")
    model, config = load_model_and_config(args.weights, args.config, device)
    logger.info("✓ Model loaded")

    # Get test images
    test_dir = Path(args.test_dir)
    image_paths = sorted(list(test_dir.glob('*.png')) + list(test_dir.glob('*.jpg')))

    if args.dry_run:
        logger.info(f"DRY RUN: Processing first {args.dry_run_size} images")
        image_paths = image_paths[:args.dry_run_size]

    logger.info(f"Found {len(image_paths)} test images")

    # Run inference
    results = []

    for image_path in tqdm(image_paths, desc="Inference"):
        # Extract case_id from filename
        case_id = image_path.stem

        # Infer
        mask = infer_single_image(model, image_path, config, device)

        # Encode
        if mask.sum() == 0:
            annotation = 'authentic'
        else:
            annotation = rle_encode(mask)

        results.append({
            'case_id': case_id,
            'annotation': annotation
        })

    # Create DataFrame
    df = pd.DataFrame(results)

    # Validate
    if not validate_submission(df, logger):
        logger.error("Submission validation failed. Not writing file.")
        return 1

    # Write
    df.to_csv(args.output, index=False)
    logger.info(f"✓ Submission written to {args.output}")

    # Stats
    n_authentic = (df['annotation'] == 'authentic').sum()
    n_forged = len(df) - n_authentic
    logger.info(f"Stats: {n_authentic} authentic, {n_forged} forged")

    # Sample
    logger.info("\nSample rows:")
    logger.info(df.head(3).to_string())

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create Kaggle submission from trained model')
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to model weights (.pth)')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML')
    parser.add_argument('--test_dir', type=str, required=True,
                       help='Directory containing test images')
    parser.add_argument('--output', type=str, default='submission.csv',
                       help='Output submission CSV path')
    parser.add_argument('--dry_run', action='store_true',
                       help='Test on small subset first')
    parser.add_argument('--dry_run_size', type=int, default=10,
                       help='Number of images for dry run')

    args = parser.parse_args()
    sys.exit(main(args))
