"""Training script for CMFD model."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import numpy as np
from tqdm import tqdm

try:
    from .model import CMFDNet
    from .dataset import CMFDDataset, collate_fn
    from .utils import set_all_seeds, setup_logger, Timer, memory_stats
except ImportError:
    # Fallback for direct execution
    from model import CMFDNet
    from dataset import CMFDDataset, collate_fn
    from utils import set_all_seeds, setup_logger, Timer, memory_stats


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """
    Dice loss for binary segmentation.

    Args:
        pred: Predicted logits (B, 1, H, W)
        target: Ground truth masks (B, 1, H, W)
        smooth: Smoothing factor

    Returns:
        Dice loss value
    """
    pred = torch.sigmoid(pred)

    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()
    dice = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

    return 1.0 - dice


def combined_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Combined BCE + Dice loss.

    Args:
        pred: Predicted logits
        target: Ground truth masks

    Returns:
        Combined loss
    """
    bce = nn.BCEWithLogitsLoss()(pred, target)
    dice = dice_loss(pred, target)

    return bce + dice


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    Compute evaluation metrics.

    Args:
        pred: Predicted logits (B, 1, H, W)
        target: Ground truth masks (B, 1, H, W)

    Returns:
        Dictionary with metrics
    """
    pred_binary = (torch.sigmoid(pred) > 0.5).float()

    tp = (pred_binary * target).sum()
    fp = (pred_binary * (1 - target)).sum()
    fn = ((1 - pred_binary) * target).sum()

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)

    return {
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item()
    }


def train_epoch(
    model: CMFDNet,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    logger
) -> dict:
    """
    Train for one epoch.

    Args:
        model: CMFD model
        dataloader: Training data loader
        optimizer: Optimizer
        scaler: AMP gradient scaler
        device: Device
        logger: Logger

    Returns:
        Dictionary with epoch statistics
    """
    model.train()

    total_loss = 0
    total_f1 = 0
    n_batches = 0

    pbar = tqdm(dataloader, desc="Training")

    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        optimizer.zero_grad()

        # Forward with AMP
        with autocast():
            output = model(images)
            logits = output['logits']

            # Compute loss
            loss = combined_loss(logits, masks)

        # Backward with AMP
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Metrics
        with torch.no_grad():
            metrics = compute_metrics(logits, masks)

        total_loss += loss.item()
        total_f1 += metrics['f1']
        n_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'f1': f"{metrics['f1']:.4f}"
        })

    return {
        'loss': total_loss / n_batches,
        'f1': total_f1 / n_batches
    }


@torch.no_grad()
def validate(
    model: CMFDNet,
    dataloader: DataLoader,
    device: torch.device
) -> dict:
    """
    Validation loop.

    Args:
        model: CMFD model
        dataloader: Validation data loader
        device: Device

    Returns:
        Dictionary with validation statistics
    """
    model.eval()

    total_loss = 0
    total_f1 = 0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Validation"):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        with autocast():
            output = model(images)
            logits = output['logits']

            loss = combined_loss(logits, masks)

        metrics = compute_metrics(logits, masks)

        total_loss += loss.item()
        total_f1 += metrics['f1']
        n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'f1': total_f1 / n_batches
    }


def main(
    data_root: str,
    config_path: str = None,
    weights_backbone: str = None,
    weights_out: str = "best_model.pth",
    epochs: int = 50,
    early_stop: int = 10,
    amp: bool = True,
    val_split: float = 0.2,
    batch_size: int = None,
    lr: float = None,
    seed: int = 42
):
    """
    Main training function with config support.

    Args:
        data_root: Root directory containing train_images/forged, train_images/authentic, train_masks
        config_path: Path to YAML config file
        weights_backbone: Path to pretrained backbone weights
        weights_out: Output path for trained model
        epochs: Number of epochs
        early_stop: Early stopping patience
        amp: Use automatic mixed precision
        val_split: Validation split ratio
        batch_size: Batch size (overrides config if provided)
        lr: Learning rate (overrides config if provided)
        seed: Random seed
    """
    import yaml

    # Load config if provided
    cfg = {}
    if config_path:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

    # Get training params from config or defaults (ensure correct types)
    if batch_size is None:
        batch_size = int(cfg.get('training', {}).get('batch_size', 16))
    if lr is None:
        lr = float(cfg.get('training', {}).get('lr', 1e-4))

    # Ensure types even if provided via command line
    batch_size = int(batch_size)
    lr = float(lr)

    # Setup
    set_all_seeds(seed)
    logger = setup_logger()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"Device: {device}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Config: {config_path if config_path else 'None (using defaults)'}")
    logger.info(f"Batch size: {batch_size}, LR: {lr}, Epochs: {epochs}, Early stop: {early_stop}")

    # Create dataset
    logger.info("Loading dataset...")
    full_dataset = CMFDDataset(
        root=data_root,
        split="train"
    )

    # Train/val split
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    logger.info(f"Train size: {train_size}, Val size: {val_size}")

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Create model with config params
    logger.info("Creating model...")
    model_cfg = cfg.get('model', {})
    model = CMFDNet(
        backbone=model_cfg.get('backbone', 'dinov2_vits14'),
        freeze_backbone=model_cfg.get('freeze_backbone', True),
        patch=model_cfg.get('patch', 12),
        stride=model_cfg.get('stride', 4),
        top_k=model_cfg.get('top_k', 5),
        use_decoder=model_cfg.get('use_decoder', True),
        use_strip_pool=model_cfg.get('use_strip_pool', False)
    )

    logger.info(f"Backbone feature dim: {model.backbone.feat_dim}")

    # Load backbone weights if provided
    if weights_backbone:
        weights_path = Path(weights_backbone)
        if weights_path.exists():
            logger.info(f"Loading backbone weights from {weights_backbone}")
            try:
                state_dict = torch.load(weights_backbone, map_location='cpu')
                # Load only backbone weights
                model.backbone.load_state_dict(state_dict, strict=False)
                logger.info("Backbone weights loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load backbone weights: {e}")
        else:
            logger.warning(f"Backbone weights not found at {weights_backbone}, using random initialization")

    model = model.to(device)

    # Enable channels-last for performance
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)

    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # AMP scaler
    scaler = GradScaler() if amp else None

    # Training loop
    best_f1 = 0
    patience_counter = 0

    logger.info("Starting training...")

    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch + 1}/{epochs}")

        # Train
        with Timer("Train epoch", logger):
            if amp and scaler:
                train_stats = train_epoch(model, train_loader, optimizer, scaler, device, logger)
            else:
                train_stats = train_epoch(model, train_loader, optimizer, GradScaler(), device, logger)

        logger.info(f"Train - Loss: {train_stats['loss']:.4f}, F1: {train_stats['f1']:.4f}")

        # Validate
        with Timer("Validation", logger):
            val_stats = validate(model, val_loader, device)

        logger.info(f"Val - Loss: {val_stats['loss']:.4f}, F1: {val_stats['f1']:.4f}")

        # Scheduler step
        scheduler.step(val_stats['f1'])

        # Save best model
        if val_stats['f1'] > best_f1:
            best_f1 = val_stats['f1']
            patience_counter = 0

            torch.save(model.state_dict(), weights_out)
            logger.info(f"Saved best model to {weights_out} (F1: {best_f1:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= early_stop:
            logger.info(f"Early stopping after {epoch + 1} epochs")
            break

        # Memory stats
        if device.type == 'cuda':
            mem = memory_stats(device)
            logger.info(f"GPU Memory: {mem['allocated_gb']:.2f}GB")

    logger.info(f"\nTraining complete! Best F1: {best_f1:.4f}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train CMFD model with config support')
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing train_images and train_masks')
    parser.add_argument('--weights_backbone', type=str,
                        help='Path to pretrained backbone weights (.pth file)')
    parser.add_argument('--weights_out', type=str, default='best_model.pth',
                        help='Output path for trained model')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--early_stop', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--amp', type=int, default=1,
                        help='Use automatic mixed precision (1=on, 0=off)')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--batch_size', type=int,
                        help='Batch size (overrides config if provided)')
    parser.add_argument('--lr', type=float,
                        help='Learning rate (overrides config if provided)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    main(
        data_root=args.data_root,
        config_path=args.config,
        weights_backbone=args.weights_backbone,
        weights_out=args.weights_out,
        epochs=args.epochs,
        early_stop=args.early_stop,
        amp=bool(args.amp),
        val_split=args.val_split,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed
    )
