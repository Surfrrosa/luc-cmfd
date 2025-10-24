"""Training script for CMFD model."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import numpy as np
from tqdm import tqdm

from .model import CMFDNet
from .dataset import CMFDDataset, collate_fn
from .utils import set_all_seeds, setup_logger, Timer, memory_stats


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
    train_image_dir: str,
    train_mask_dir: str,
    val_split: float = 0.2,
    batch_size: int = 8,
    epochs: int = 50,
    lr: float = 1e-4,
    output_dir: str = "weights",
    seed: int = 42
):
    """
    Main training function.

    Args:
        train_image_dir: Training images directory
        train_mask_dir: Training masks directory
        val_split: Validation split ratio
        batch_size: Batch size
        epochs: Number of epochs
        lr: Learning rate
        output_dir: Output directory for weights
        seed: Random seed
    """
    # Setup
    set_all_seeds(seed)
    logger = setup_logger()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"Device: {device}")
    logger.info(f"Seed: {seed}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Create dataset
    logger.info("Loading dataset...")
    full_dataset = CMFDDataset(
        image_dir=train_image_dir,
        mask_dir=train_mask_dir,
        normalize=True
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
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Create model
    logger.info("Creating model...")
    model = CMFDNet(
        backbone='dinov2_vits14',
        freeze_backbone=True,
        patch=12,
        stride=4,
        top_k=5
    )
    model = model.to(device)

    # Enable channels-last for performance
    model = model.to(memory_format=torch.channels_last)

    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    # AMP scaler
    scaler = GradScaler()

    # Training loop
    best_f1 = 0
    patience = 10
    patience_counter = 0

    logger.info("Starting training...")

    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch + 1}/{epochs}")

        # Train
        with Timer("Train epoch", logger):
            train_stats = train_epoch(model, train_loader, optimizer, scaler, device, logger)

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

            save_path = output_path / "best_model.pth"
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved best model (F1: {best_f1:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping after {epoch + 1} epochs")
            break

        # Memory stats
        mem = memory_stats(device)
        logger.info(f"GPU Memory: {mem['allocated_gb']:.2f}GB")

    logger.info(f"\nTraining complete! Best F1: {best_f1:.4f}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_images', type=str, required=True)
    parser.add_argument('--train_masks', type=str, required=True)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output_dir', type=str, default='weights')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    main(
        train_image_dir=args.train_images,
        train_mask_dir=args.train_masks,
        val_split=args.val_split,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        output_dir=args.output_dir,
        seed=args.seed
    )
