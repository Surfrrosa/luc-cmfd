"""Verify that everything is set up correctly before training."""

import sys
sys.path.insert(0, 'src')

import torch
from dataset import CMFDDataset
from model import CMFDNet

print("=" * 60)
print("LUC-CMFD Setup Verification")
print("=" * 60)

# Check PyTorch and CUDA
print("\n1. PyTorch Setup:")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Check dataset
print("\n2. Dataset:")
try:
    ds = CMFDDataset(root="data/recodai-luc-scientific-image-forgery-detection", split="train")
    print(f"   ✓ Dataset loaded successfully")
    print(f"   Total samples: {len(ds)}")

    # Count authentic vs forged
    n_forged = sum(1 for i in range(len(ds)) if ds.items[i]['is_forged'])
    n_auth = len(ds) - n_forged
    print(f"   Authentic: {n_auth}, Forged: {n_forged}")

    # Test loading a sample
    sample = ds[0]
    print(f"   ✓ Sample loaded: image {sample['image'].shape}, mask {sample['mask'].shape}")
except Exception as e:
    print(f"   ✗ Dataset error: {e}")

# Check model
print("\n3. Model:")
try:
    model = CMFDNet(backbone='dinov2_vits14', freeze_backbone=True, patch=12, stride=4, top_k=5)
    print(f"   ✓ Model created successfully")

    # Test forward pass
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
    print(f"   ✓ Forward pass successful: output shape {output['logits'].shape}")
except Exception as e:
    print(f"   ✗ Model error: {e}")

# Check disk space
print("\n4. Disk Space:")
import shutil
total, used, free = shutil.disk_usage("/")
print(f"   Total: {total / (2**30):.1f} GB")
print(f"   Used: {used / (2**30):.1f} GB")
print(f"   Free: {free / (2**30):.1f} GB")
if free / (2**30) < 5:
    print(f"   ⚠️  WARNING: Low disk space!")

# Summary
print("\n" + "=" * 60)
print("✓ Setup verification complete!")
print("\nReady to train with:")
print("  python train_kaggle.py --batch_size 8 --epochs 50")
print("=" * 60)
