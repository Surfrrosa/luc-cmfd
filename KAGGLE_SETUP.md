# Kaggle Training Setup Guide

**Last Updated:** 2025-10-24
**Purpose:** Avoid common pitfalls when training on Kaggle notebooks

---

## 🚨 Critical Issues & Solutions

### 1. **GPU Setup**
```python
# First cell - Always verify GPU
import torch
assert torch.cuda.is_available(), "GPU not enabled!"
print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
```

**Common mistake:** Forgetting to enable GPU in notebook settings (Settings → Accelerator → GPU T4 x2)

---

### 2. **Cloning Repository (AVOID NESTED REPOS!)**

**❌ WRONG - Creates nested luc-cmfd/luc-cmfd/:**
```bash
%cd /kaggle/working/luc-cmfd  # If this exists
!git clone https://github.com/Surfrrosa/luc-cmfd.git  # Creates nested!
```

**✅ CORRECT - Clean clone:**
```python
# Option A: Clone fresh
%cd /kaggle/working
!rm -rf luc-cmfd  # Remove if exists
!git clone https://github.com/Surfrrosa/luc-cmfd.git
%cd luc-cmfd

# Option B: Pull updates (if already cloned)
%cd /kaggle/working/luc-cmfd
!git pull origin main
```

---

### 3. **Module Import Issues (Critical!)**

**Problem:** Python caches old module versions after `git pull`

**✅ SOLUTION - Use this import pattern:**

```python
import sys
import importlib.util
from pathlib import Path

# Detect repo root
ROOT = Path("/kaggle/working/luc-cmfd")
SRC = ROOT / "src"

# Force-load all modules using importlib (bypasses cache)
modules_to_load = ['dataset', 'corr', 'geom', 'post', 'utils', 'model']

for module_name in modules_to_load:
    module_path = SRC / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    print(f"✓ Loaded {module_name}")

# Now you can import normally
CMFDNet = sys.modules['model'].CMFDNet
CMFDDataset = sys.modules['dataset'].CMFDDataset
```

**Alternative:** Restart kernel after `git pull` (slower but reliable)

---

### 4. **Kaggle Command Syntax**

**❌ WRONG - Multiline with backslashes doesn't work:**
```python
!python train.py \
  --config config.yaml \
  --epochs 50
```

**✅ CORRECT - Use subprocess.run:**
```python
import subprocess
subprocess.run([
    'python', 'src/train.py',
    '--config', 'configs/accuracy.yaml',
    '--data_root', '/kaggle/input/recodai-luc-scientific-image-forgery-detection',
    '--epochs', '50'
], capture_output=False)
```

**Alternative:** Single line (harder to read)
```python
!python src/train.py --config configs/accuracy.yaml --data_root /kaggle/input/recodai-luc-scientific-image-forgery-detection --epochs 50
```

---

### 5. **Memory Management (OOM Prevention)**

**Issue:** `unfold()` operation in correlation creates huge tensors

**✅ Safe Batch Sizes:**
- Start with `batch_size=4` (safe)
- Try `batch_size=8` if that works
- **Never exceed batch_size=8** on T4 GPU

**Test before full training:**
```python
# Quick memory test (run before 50 epochs!)
test_img = torch.randn(4, 3, 448, 448).cuda()  # batch_size=4
model = CMFDNet(...).cuda()
output = model(test_img)  # Should not OOM
print(f"✓ Memory test passed with batch_size=4")
```

---

### 6. **Dataset Path Issues**

**✅ Correct paths:**
```python
DATA_ROOT = "/kaggle/input/recodai-luc-scientific-image-forgery-detection"

# Verify paths exist
from pathlib import Path
assert (Path(DATA_ROOT) / "train_images" / "authentic").exists()
assert (Path(DATA_ROOT) / "train_images" / "forged").exists()
assert (Path(DATA_ROOT) / "train_masks").exists()
print("✓ Dataset paths verified")
```

---

### 7. **Test Before Long Training**

**Always run a quick smoke test:**

```python
import torch
from model import CMFDNet

# Test with larger image (448x448 needed for top_k=5)
test_img = torch.randn(2, 3, 448, 448).cuda()
model = CMFDNet(
    backbone='dinov2_vits14',
    freeze_backbone=True,
    patch=12,
    stride=4,
    top_k=5,
    use_decoder=True,
    use_strip_pool=True
).cuda()

output = model(test_img)
print(f"✓ Model works! Output shape: {output['logits'].shape}")
```

**If this fails, DON'T start training!** Debug first.

---

## 📋 Complete Training Workflow

### Cell 1: Setup and Import
```python
import sys
import importlib.util
from pathlib import Path
import torch

# Verify GPU
assert torch.cuda.is_available(), "Enable GPU in settings!"
print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

# Set paths
ROOT = Path("/kaggle/working/luc-cmfd")
SRC = ROOT / "src"
DATA_ROOT = "/kaggle/input/recodai-luc-scientific-image-forgery-detection"

# Force-load modules
modules_to_load = ['dataset', 'corr', 'geom', 'post', 'utils', 'model']
for module_name in modules_to_load:
    module_path = SRC / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    print(f"✓ Loaded {module_name}")

# Store for other cells
import builtins
builtins.ROOT = ROOT
builtins.SRC = SRC
builtins.DATA_ROOT = DATA_ROOT
```

### Cell 2: Quick Test
```python
import torch
from model import CMFDNet

test_img = torch.randn(2, 3, 448, 448).cuda()
model = CMFDNet(
    backbone='dinov2_vits14',
    freeze_backbone=True,
    patch=12, stride=4, top_k=5,
    use_decoder=True, use_strip_pool=True
).cuda()
output = model(test_img)
print(f"✓ Test passed! Shape: {output['logits'].shape}")
```

### Cell 3: Train
```python
import subprocess

result = subprocess.run([
    'python', f'{SRC}/train.py',
    '--config', f'{ROOT}/configs/accuracy.yaml',
    '--data_root', DATA_ROOT,
    '--weights_out', '/kaggle/working/best_model.pth',
    '--epochs', '50',
    '--early_stop', '10',
    '--amp', '1',
    '--batch_size', '4'  # Safe for T4 GPU
], capture_output=False)
```

### Cell 4: Verify Output
```python
from pathlib import Path
model_path = Path('/kaggle/working/best_model.pth')
assert model_path.exists(), "Training failed - no model saved!"
print(f"✓ Model saved: {model_path}")
print(f"  Size: {model_path.stat().st_size / 1e6:.1f} MB")
```

---

## 🐛 Common Errors & Fixes

### Error: "ModuleNotFoundError: No module named 'model'"
**Cause:** Import caching or wrong path
**Fix:** Use importlib pattern above OR restart kernel

### Error: "CUDA out of memory"
**Cause:** Batch size too large
**Fix:** Reduce `--batch_size` to 4 or 2

### Error: "selected index k out of range"
**Cause:** Test image too small for top_k
**Fix:** Use 448x448 or larger test images

### Error: "unexpected indent" or "invalid syntax"
**Cause:** Copy-pasted code with incorrect indentation
**Fix:** Manually type code or use "Raw" paste mode

### Error: Training shows no progress after git pull
**Cause:** Old cached modules still loaded
**Fix:** Restart kernel before re-running

---

## ⏱️ Expected Training Time

- **Batch size 4:** ~40-50 minutes per epoch
- **50 epochs:** ~30-40 hours total
- **With early stopping (10 patience):** Usually stops at 15-25 epochs = 10-20 hours

**GPU Quota:** 30 hours/week. Plan accordingly!

---

## 📊 Monitoring Training

**Good signs:**
- Loss decreasing (1.5 → <1.0)
- F1 score increasing (0.2 → >0.6)
- "Saved best model" messages

**Bad signs:**
- Loss stuck or increasing
- F1 score not improving after 5 epochs
- OOM errors → reduce batch size

---

## 💾 Saving Artifacts

```python
# After training completes, download model
from IPython.display import FileLink
FileLink('/kaggle/working/best_model.pth')
```

Or use Kaggle Output:
1. Check "Add output" on notebook
2. Files in `/kaggle/working/` become downloadable outputs
3. Version and download from notebook output tab

---

## 🔄 Updating Code Mid-Training

**If training is running and you need to update code:**

1. **DON'T** stop training if it's working
2. Push changes to GitHub
3. In a **NEW notebook**, clone and restart
4. Old notebook continues training
5. Use new notebook for next run

**If you MUST update during training:**
1. Interrupt kernel (this stops training!)
2. `!git pull origin main`
3. **Restart kernel** (critical!)
4. Re-run all cells
5. Resume training (will start from epoch 1, not resume)

---

## 📝 Checklist Before Long Training

- [ ] GPU enabled in notebook settings
- [ ] Repository cloned correctly (no nested dirs)
- [ ] Modules imported with importlib pattern
- [ ] Quick smoke test passed (Cell 2)
- [ ] Dataset paths verified
- [ ] Batch size safe (4 for first run)
- [ ] Output path writable (`/kaggle/working/`)
- [ ] GPU quota sufficient (~10-20 hours needed)

---

## 🎯 Quick Reference

| Task | Command |
|------|---------|
| Enable GPU | Settings → Accelerator → GPU T4 x2 |
| Clone repo | `!git clone URL` in `/kaggle/working` |
| Update code | `!git pull origin main` + restart kernel |
| Check GPU | `torch.cuda.is_available()` |
| Safe batch size | 4 (can try 8 if no OOM) |
| Training time | 10-20 hours (early stopping) |

---

**Last updated after resolving all training issues on 2025-10-24**
