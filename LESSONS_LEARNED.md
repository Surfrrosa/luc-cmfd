# Technical Lessons Learned - 2025-10-24/25

## Session Summary
First Kaggle training attempt. Hit 7 major issues before successful training started.
Second training attempt ran out of GPU quota due to x2 GPU configuration.
Third lesson: Model file lost when quota exhausted - learned about Kaggle file persistence.

---

## 🐛 Bug #1: DINOv2 reshape API Misunderstanding

### The Problem
```python
# WRONG - This scrambles dimensions!
out = self.backbone.get_intermediate_layers(x, n=1, reshape=True)[0]
feats = out.permute(0, 3, 1, 2)  # Expected (B,H,W,C) → (B,C,H,W)
```

**Actual behavior:** `reshape=True` already returns `(B, C, H, W)`, NOT `(B, H, W, C)`!

**Symptoms:**
- Channel mismatch errors
- Expected 384 channels, got 16 or 128
- Decoder input shape wrong

### The Fix
```python
# CORRECT - No permute needed!
feats = self.backbone.get_intermediate_layers(x, n=1, reshape=True)[0]
# feats is already (B, C, H, W) where C=384
```

**Root cause:** DINOv2 documentation unclear about reshape parameter output format.

**Files affected:** `src/model.py:91-104`

---

## 🐛 Bug #2: AMP Float16 Overflow

### The Problem
```python
# WRONG - Overflows float16!
corr_matrix = corr_matrix.masked_fill(mask, -1e9)
```

**Error:** `RuntimeError: value cannot be converted to type at::Half without overflow`

**Root cause:**
- Float16 range: ±65,504
- `-1e9 = -1,000,000,000` way exceeds this

### The Fix
```python
# CORRECT - Safe for float16
corr_matrix = corr_matrix.masked_fill(mask, -65500.0)
```

**Why it works:** -65500 is within float16 range and still acts as effective -∞ for masking.

**Files affected:** `src/corr.py:94-96`

---

## 🐛 Bug #3: Padding Alignment for DINOv2

### The Problem
```python
# WRONG - Pad to multiple of 32
Ht = ((Ht + 31) // 32) * 32
```

**Error:** `AssertionError: Input image height not a multiple of patch height 14`

**Root cause:**
- DINOv2 requires multiples of patch_size (14)
- Decoder requires multiples of stride (32)
- 32 is NOT a multiple of 14!

### The Fix
```python
# CORRECT - Pad to LCM(14, 32) = 224
mul = 224
Ht = ((Ht + mul - 1) // mul) * mul
Wt = ((Wt + mul - 1) // mul) * mul
```

**Math:** LCM(14, 32) = 224 satisfies both requirements.

**Files affected:** `src/dataset.py:291-297`

---

## 🐛 Bug #4: OOM from Unfold Operation

### The Problem
**Error:** `CUDA out of memory. Tried to allocate 8.08 GiB`

**Root cause:** `F.unfold()` in correlation creates huge intermediate tensors:
- Input: (B=8, C=384, H=32, W=32)
- Unfold with patch=12: Creates (8, 384*144, 529) = **~8GB!**

### The Fix
```python
# Reduce batch size
--batch_size 4  # Instead of 8
```

**Better solution (future):** Process patches in chunks or use sliding window without unfold.

**Files affected:** Training command

---

## 🐛 Bug #5: YAML Type Coercion Issues

### The Problem
```python
lr = cfg.get('training', {}).get('lr', 1e-4)
# lr = "1e-4" (string!) from YAML
```

**Error:** `TypeError: '<=' not supported between instances of 'float' and 'str'`

### The Fix
```python
# CORRECT - Explicit type conversion
lr = float(cfg.get('training', {}).get('lr', 1e-4))
batch_size = int(cfg.get('training', {}).get('batch_size', 16))
```

**Files affected:** `src/train.py:244-250`

---

## 🐛 Bug #6: Kaggle Import Caching

### The Problem
After `git pull`, Python continues using old cached module versions.

```python
!git pull origin main
from model import CMFDNet  # Still loads OLD model.py!
```

**Symptoms:**
- Bug fixes don't take effect
- Old debug prints still appear
- `ModuleNotFoundError` despite file existing

### The Fix

**Option A:** Restart kernel (slow but reliable)

**Option B:** Force reload with importlib
```python
import importlib.util
spec = importlib.util.spec_from_file_location("model", "src/model.py")
module = importlib.util.module_from_spec(spec)
sys.modules["model"] = module
spec.loader.exec_module(module)
```

---

## 🐛 Bug #7: Nested Git Repository

### The Problem
```bash
cd /kaggle/working/luc-cmfd
git clone https://github.com/user/luc-cmfd.git
# Creates: luc-cmfd/luc-cmfd/src/... (NESTED!)
```

**Symptoms:**
- Imports fail
- "No module named 'model'" despite file existing
- Wrong paths in sys.path

### The Fix
```python
# Auto-detect correct root
candidates = [
    Path("/kaggle/working/luc-cmfd"),
    Path("/kaggle/working/luc-cmfd/luc-cmfd"),
]
for c in candidates:
    if (c/"src/train.py").exists():
        ROOT = c
        break
```

---

## Bug #8: Kaggle GPU x2 Quota Drain

### The Problem
Enabled "GPU T4 x2" (2 GPUs) in Kaggle notebook settings.

**Result:** Used 30 hours of quota in ~12 hours of real time!

**What happened:**
- Training started at ~3-4pm on 2025-10-24
- Ran for ~12 real hours
- Quota showed 00:00 / 30 hrs at ~3am on 2025-10-25
- Training stopped at 85% through epoch 7

### The Root Cause
Kaggle counts GPU x2 as **2x quota usage**:
- 1 real hour with 2 GPUs = 2 quota hours consumed
- 12 real hours × 2 GPUs = 24 quota hours
- Plus ~6 hours from earlier testing = 30 hours used

### The Fix
**ALWAYS use single GPU for training:**
1. In notebook: Session options → Accelerator → **GPU T4** (not x2)
2. Verify setting before starting long runs
3. Check quota remaining: Settings → Account → GPU usage

### Why x2 Doesn't Help Anyway
Our code doesn't use multi-GPU training (no DataParallel or DistributedDataParallel).
- x2 wastes quota without speedup
- Single GPU is sufficient for our model size
- batch_size=4 fits easily on one T4

### Prevention
Add to pre-training checklist:
- [ ] Verify accelerator is "GPU T4" (single, not x2)
- [ ] Check GPU quota remaining (need >15 hours for full run)
- [ ] Use notebook naming to remind: `luc-cmfd-m1-baseline-v2` (no "2gpu" in name)

**Time lost:** Entire week's GPU quota (30 hours)
**Best model saved:** Epoch 4-6, F1 ~0.26-0.28 (lower than expected)

---

## Bug #9: Kaggle File Persistence and Model Loss

### The Problem
Trained for 7 epochs (~12 hours), saw "Saved best model" messages, but when checking `/kaggle/working/best_model.pth` after quota exhaustion - **file was gone**.

**What happened:**
- Training saved `best_model.pth` to `/kaggle/working/` at epochs 4, 5, 6
- GPU quota ran out at 85% of epoch 7
- Kaggle killed the session immediately
- `/kaggle/working/` directory was wiped
- Model file permanently lost

### The Root Cause
**Kaggle's file system has two types of storage:**

1. **Temporary (`/kaggle/working/`):**
   - Fast, but **ephemeral**
   - Cleared when session ends
   - Cleared when quota exhausted
   - **NOT preserved unless you "Save Version"**

2. **Permanent (Kaggle Datasets or Output after "Save & Run All"):**
   - Survives session termination
   - Can be accessed across notebooks
   - Requires explicit action to save

**Our mistake:** We saved to `/kaggle/working/` without:
- Downloading periodically during training
- Copying to a Kaggle Dataset
- Running "Save & Run All" to persist output

### The Fix

**Immediate actions (during training):**
1. Download `best_model.pth` every 5-10 epochs manually
2. Add code to auto-copy to Kaggle Dataset every N epochs
3. Monitor GPU quota and download before it runs out

**Long-term solution:**
Create a Kaggle Dataset for persistent model storage:

```python
# In training notebook, add periodic saves
import shutil
from pathlib import Path

def save_to_dataset(model_path, epoch):
    """Save model to Kaggle Dataset (persistent storage)."""
    dataset_dir = Path('/kaggle/working/model-checkpoints')
    dataset_dir.mkdir(exist_ok=True)

    checkpoint_name = f'best_model_epoch{epoch}.pth'
    dest = dataset_dir / checkpoint_name
    shutil.copy(model_path, dest)
    print(f"✓ Copied to persistent storage: {dest}")

# In training loop, after saving best model:
if val_stats['f1'] > best_f1:
    best_f1 = val_stats['f1']
    torch.save(model.state_dict(), weights_out)
    logger.info(f"Saved best model (F1: {best_f1:.4f})")

    # CRITICAL: Also save to dataset every 5 epochs
    if (epoch + 1) % 5 == 0:
        save_to_dataset(weights_out, epoch + 1)
```

**Post-training:**
Always create a new cell immediately after training and download:
```python
from IPython.display import FileLink
FileLink('/kaggle/working/best_model.pth')
```

### Why This Is Critical

**Time lost:** All 12 hours of training wasted (7 epochs)
**Model performance:** F1 ~0.26-0.28 at best epoch (epoch 4)
**GPU quota:** 30 hours consumed with nothing to show

This is the **most expensive mistake** so far - not in debugging time, but in lost GPU quota and training progress.

### Prevention Checklist

**Before starting training:**
- [ ] Create Kaggle Dataset for model checkpoints (if doing long runs)
- [ ] Add periodic download reminders in notebook

**During training (every 5 epochs):**
- [ ] Download `best_model.pth` manually
- [ ] Check GPU quota remaining
- [ ] If quota < 2 hours, stop training and download immediately

**After training completes:**
- [ ] Download model BEFORE clicking anything else
- [ ] Verify file downloaded successfully (check file size)
- [ ] Upload to GitHub or other permanent storage

**Understanding "Save & Run All" vs "Quick Save":**
- **Quick Save:** Just saves notebook code, NOT files in `/kaggle/working/`
- **Save & Run All:** Runs notebook + saves output (but requires GPU quota!)
- **Save Version:** Saves notebook snapshot + preserves output tab files
- **None of these save `/kaggle/working/` unless you use Output or Datasets!**

**The golden rule:**
> If it's in `/kaggle/working/` and you haven't downloaded it or copied it to a Dataset, it WILL be lost when the session ends.

**Time lost:** 12 hours of training (7 epochs)
**Model lost:** F1 ~0.26-0.28 (all epochs)
**GPU quota wasted:** 30 hours with no model to show for it

---

## Impact Summary

| Issue | Time Lost | Severity | Prevention |
|-------|-----------|----------|------------|
| DINOv2 reshape | 2 hours | Critical | Read API docs carefully |
| AMP overflow | 30 mins | High | Use float16-safe values |
| Padding alignment | 45 mins | High | Calculate LCM properly |
| OOM unfold | 20 mins | Medium | Test memory first |
| YAML types | 15 mins | Low | Explicit type conversion |
| Import caching | 1 hour | Medium | Restart kernel after git pull |
| Nested repo | 45 mins | Medium | Check paths before clone |
| **GPU x2 quota drain** | **30 GPU hours** | **Critical** | **Verify single GPU before training** |
| **Model file loss** | **7 epochs (12 hours)** | **CATASTROPHIC** | **Download periodically + use Datasets** |
| ReduceLROnPlateau verbose | 10 mins | Low | Test locally before Kaggle |

**Total debugging time:** ~6 hours
**Total wasted GPU quota:** 30 hours (1 week)
**Total lost training progress:** 7 epochs with no recoverable model
**Local testing:** ✅ Working on CPU

---

## 🎯 Key Takeaways

### 1. **Always Read API Documentation**
The DINOv2 reshape issue cost us 2 hours. A quick read of the actual source code would have revealed the truth immediately.

### 2. **Test Memory Usage First**
Run a quick smoke test with target batch size before starting long training. OOM at epoch 1 wastes GPU quota.

### 3. **Restart Kernel After Code Updates**
Python's import caching is aggressive in Jupyter. When in doubt, restart.

### 4. **Use Type Conversion Defensively**
YAML loaders are unpredictable. Always use `int()`, `float()`, `bool()` explicitly.

### 5. **Check Numeric Ranges for Mixed Precision**
Float16 has a small range (±65k). Any constant >65k needs adjustment.

### 6. **LCM for Multiple Alignment Requirements**
When you need multiples of both A and B, use LCM(A,B), not just B.

### 7. **Auto-Detect Paths**
Hardcoded paths break easily. Use path detection with fallbacks.

### 8. **ALWAYS Check GPU Settings Before Training**
Kaggle's GPU x2 option drains quota 2x faster without any benefit for our single-model training. Always verify you're using single GPU.

### 9. **Download Models Periodically - Kaggle Files Are Ephemeral**
`/kaggle/working/` is TEMPORARY storage. When quota runs out or session ends, ALL files are lost. Download important files every few epochs, or use Kaggle Datasets for persistent storage.

---

## 🔄 Process Improvements

### Before Starting Training:
1. [ ] Read relevant API docs (don't assume)
2. [ ] Run memory smoke test with target batch size
3. [ ] Verify dataset paths exist
4. [ ] Test model forward pass with realistic input sizes
5. [ ] Check sys.path and module loading

### After Code Changes:
1. [ ] Restart kernel (Kaggle)
2. [ ] Re-run all setup cells
3. [ ] Verify changes took effect (check debug output)

### Before Long Runs:
1. [ ] **VERIFY SINGLE GPU** (not x2!)
2. [ ] Estimate GPU time needed (50 epochs ≈ 20-25 hours)
3. [ ] Check quota remaining (need >20 hours)
4. [ ] Test with `epochs=1` first
5. [ ] Double-check accelerator setting in session options
6. [ ] **Set calendar reminder to download model every 5 epochs**
7. [ ] Create Kaggle Dataset for checkpoints (if run >10 epochs)

### During Training (Every 5 Epochs):
1. [ ] **Download current best_model.pth**
2. [ ] Check GPU quota remaining
3. [ ] If quota <2 hours left: stop, download, continue later

### Immediately After Training:
1. [ ] **Download best_model.pth FIRST** (before anything else!)
2. [ ] Verify file size (should be ~100-500 MB)
3. [ ] Upload to permanent storage (GitHub, Google Drive, etc.)
4. [ ] Only then click "Save Version" or navigate away

---

## 📚 References

- DINOv2 API: https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py#L321
- PyTorch AMP: https://pytorch.org/docs/stable/amp.html
- Float16 range: https://en.wikipedia.org/wiki/Half-precision_floating-point_format

---

## Bug #10: PyTorch ReduceLROnPlateau Verbose Parameter (Local Testing)

### The Problem
**Error:** `TypeError: ReduceLROnPlateau.__init__() got an unexpected keyword argument 'verbose'`

**Root cause:** The `verbose` parameter was removed in PyTorch 2.0+, but older code may still use it.

### The Fix
```python
# WRONG - Fails on PyTorch 2.0+
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5, verbose=True
)

# CORRECT - Parameter removed
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5
)
```

**Files affected:** `train_kaggle.py:210-212`

**Impact:** Low - Only affects local testing, not Kaggle
**Time lost:** 10 minutes

---

**Total issues resolved:** 10
**Training status:** Local testing successful! Ready for Kaggle retry.
**Critical lessons:**
- GPU x2 drains quota 2x faster
- Kaggle files are temporary - download frequently!
- 7 epochs of work lost forever - painful but valuable lesson
- Test locally first to catch compatibility issues

**Next steps:**
- ✅ Local testing works on CPU
- Wait for weekly GPU quota reset (check Settings → Account for date)
- Create new notebook: `luc-cmfd-m1-baseline-v2`
- **BEFORE training:** Set up periodic download strategy
- **DURING training:** Download model every 5 epochs manually
- **Use SINGLE GPU only** (verify before each run)
- Consider adding auto-save to Kaggle Dataset for runs >10 epochs
