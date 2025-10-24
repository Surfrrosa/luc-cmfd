# Technical Lessons Learned - 2025-10-24

## Session Summary
First Kaggle training attempt. Hit 7 major issues before successful training started.

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

## 📊 Impact Summary

| Issue | Time Lost | Severity | Prevention |
|-------|-----------|----------|------------|
| DINOv2 reshape | 2 hours | Critical | Read API docs carefully |
| AMP overflow | 30 mins | High | Use float16-safe values |
| Padding alignment | 45 mins | High | Calculate LCM properly |
| OOM unfold | 20 mins | Medium | Test memory first |
| YAML types | 15 mins | Low | Explicit type conversion |
| Import caching | 1 hour | Medium | Restart kernel after git pull |
| Nested repo | 45 mins | Medium | Check paths before clone |

**Total:** ~5.5 hours of debugging

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
1. [ ] Estimate GPU time needed
2. [ ] Check quota remaining
3. [ ] Test with `epochs=1` first

---

## 📚 References

- DINOv2 API: https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py#L321
- PyTorch AMP: https://pytorch.org/docs/stable/amp.html
- Float16 range: https://en.wikipedia.org/wiki/Half-precision_floating-point_format

---

**Total issues resolved:** 7
**Training status:** ✅ Successfully started
**Next steps:** Monitor F1 score, run inference after training completes
