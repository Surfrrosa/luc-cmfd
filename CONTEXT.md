# Project Context & History

**Last Updated:** Oct 24, 2025  
**Competition:** Recod.ai/LUC - Scientific Image Forgery Detection (Kaggle)  
**Status:** Pre-training, ready for Kaggle GPU run  

---

## Competition Details

**URL:** https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection

**Task:** Detect and segment copy-move forgeries in biomedical images (gel electrophoresis, microscopy, etc.)

**Constraints:**
- Max 4 hours GPU runtime on full test set
- No internet access during inference
- Submission format: CSV with case_id + RLE-encoded masks OR "authentic"

**Dataset:**
- Training: 5,128 images (2,377 authentic, 2,751 forged)
- Masks: .npy files (H,W) or (2,H,W) or (3,H,W) for multi-region forgeries
- Located at: `data/recodai-luc-scientific-image-forgery-detection/`

**Metric:** F1 Score (primary), runtime (secondary constraint)

---

## User Preferences & Goals

**Explicit User Directives:**
1. "Accuracy is more important than speed right now"
2. "Quality is imperative"
3. "No emojis unless necessary" (documentation style)
4. Working as a team for competition ("we're entering a kaggle competition together!")

**Priority:** Maximum F1 score while staying under 4h runtime budget

---

## Current Architecture

### Core Pipeline
1. **DINOv2 ViT-S** backbone (frozen) for feature extraction
2. **Self-correlation** matching to find duplicated regions
3. **RANSAC** geometric verification (similarity/affine transforms)
4. **TinyDecoder** U-Net for mask refinement
5. **Multi-prediction fusion** (decoder + keypoints + patchmatch)
6. **Post-processing** with component-aware pruning

### SOTA Features (All Config-Toggleable)

**Deep Learning:**
- Strip Pooling: Horizontal/vertical global context (CMFDFormer-inspired)
- Deep PatchMatch: Multi-scale correspondence discovery
- Keypoint Proposals: SIFT/ORB for rotation/scale invariance

**Post-Processing:**
- Component-Aware Pruning: Multi-factor scoring (correlation + inlier ratio)
- Periodicity Detection: FFT-based periodic texture filtering
- Gel Band Prior: Domain-specific aspect ratio heuristics

**Expected F1 Improvements:**
- Baseline (decoder only): ~0.65-0.70
- Strip Pooling: +0.01-0.02
- PatchMatch: +0.02-0.04
- Keypoints: +0.01-0.02
- Component Pruning: +0.01-0.03
- Combined (accuracy.yaml): ~0.70-0.78 estimated

---

## Configuration Files

### `configs/safe.yaml` - Production Safety
- **Purpose:** Guaranteed <2h runtime
- **Features:** Only component pruning enabled, all expensive features OFF
- **Use:** When unsure about timing or need guaranteed completion

### `configs/accuracy.yaml` - Maximum F1 (RECOMMENDED)
- **Purpose:** Best possible F1 score, runtime secondary
- **Features:** Strip pooling, PatchMatch, keypoints, component pruning, periodicity, gel priors ALL enabled
- **Target:** <3.5h runtime
- **Use:** Competition submission (current priority)

### `configs/ablation.yaml` - A/B Testing
- **Purpose:** Systematic feature evaluation
- **Features:** All OFF by default, safety limits enforced
- **Use:** Testing individual feature contributions

**Important:** Old configs (fast.yaml, tiny_clones.yaml) are deprecated. Use the three above.

---

## Implementation Status

### Completed ✓
- All core modules (model, corr, geom, post, rle, dataset, infer)
- All SOTA feature modules (pm.py, kp.py, strip pooling)
- Graceful fallbacks for all optional features
- Submission validation with round-trip tests
- Three config presets
- 150-image dev subset (data/dev_subset.json)
- Probability clamping for numerical stability
- Deterministic seeding with thread limiting
- Environment banner for debugging
- Git repository with comprehensive documentation

### Not Yet Done
- Model training (no weights yet)
- Kaggle notebook enhancements (validation call, env checks)
- Actual competition submission
- A/B testing of features
- Performance validation on dev subset

---

## Critical Decisions & Rationale

### Why DINOv2?
- SOTA self-supervised features
- Works well on out-of-domain data (biomedical)
- Frozen backbone saves memory/time

### Why Multi-Prediction Fusion?
- Combines decoder (learns from data) + keypoints (handles rotation) + PatchMatch (robust matching)
- Max fusion allows each method to contribute where it's strongest
- Probability clamping prevents numerical issues

### Why Component-Aware Pruning?
- Reduces false positives from textures
- Multi-factor scoring more robust than simple thresholding
- Size-adaptive handles both large and small forgeries

### Why Config Toggles?
- Allows quick A/B testing without code changes
- Safe fallback if runtime too tight
- Incremental feature enablement based on results

---

## Data Structure Details

**Location:** `data/recodai-luc-scientific-image-forgery-detection/`

**Structure:**
```
train_images/
  authentic/  (2,377 images)
  forged/     (2,751 images)
train_masks/  (2,751 .npy files, one per forged image)
test_images/  (images for final submission)
```

**Mask Handling:**
- Authentic images: No mask file
- Forged images: .npy mask (H,W), (2,H,W), or (3,H,W)
- Multi-channel: Indicates multiple forged regions
- Our dataset.py merges channels with max() for union

**Dev Subset:**
- Located: `data/dev_subset.json`
- Size: 150 images (75 authentic, 75 forged)
- Seeded: random seed 42 for reproducibility
- Purpose: Quick ablation testing without full dataset

---

## Known Issues & Gotchas

### Import Structure
- Use `from corr import self_corr` NOT `from .corr import self_corr`
- Relative imports fail when using sys.path.insert(0, 'src')
- Fixed in src/model.py line 366

### PyTorch Version Compatibility
- ReduceLROnPlateau `verbose` parameter removed in PyTorch 2.9
- Remove verbose=True if using latest PyTorch
- Fixed in train_kaggle.py

### Variable Image Sizes
- Competition images are different dimensions
- Must resize to fixed size before batching
- Handled by resize_transform() in dataset/training scripts

### Disk Space
- Competition data is ~2.5GB
- Don't copy data unnecessarily
- .gitignore excludes data/weights to save space

### Multi-Channel Masks
- Some masks are (2,H,W) or (3,H,W) for multiple forgery regions
- Use mask.max(axis=0) to get union
- Handled in dataset.py line ~80

---

## Training Status

**Current:** No trained model yet (weights/ directory empty)

**Next Steps:**
1. Upload kaggle_train.ipynb to Kaggle
2. Attach competition dataset
3. Train with configs/accuracy.yaml
4. Save weights
5. Run inference
6. Validate submission with scripts/validate_submission.py
7. Submit to competition

**Training Command (local testing):**
```bash
python train_kaggle.py \
  --batch_size 16 \
  --epochs 50 \
  --output_dir weights \
  --config configs/accuracy.yaml
```

**Expected Training Time:**
- Local CPU: Very slow, not recommended
- Kaggle GPU (T4): ~2-3 hours for 50 epochs

---

## Validation Workflow

**Before Submission:**
1. Train model → weights/best_model.pth
2. Run inference → submission.csv
3. Validate: `python scripts/validate_submission.py submission.csv`
4. Check: All rows valid, "authentic" literal correct, RLE 1-indexed
5. If validation passes → submit to Kaggle

**Critical:** Always validate before submitting. Invalid RLE will score 0.

---

## Quick Reference Commands

```bash
# Validate submission format
python scripts/validate_submission.py submission.csv

# Train with accuracy config
python train_kaggle.py --config configs/accuracy.yaml

# Run inference with SAFE config (guaranteed <2h)
python src/infer.py \
  --weights weights/best_model.pth \
  --image_dir data/test_images \
  --config configs/safe.yaml \
  --output submission.csv

# Run inference with ACCURACY config (max F1)
python src/infer.py \
  --weights weights/best_model.pth \
  --image_dir data/test_images \
  --config configs/accuracy.yaml \
  --output submission.csv

# Git workflow
git add .
git commit -m "Description"
git push origin main
```

---

## Key Files to Read First

If starting fresh, read these in order:

1. **CONTEXT.md** (this file) - Overall context
2. **QUALITY_STATUS.md** - What's implemented and why
3. **CHECKLIST.md** - Detailed implementation status
4. **configs/accuracy.yaml** - Current recommended config
5. **src/infer.py** - Full inference pipeline
6. **src/model.py** - Architecture details

---

## User Communication Style

- No emojis unless explicitly requested
- Technical and concise
- Focus on facts over enthusiasm
- Document decisions with rationale
- Quality over speed

---

## Competition Strategy

**Phase 1: Baseline** (Current)
- Get working submission with decoder-only
- Validate F1 score on dev subset
- Ensure runtime <4h

**Phase 2: SOTA Features** (Recommended)
- Enable all features in configs/accuracy.yaml
- A/B test each feature on dev subset
- Keep features with ΔF1 ≥ +0.01

**Phase 3: Optimization**
- If runtime tight, disable expensive features
- Fallback to configs/safe.yaml if needed
- Tune thresholds for precision/recall balance

**Phase 4: Ensemble** (If Time)
- Train multiple models with different configs
- Ensemble predictions
- Typically +0.02-0.05 F1 improvement

---

## Important Warnings

1. **Never train without validation set** - Will overfit
2. **Always validate submission format** - Invalid RLE scores 0
3. **Watch runtime carefully** - >4h will timeout
4. **Don't push data/weights to GitHub** - Already in .gitignore
5. **Use configs/accuracy.yaml for competition** - Not deprecated configs
6. **Probability clamping is critical** - Already implemented at infer.py:216

---

## Success Criteria

**Minimum Viable:**
- Valid submission (passes validate_submission.py)
- Runtime <4h
- F1 > 0.5 (baseline)

**Target:**
- F1 > 0.70 with SOTA features
- Runtime <3h
- Leaderboard top 25%

**Stretch:**
- F1 > 0.75
- Leaderboard top 10%

---

## Notes for Future Claude

If you're reading this in a new conversation:

1. User prioritizes quality/accuracy over speed
2. All SOTA features are implemented and ready
3. No model training has happened yet - need Kaggle GPU
4. Use configs/accuracy.yaml for competition submission
5. Always validate submission format before submitting
6. No emojis in documentation
7. User is collaborative and detail-oriented
8. We're working as a team to win this competition

**Immediate Next Action:** Train model on Kaggle GPU using kaggle_train.ipynb

Good luck! We've built a solid foundation.
