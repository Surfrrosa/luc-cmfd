# Pre-Ablation Checklist - Implementation Status

Last updated: Oct 24, 2025

## Implemented ✓

### 1. Submission Correctness
- [x] **validate_submission.py** created (`scripts/validate_submission.py`)
  - Validates CSV schema (case_id, annotation)
  - Checks "authentic" literal for empty masks
  - Validates 1-indexed RLE format
  - Round-trip tests on synthetic masks
  - Usage: `python scripts/validate_submission.py submission.csv`

### 2. Determinism & Environment Guards
- [x] **set_all_seeds()** updated with thread limiting
  - Sets `torch.set_num_threads(1)` and `torch.set_num_interop_threads(1)`
  - Logs seed value
- [x] **print_env_banner()** added (`src/utils.py`)
  - Prints torch/cv2/skimage versions
  - Shows GPU name and AMP status
  - One-line banner format

### 3. Robust Fallbacks
- [x] **Keypoint proposals** - Graceful fallback if SIFT unavailable
  - Falls back to ORB if SIFT not found
  - Returns None if < 8 matches
- [x] **PatchMatch** - Returns empty arrays if fails
- [x] **Inference pipeline** - Catches exceptions, always produces mask
  - Try-except blocks around KP and PM
  - Falls back to decoder-only if geometric verification fails

### 4. Configuration & Safety
- [x] **configs/safe.yaml** created
  - All expensive features OFF
  - Only component pruning enabled
  - Target <2h runtime
- [x] **configs/ablation.yaml** updated with safety limits
  - `safety.max_runtime_hours: 4.0`
  - `safety.require_force_for_combined: true`
- [x] **Feature defaults** - All toggles OFF except component_pruning

### 5. Integrated Features
- [x] Deep PatchMatch module
- [x] Strip Pooling in decoder
- [x] Component-aware pruning with periodicity detection
- [x] Keypoint proposals (SIFT/ORB)
- [x] Updated inference pipeline with all integrations

---

## Partially Implemented / Needs Enhancement

### 6. Runtime Budget & Timing
- [ ] Dry-run timer in ablation script
  - Placeholder exists in `scripts/run_ablate.py`
  - Needs refinement to print p50/p95 times
- [ ] Feature pyramid caching within image
  - Not implemented yet
- [ ] Auto-abort if projected >4h

### 7. Metrics & Ablation Rigor
- [ ] Local scorer matching competition F1
  - Basic F1 implemented, needs validation against competition metric
- [ ] Updated `run_ablate.py` with full logging
  - Needs git hash, config hash, runtime stats
  - CSV logging structure exists but needs enhancement
- [x] 150-image dev subset JSON list
  - Created at data/dev_subset.json (75 authentic, 75 forged)

### 8. Mask Logic
- [x] Component separation (already in post.py)
- [x] Component scoring with α/β weights
- [x] Size-aware pruning
- [ ] Gel-band heuristic logging in debug mode
  - Function exists but not integrated with logging

### 9. Dataset Loader
- [x] `.npy` mask handling with mmap
- [x] Multi-channel mask merging (max strategy)
- [x] NEAREST resize for mismatched sizes
- [ ] `scripts/peek_samples.py` for visual check
  - Not created yet

---

## Not Yet Implemented

### 10. Notebook Ergonomics
- [ ] Update `kaggle_train.ipynb` with:
  - Mount path checks
  - Weights existence check
  - Config/seed/env banner
  - Single Run cell
  - validate_submission.py call

### 11. Numeric Stability & AMP
- [ ] Explicit float32 for RANSAC (currently implicit)
- [ ] Correlation clamping to [-1,1]
- [x] Probability clamping to [0,1] after fusion
- [ ] Rot90 TTA with proper alignment verification

### 12. Logging & Debugging
- [ ] `--debug` flag for rich per-image logs
- [ ] `logs/last_run.json` generation
- [ ] PM/KP match counts logging
- [ ] Component keep/drop logging

### 13. Extra Unit Tests
- [ ] `tests/test_infer_paths.py` - Fallback testing
- [ ] `tests/test_tta.py` - Rot90 round-trip
- [ ] `tests/test_submission.py` - RLE validation

### 14. Miscellaneous
- [ ] `scripts/peek_samples.py` - Visual dataset checker
- [ ] Git hash extraction in ablation runner
- [ ] Full ablation CSV logging with all fields
- [ ] Compare_results.py script

---

## Critical Path to Green Light

### Must Have (Blocking)
1. **validate_submission.py passes on test masks** ✓ DONE
2. **Safe config exists** ✓ DONE
3. **Basic ablation logging works** - NEEDS TESTING

### Should Have (Important)
4. Create 150-image dev subset JSON
5. Run dry-run timing on dev subset
6. Update notebook with safety checks
7. Add probability clamping after fusion

### Nice to Have (Polish)
8. Debug logging
9. Unit tests for fallback paths
10. Visual dataset checker

---

## Immediate Next Steps (Priority Order)

1. ~~**Test validate_submission.py** on synthetic data~~ DONE
   - Passes all validation checks

2. ~~**Create 150-image dev subset**~~ DONE
   - Created at data/dev_subset.json (75 authentic, 75 forged)

3. ~~**Add probability clamping to infer.py**~~ DONE
   - Added at line 216: `prob = np.clip(prob, 0.0, 1.0)`

4. **Update kaggle_train.ipynb**
   - Add environment checks
   - Add validate_submission call

5. **Run training on Kaggle GPU**
   - Upload notebook and configs
   - Train model with ACCURACY config
   - Generate first submission

---

## Acceptance Checklist

**Green Light Criteria:**

- [x] validate_submission.py exists and has round-trip tests
- [x] validate_submission.py passes on synthetic masks
- [x] SAFE config exists with <2h target
- [ ] Dry-run on 150-image subset prints p50/p95 ms/img
- [ ] Dry-run projects <4h for full set with SAFE config
- [ ] At least one A/B log shows clean ΔF1 and Δruntime
- [ ] Notebook writes valid submission.csv from smoke set

---

## Quick Commands

```bash
# 1. Validate a submission
python scripts/validate_submission.py submission.csv

# 2. Run with SAFE config (when ready)
python src/infer.py \
  --weights weights/best_model.pth \
  --image_dir data/test_images \
  --config configs/safe.yaml \
  --output submission.csv

# 3. Validate output
python scripts/validate_submission.py submission.csv
```

---

## Notes

- Most core functionality is implemented and ready
- Main gaps are in testing infrastructure and polish
- Can proceed with manual testing using existing tools
- Automated ablation runner needs one more iteration but can be done manually for now
