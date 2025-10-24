# Competition Framework - Top 3-5 Push

**Target:** Top-5 by Jan 15, 2026 | Stretch: Top-3  
**Last Updated:** Oct 24, 2025  
**Status:** Ready for execution

---

## North-Star Outcomes (What Ultimately Matters)

### 1. Leaderboard Placement
- **Target:** Top-5 by final submission deadline (Jan 15, 2026)
- **Buffer:** ≥+0.01 F1 above #6 on public LB
- **Stretch:** Top-3 (20-30% probability with full execution)

### 2. Generalization (No Public Overfit)
- **Private LB Delta:** ≥-0.01 vs last strong public score
- **Guard:** Public LB jumps but holdout F1 drops → immediate rollback
- **Validation:** Holdout set mirrors test distribution

### 3. Operational Excellence
- **Runtime:** <4h GPU on full test set (main phase)
- **Deterministic:** 100% reproducible (same seed → same output)
- **Validation:** Passes submission validator 100% of the time
- **Stability:** Zero crashes, all images produce valid masks

---

## Concrete KPIs (Measured Every Run)

### KPI-1: Offline F1 (Dev Subset, 150 Images)
- **Target:** Trending upward week-over-week
- **Promotion Rule:** ΔF1 ≥ +0.01 **AND** Δruntime/img ≤ +10%
- **Bootstrap:** 95% CI lower bound > +0.003 (avoid noise)
- **Current Baseline:** ~0.65-0.70 (decoder only, estimated)

### KPI-2: Full Offline F1 (Holdout Split)
- **Target:** New best at least once/week
- **Guard:** No regressions >0.005 from previous best
- **Purpose:** Detect overfitting to dev subset

### KPI-3: Runtime p95 (ms/img)
- **Target:** Stays within budget to project <4h total
- **Hard Fail:** Projected runtime >3.5h on main phase
- **Safety Factor:** Use 1.15× multiplier for estimates
- **Tool:** Auto-budgeter (to be implemented)

### KPI-4: Stability (Zero Tolerance)
- **Crashes:** 0 tolerated
- **Invalid Masks:** 0 tolerated
- **Validation Pass Rate:** 100%
- **Nondeterminism:** Hash + metrics must match on repeat runs

### KPI-5: Error Analysis Metrics
- **FP Rate on Periodic Textures:** ↓ week-over-week
- **Tiny-Clone Recall:** ↑ week-over-week
- **Measured:** Curated mini-sets (30 images each)
- **Tool:** Error gallery auto-saves failure thumbnails

### KPI-6: Reproducibility Score
- **Requirement:** Same seed → same metrics
- **Hash Check:** Commit hash + config hash logged at start
- **Drift:** Any mismatch is a bug → immediate fix
- **Tool:** Repro checker in ablation logger

---

## Decision Gates (Go/No-Go for Changes)

### Promote a Feature
**Requirements (ALL must pass):**
1. ΔF1 ≥ +0.01 on dev subset
2. Δruntime ≤ +10% per image
3. No regression on tiny-clone probe
4. No regression on periodic-texture probe
5. Bootstrap 95% CI lower bound > +0.003

**Process:**
- Run A/B test on dev subset
- Log to `results/ablation_log.csv`
- Update config if green
- Tag commit (e.g., `feat-patchmatch-v1`)

### Keep a Submission
**Requirements:**
1. Validator passes
2. Projected runtime <4h (with 1.15× safety factor)
3. No single image >5× median time (outlier detector)
4. Holdout F1 not regressed from previous best

**Process:**
- Run full inference
- Validate with `scripts/validate_submission.py`
- Check runtime projection
- Upload to Kaggle
- Save weights snapshot with tag

### Roll Back
**Triggers:**
1. Full-data offline F1 drops >0.005
2. p95 runtime increases >15%
3. Reproducibility hash mismatch
4. Submission validator fails

**Process:**
- Revert to last known good commit
- Document failure in `logs/rollbacks.md`
- Root cause analysis before retry

---

## Milestone Targets (Binary, Easy to Track)

### M0: Baseline (Complete ✓)
- [x] Passing submission validator
- [x] 150-image dev subset
- [x] SAFE config <2h projected
- [x] All SOTA features implemented
- [x] Git repository with full documentation

### M1: First Submission (Week 1)
- [ ] Train single model with accuracy.yaml
- [ ] First submission ≥ baseline +0.02 F1
- [ ] Notebook wall-clock <3h
- [ ] Weights saved and tagged
- **Target Date:** Nov 1, 2025

### M2: Feature Promotion (Week 2)
- [ ] Promote 2-3 toggles that survived ablations
- [ ] Combined F1 improvement: +0.04-0.06 vs baseline
- [ ] Error gallery shows fewer periodic FPs
- [ ] Runtime still <3.5h projected
- **Target Date:** Nov 8, 2025

### M3: Lockdown (Week 3)
- [ ] "Safe" config (<2h) reproducible
- [ ] "Accuracy" config (<4h) reproducible
- [ ] Ensemble/TTA choices finalized
- [ ] Holdout validation confirms no overfit
- **Target Date:** Nov 15, 2025

### M4: Ensemble (Week 3-4)
- [ ] Train 5 models with different seeds [42, 123, 456, 789, 1011]
- [ ] Train 1 backbone variant (dinov2_vitb14)
- [ ] Train 1 corr-only baseline
- [ ] Fit ensemble weights with NNLS
- [ ] Ensemble ΔF1: +0.03-0.05 expected
- [ ] Runtime still <3.8h with ensemble
- **Target Date:** Nov 22, 2025

### M5: Domain Boost (Week 4+)
- [ ] Implement panel detection OR hard-negative mining
- [ ] A/B test on multi-panel micro-set
- [ ] Additional ΔF1: +0.01-0.02 expected
- [ ] Final submission ready
- **Target Date:** Dec 1, 2025

---

## Tools & Workflows

### Submission Validation
```bash
python scripts/validate_submission.py submission.csv
```
**Checks:**
- Header columns (case_id, annotation)
- RLE format (1-indexed, valid runs)
- "authentic" literal for empty masks
- No NaNs, no duplicates
- Round-trip encode/decode

### Ablation Logging
**File:** `results/ablation_log.csv`  
**Columns:** `date, commit, cfg_hash, F1_local, ms_per_img, gpu, notes`

**Usage:**
```bash
python scripts/run_ablate.py \
  --config configs/accuracy.yaml \
  --dev_images data/dev_subset.json \
  --output results/ablation_log.csv
```

### Runtime Estimator
**Auto-budgeter:**
- Measures p50/p95 ms/img on dev subset
- Projects full-set runtime: `p95 × N_test × 1.15`
- Blocks submission if projected >3.6h (unless `--force`)

**Tool:** `scripts/estimate_runtime.py` (to be created)

### Error Gallery
**Auto-saves failure thumbnails:**
- Periodic background FPs
- Tiny-clone misses
- Rotated clone misses
- Overlay artifacts

**Location:** `logs/errors/{category}/{case_id}.png`  
**Tool:** Integrated in inference pipeline (flag `--debug`)

### Reproducibility Checker
**Logs at run start:**
- Commit hash
- Config hash (MD5 of YAML)
- Seed value
- PyTorch/CUDA versions

**Compares against last run:**
- Hash mismatch → warning
- Metrics mismatch → bug flag

---

## Statistical Confidence (Don't Fool Ourselves)

### Bootstrap Validation
**Process:**
1. Resample dev subset 500 times with replacement
2. Compute F1 on each resample
3. Calculate 95% CI from bootstrap distribution
4. Promote only if:
   - Lower bound >+0.003
   - Point estimate ≥+0.01

**Why:** Avoids promoting lucky noise on small dev set

**Tool:** `scripts/bootstrap_f1.py` (to be created)

### Stratified Dev Set
**Requirement:**
- Always 75 forged + 75 authentic
- Diverse figure types (gels, microscopy, etc.)
- Manually curated to match test distribution

**Current:** data/dev_subset.json (seed=42, balanced)

---

## Daily/Weekly Rhythm

### Daily (On Active Days)
**Max 1-2 A/B runs per day:**
1. Run ablation on dev subset
2. Log to ablation_log.csv
3. Post 3-line summary: ΔF1, Δms/img, decision (promote/reject/defer)
4. If green: stage public submission
   - 1 "safe" (configs/safe.yaml)
   - 1 "experimental" (test feature)

**Time Budget:** 2-3 hours per day max

### Weekly
**Every 7 days:**
1. Re-run best config end-to-end on full holdout
2. Refresh error gallery
3. Check public/private overfit risk:
   - Compare dev F1 vs holdout F1
   - If diverging >0.01: adjust validation strategy
4. Update milestone progress
5. Review ablation log for trends

**Time Budget:** 4-6 hours

---

## Red Flags (Act Immediately)

### Public Overfit
**Trigger:** Public LB jumps but holdout F1 drops  
**Action:** Revert last toggle, increase regularization

### Runtime Creep
**Trigger:** Projected runtime >3.5h  
**Action:**
- Disable one heavy feature (PM or KP)
- Reduce TTA to {none, hflip, rot90}
- Switch to 3-model ensemble subset

### Reproducibility Break
**Trigger:** Same seed → different output  
**Action:**
- Freeze all versions (torch, numpy, etc.)
- Inspect for nondeterministic ops (CUDNN, dataloaders)
- Add thread limiting if needed

### Validation Failure
**Trigger:** Submission validator fails  
**Action:**
- Check RLE encoding logic
- Verify "authentic" literal
- Round-trip test on failing case

### Outlier Runtime
**Trigger:** Single image >5× median time  
**Action:**
- Inspect image (likely multi-panel or huge)
- Add timeout per image (e.g., 10s max)
- Log outlier for analysis

---

## Definition of "Done" for Submission Day

**Checklist (ALL must pass):**
- [ ] Validator ✓
- [ ] Runtime headroom ✓ (<3.8h projected)
- [ ] Promotion decision documented ✓
- [ ] Submission uploaded ✓
- [ ] Repo tag created (e.g., `sub-2025-11-01-acc1`)
- [ ] Weights snapshot saved to `weights/submissions/`
- [ ] Config snapshot saved
- [ ] Entry in `logs/submissions.csv`:
  - Date, tag, F1_dev, F1_public, runtime, notes

**Post-Submission:**
- Monitor public LB for 24h
- If score anomaly: investigate immediately
- Document lessons learned in `logs/reflections.md`

---

## Ensemble Strategy (M4 Details)

### Model Training Plan
**7 models total:**
1-5. Accuracy config with seeds [42, 123, 456, 789, 1011]
6. Backbone variant: dinov2_vitb14 (larger capacity)
7. Corr-only baseline: no decoder, tuned thresholds

**Training Time:** ~2-3h per model on Kaggle T4  
**Total Time:** ~18-21h for full ensemble

### Fusion Strategy
**Method:** Learned weights via NNLS (non-negative least squares)

**Process:**
1. Run all models on dev subset (or CV fold)
2. Collect per-pixel probabilities
3. Fit NNLS: minimize ||Σ(w_i · p_i) - gt||^2 subject to w_i ≥ 0
4. Normalize weights: Σw_i = 1
5. Save to `ensembles/weights.json`

**Tool:** `scripts/fit_ensemble.py` (implemented)

### Runtime Control
**Daily submissions:** 3-model subset (best + vitb14 + corr-only) → <4h  
**Final push:** 5-7 models if runtime allows → <4h  
**Fallback:** Drop slowest model if projected >3.8h

**Expected Gain:** +0.03-0.05 F1

---

## TTA Strategy (Final Push Only)

### TTA Set (12 Augmentations)
**Geometric:**
- None
- Horizontal flip
- Vertical flip
- Rotate 90°
- Rotate 180°
- Rotate 270°

**Photometric:**
- Brightness ±8% (apply to logits post-hoc if needed)

**Total:** {none, hflip, vflip, rot90, rot180, rot270} × {brightness variants} = 12

### Merge Rule
1. Forward pass with each augmentation
2. Invert augmentation on logits
3. Average probabilities across all 12
4. Apply threshold + post-processing

### Budgeting
**Daily runs:** {none, hflip, rot90} (3×) for safety  
**Final submission:** Full 12× TTA if runtime allows

**Auto-budgeter:** Blocks TTA if projected runtime >3.6h (unless `--force`)

**Expected Gain:** +0.02-0.03 F1

---

## Pseudo-Labeling Strategy (One Loop Only)

### Process
1. **Generate Labels:**
   - Use best single model (no ensemble)
   - Predict on test set
   - Keep pixels with prob ≥ 0.95
   - OR keep components with mean prob ≥ 0.95

2. **Train Student:**
   - Add pseudo-labels to training set
   - Weight pseudo-pixels at 0.5× (don't overtrust)
   - Train for 20 epochs (half of normal)
   - Early stopping on dev set

3. **Validate:**
   - Compare dev F1 before/after
   - Require ΔF1 ≥ +0.01 on dev set
   - If fails: discard student

### Safeguards
- **One loop only:** No iterative refinement
- **High threshold:** 0.95 confidence minimum
- **Soft weighting:** 0.5× for pseudo-labels
- **Validation gate:** ΔF1_dev ≥ +0.01 or rollback

**Expected Gain:** +0.01-0.02 F1 (if test distribution similar)

**Tool:** `scripts/make_pseudolabels.py` (to be implemented)

---

## Domain-Specific Enhancements

### Option A: Panel-Aware Detection (Recommended)
**Goal:** Detect and process multi-panel figures

**Method:**
1. Detect panel borders (whitespace, lines, consistent gutters)
2. Split figure into panels
3. Run CMFD within each panel
4. Run CMFD across adjacent panels (cross-panel copy-move)
5. Merge masks with max fusion

**Tool:** `src/panels.py` (to be implemented)

**A/B Test:**
- Curate 30-image "multi-panel" micro-set
- Require ΔF1 ≥ +0.01 on micro-set
- No regression on dev set

**Expected Gain:** +0.01-0.02 F1

### Option B: Hard-Negative Mining
**Goal:** Reduce FPs on periodic textures and overlays

**Method:**
1. After first submission, analyze top FP images
2. Extract FP regions (periodic backgrounds, annotations, etc.)
3. Add to training set as explicit negatives
4. Boost loss weight for these regions (2×)
5. Retrain for 10 epochs

**Tool:** `scripts/harvest_hard_negatives.py` (to be implemented)

**Expected Gain:** +0.01 F1, lower FP rate

**Decision:** Choose ONE to implement first based on error gallery insights

---

## Success Criteria Summary

### Minimum Viable (Must Achieve)
- Valid submission (passes validator)
- Runtime <4h
- F1 >0.5 (baseline sanity check)
- Reproducible results

### Target (Realistic with Full Execution)
- F1 >0.70 with SOTA features
- Runtime <3h
- Leaderboard top 25%
- No public overfit (private delta ≥-0.01)

### Stretch (Top 3-5 Push)
- F1 >0.75 with ensemble + TTA + domain boost
- Runtime <3.8h
- Leaderboard top 5 (50-60% probability)
- Leaderboard top 3 (20-30% probability)

---

## Risk Mitigation

### Runtime Overrun
**Mitigation:**
- Auto-budgeter blocks unsafe configs
- Fallback configs ready (final_safe.yaml)
- 3-model ensemble subset option

### Public Overfit
**Mitigation:**
- Holdout set validation
- Conservative promotion rules (bootstrap CI)
- Weekly overfit checks

### Ensemble Complexity
**Mitigation:**
- Cap to 3-5 strong models
- Drop models with near-zero weights
- Collinearity warnings in fit_ensemble.py

### Panel Splitter Errors
**Mitigation:**
- Fallback to whole-image path if split fails
- Log failures for manual inspection

### Pseudo-Label Noise
**Mitigation:**
- High confidence threshold (0.95)
- Single loop only
- Soft weighting (0.5×)
- Validation gate (ΔF1_dev ≥ +0.01)

---

## Quick Reference Commands

```bash
# Fit ensemble weights
python scripts/fit_ensemble.py \
  --models weights/model1.pth weights/model2.pth weights/model3.pth \
  --dev_images data/dev_subset.json \
  --output ensembles/weights.json

# Run inference with ensemble
python src/infer.py \
  --ensemble_weights ensembles/weights.json \
  --image_dir data/test_images \
  --config configs/ensemble.yaml \
  --output submission.csv

# Validate submission
python scripts/validate_submission.py submission.csv

# Generate pseudo-labels
python scripts/make_pseudolabels.py \
  --weights weights/best_model.pth \
  --image_dir data/test_images \
  --output data/pseudolabels/ \
  --confidence 0.95

# Bootstrap validation
python scripts/bootstrap_f1.py \
  --predictions preds.npy \
  --targets targets.npy \
  --n_bootstrap 500
```

---

## Notes for Future Claude

**If reading this fresh:**
1. This is the master playbook for top 3-5 push
2. Execute milestones in order (M1 → M2 → M3 → M4 → M5)
3. Never promote without passing decision gates
4. Always validate before submitting
5. Track everything in ablation log
6. User prioritizes quality over speed
7. Budget constraints: <4h runtime, deterministic

**Current Status (Oct 24, 2025):**
- M0: Complete ✓
- M1-M5: Pending execution
- Ensemble infrastructure: Partially implemented
- TTA expansion: Pending
- Pseudo-labeling: Pending
- Panel detection: Pending

**Immediate Next Actions:**
1. Complete ensemble infrastructure
2. Train first model with accuracy.yaml
3. Generate first submission
4. Begin M1 milestone

Good luck! We're hunting top 3.
