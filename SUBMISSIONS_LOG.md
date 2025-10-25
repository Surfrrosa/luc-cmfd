# Competition Submissions Log

**Competition:** RECOD.ai/LUC Scientific Image Forgery Detection
**Team:** [Your Team Name]
**Started:** 2025-10-24

---

## Submissions Tracker

| # | Date | Submission File | Experiment | Val F1 | Public F1 | Private F1 | Selected | Notes |
|---|------|----------------|------------|--------|-----------|------------|----------|-------|
| 001 | YYYY-MM-DD | `submission_001.csv` | M1-baseline | 0.XXXX | 0.XXXX | ??? | ☐ | First baseline submission |
| 002 | | | | | | ??? | ☐ | |
| 003 | | | | | | ??? | ☐ | |

**Legend:**
- **Val F1:** Validation F1 score (local cross-validation)
- **Public F1:** Public leaderboard F1 score (~30% of test set)
- **Private F1:** Private leaderboard F1 (revealed after competition, ~70% of test set)
- **Selected:** ☑ = Selected for final scoring (pick 2 best before deadline!)

---

## Final Submission Selection

**Deadline to select:** [Competition end date - 2 weeks]

**Selected Submissions (pick 2):**

1. **Submission #XXX** - `submission_XXX.csv`
   - Experiment:
   - Reasoning:
   - Public F1:

2. **Submission #YYY** - `submission_YYY.csv`
   - Experiment:
   - Reasoning:
   - Public F1:

**Selection Strategy:**
- [ ] Reviewed all submission scores
- [ ] Checked validation F1 vs public F1 gaps (overfitting?)
- [ ] Selected diverse approaches (different models/configs)
- [ ] One conservative (best public LB), one risky (best validation)

---

## Detailed Submission Records

### Submission #001: M1-baseline (Example)

**Date:** 2025-10-24
**Experiment:** `runs/2025-10-24_15-30-00_M1-baseline/`

**Reproducibility Info:**
- Git commit: `061272d...`
- Config: `configs/accuracy.yaml`
- Training command:
  ```bash
  python src/train.py \
      --config runs/.../config.yaml \
      --data_root /kaggle/input/... \
      --epochs 50 --batch_size 4 --amp 1
  ```
- Inference command:
  ```bash
  python scripts/create_submission.py \
      --weights runs/.../best_model.pth \
      --config runs/.../config.yaml \
      --test_dir /kaggle/input/.../test_images \
      --output submission_001.csv
  ```

**Scores:**
- Validation F1: 0.XXXX
- Public F1: 0.XXXX
- Private F1: ??? (revealed after competition)

**Post-processing:**
- Threshold: 0.5
- Min area: 16
- Morphological: close=3, open=1

**Notes:**
- First submission with DINOv2-S/14 backbone
- No TTA, no ensemble
- Baseline for future improvements

---

### Submission #002: [Your next submission]

**Date:**
**Experiment:**

**Reproducibility Info:**
- Git commit:
- Config:
- Training command:
  ```bash

  ```
- Inference command:
  ```bash

  ```

**Scores:**
- Validation F1:
- Public F1:
- Private F1:

**Post-processing:**
- Threshold:
- Min area:
- Morphological:

**Notes:**


---

## Score Analysis

### Public vs Private LB Gap

Track overfitting by comparing public and validation scores:

| Submission | Val F1 | Public F1 | Gap | Overfitting? |
|------------|--------|-----------|-----|--------------|
| 001 | 0.XXXX | 0.XXXX | +0.XX | No |
| 002 | | | | |

**Threshold for concern:** Gap > 0.05 suggests overfitting to public LB

### Best Performers

**By Validation F1:**
1. Submission #XXX: 0.XXXX
2. Submission #YYY: 0.YYYY
3. Submission #ZZZ: 0.ZZZZ

**By Public F1:**
1. Submission #XXX: 0.XXXX
2. Submission #YYY: 0.YYYY
3. Submission #ZZZ: 0.ZZZZ

---

## Experiments Summary

Quick reference of what each milestone explored:

| Milestone | Key Changes | Submissions | Best Val F1 | Best Public F1 |
|-----------|-------------|-------------|-------------|----------------|
| M1 - Baseline | DINOv2-S/14, basic training | 001-003 | 0.XXXX | 0.XXXX |
| M2 - TTA | Test-time augmentation | 004-006 | | |
| M3 - Ensemble | Multiple models | 007-010 | | |
| M4 - Optimization | Hyperparameter tuning | 011-015 | | |

---

## Important Reminders

- [ ] **2 submissions per day limit** - Don't waste them!
- [ ] **Select 2 finals before deadline** - Or Kaggle picks randomly!
- [ ] **Track git commit for each submission** - For reproducibility
- [ ] **Save all submission files** - May need to resubmit
- [ ] **Document post-processing changes** - Affects final results

---

## Lessons Learned

### What Worked
-

### What Didn't Work
-

### Surprises
-

---

## Pre-Submission Checklist

Before each submission, verify:

- [ ] Ran validation locally - F1 score is reasonable
- [ ] All test images present in submission (no missing rows)
- [ ] Format validation passed (`scripts/validate_submission.py`)
- [ ] Header is exactly `case_id,annotation`
- [ ] "authentic" for no-forgery cases (not empty string)
- [ ] Git commit recorded
- [ ] Config and commands documented
- [ ] Noted any post-processing changes

---

**Last updated:** 2025-10-24
**Status:** In progress - Training first model (M1)
