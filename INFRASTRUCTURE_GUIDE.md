# Infrastructure Tools Guide

**Created:** 2025-10-24
**Purpose:** Quick reference for experiment tracking and iteration tools

---

## 🎯 Overview

We now have a complete infrastructure for efficient iteration during the 3-month competition:

1. **Experiment tracking** - Organized runs with metadata
2. **CSV logging** - Per-epoch metrics for analysis
3. **Prediction saving** - Fast post-processing optimization
4. **Visualization** - Visual sanity checks
5. **Submission pipeline** - End-to-end inference
6. **Threshold sweeping** - Quick wins without retraining

---

## 🚀 Quick Start

### 1. Setup New Experiment

```bash
python scripts/setup_experiment.py \
    --name M1-baseline \
    --config configs/accuracy.yaml \
    --notes "First training run with DINOv2-S/14"
```

**Output:**
```
runs/2025-10-24_15-30-00_M1-baseline/
    ├── config.yaml          # Copy of training config
    ├── train_log.csv        # Per-epoch metrics (created during training)
    ├── best_model.pth       # Best checkpoint (created during training)
    ├── val_predictions.npz  # Validation predictions (created during training)
    ├── metadata.json        # Git hash, command, notes
    └── README.md            # Usage instructions
```

### 2. Train with Logging

```bash
python src/train.py \
    --config runs/2025-10-24_15-30-00_M1-baseline/config.yaml \
    --data_root /kaggle/input/recodai-luc-scientific-image-forgery-detection \
    --log_csv runs/2025-10-24_15-30-00_M1-baseline/train_log.csv \
    --save_val_preds runs/2025-10-24_15-30-00_M1-baseline/val_predictions.npz \
    --weights_out runs/2025-10-24_15-30-00_M1-baseline/best_model.pth
```

**What happens:**
- CSV log updated after each epoch with: epoch, train_loss, train_f1, val_loss, val_f1, lr, best
- Best model saved when validation F1 improves
- After training completes, validation predictions saved for fast sweeps

### 3. Visualize Predictions (Sanity Check)

```bash
python scripts/visualize_predictions.py \
    --val_preds runs/2025-10-24_15-30-00_M1-baseline/val_predictions.npz \
    --output runs/2025-10-24_15-30-00_M1-baseline/sanity_check.png \
    --n_samples 8
```

**Output:**
- 8-row visualization with 4 columns per row:
  1. **Probability heatmap** (model output)
  2. **Ground truth** mask
  3. **Binary prediction** (threshold=0.5)
  4. **Overlay** - TP (green), FP (red), FN (blue)
- F1 score computed for each sample
- Summary stats for thresholds 0.3, 0.4, 0.5, 0.6

### 4. Sweep Post-Processing (Quick Wins)

```bash
python scripts/sweep_threshold.py \
    --val_preds runs/2025-10-24_15-30-00_M1-baseline/val_predictions.npz \
    --thresholds 0.3 0.35 0.4 0.45 0.5 0.55 0.6 \
    --min_areas 8 12 16 20 24 \
    --output runs/2025-10-24_15-30-00_M1-baseline/threshold_sweep.csv
```

**Output:** CSV ranked by F1 score
```
threshold,min_area,close_size,open_size,mean_f1,std_f1,n_images
0.45,16,3,1,0.6234,0.1234,1234
0.5,12,3,1,0.6198,0.1221,1234
...
```

**Benefit:** Find best threshold/min_area in minutes instead of retraining!

### 5. Create Submission

```bash
python scripts/create_submission.py \
    --weights runs/2025-10-24_15-30-00_M1-baseline/best_model.pth \
    --config runs/2025-10-24_15-30-00_M1-baseline/config.yaml \
    --test_dir /kaggle/input/recodai-luc-cmfd/test_images \
    --output runs/2025-10-24_15-30-00_M1-baseline/submission.csv
```

**Features:**
- Loads model and config
- Runs inference on all test images
- Applies post-processing (from config)
- RLE encodes masks
- Validates format
- Writes submission.csv

**Dry run:** Add `--dry_run` to test on first 10 images

---

## 📊 Analyzing Results

### View Training Progress

```bash
# In Python or notebook
import pandas as pd
df = pd.read_csv('runs/.../train_log.csv')

# Plot F1 over time
import matplotlib.pyplot as plt
plt.plot(df['epoch'], df['train_f1'], label='Train')
plt.plot(df['epoch'], df['val_f1'], label='Val')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.savefig('training_curve.png')
```

### Compare Experiments

```bash
# Load logs from multiple runs
m1 = pd.read_csv('runs/.../M1-baseline/train_log.csv')
m2 = pd.read_csv('runs/.../M2-tta/train_log.csv')

# Compare best F1
print(f"M1 best: {m1['val_f1'].max():.4f}")
print(f"M2 best: {m2['val_f1'].max():.4f}")
```

### Check Experiment Metadata

```bash
# View git hash, notes, config used
cat runs/2025-10-24_15-30-00_M1-baseline/metadata.json
```

---

## 🎓 Workflow for Each Milestone

### Week 1 (M1): Baseline
1. `setup_experiment.py --name M1-baseline`
2. Train with `--log_csv` and `--save_val_preds`
3. `visualize_predictions.py` to check sanity
4. `sweep_threshold.py` to optimize post-processing
5. `create_submission.py` with best threshold
6. Submit to Kaggle

### Week 2 (M2): TTA & Hyperparameters
1. `setup_experiment.py --name M2-tta`
2. Train with TTA enabled
3. Compare train_log.csv with M1
4. Repeat sweep/submission

### Week 3-4 (M3): Ensemble
1. `setup_experiment.py --name M3-model1`
2. `setup_experiment.py --name M3-model2`
3. Train multiple models
4. Average predictions manually
5. Sweep on averaged predictions

---

## 🔧 Advanced Usage

### Custom Threshold Sweep

```bash
python scripts/sweep_threshold.py \
    --val_preds runs/.../val_predictions.npz \
    --thresholds $(seq 0.3 0.05 0.7) \
    --min_areas 4 8 12 16 20 24 28 32 \
    --close_size 5 \
    --open_size 2
```

### Filter Visualization Samples

```python
# In visualize_predictions.py, modify selection logic
# Current: Picks spread across confidence levels
# Customize: Pick worst F1, best F1, authentic only, etc.
```

### Reuse Predictions for Multiple Sweeps

```bash
# Save predictions once
python src/train.py ... --save_val_preds preds.npz

# Run multiple sweeps (instant!)
python scripts/sweep_threshold.py --val_preds preds.npz --thresholds ...
python scripts/sweep_threshold.py --val_preds preds.npz --min_areas ...
python scripts/sweep_threshold.py --val_preds preds.npz --close_size 5
```

---

## 📁 Directory Structure

Recommended structure for competition:

```
luc-cmfd/
├── runs/
│   ├── 2025-10-24_15-30-00_M1-baseline/
│   │   ├── config.yaml
│   │   ├── train_log.csv
│   │   ├── best_model.pth
│   │   ├── val_predictions.npz
│   │   ├── threshold_sweep.csv
│   │   ├── sanity_check.png
│   │   ├── submission.csv
│   │   └── metadata.json
│   ├── 2025-10-26_08-15-00_M2-tta/
│   └── 2025-11-02_14-00-00_M3-ensemble/
├── src/
├── scripts/
├── configs/
├── KAGGLE_RULES_CHEATSHEET.md
├── KAGGLE_SETUP.md
├── LESSONS_LEARNED.md
└── INFRASTRUCTURE_GUIDE.md (this file)
```

---

## 💡 Tips & Best Practices

### 1. Always Use Experiment Directories
- ❌ Don't scatter files in `/kaggle/working/`
- ✅ Use `setup_experiment.py` for every run
- **Why:** Easy to compare, reproduce, and share results

### 2. Save Validation Predictions Early
- ❌ Don't wait until you need them
- ✅ Use `--save_val_preds` from first training
- **Why:** Enables fast iteration without GPU time

### 3. Check Sanity Visualizations
- ❌ Don't skip visual checks
- ✅ Run `visualize_predictions.py` after every training
- **Why:** Catches bugs like all-black predictions, inverted masks, etc.

### 4. Track Metadata
- ❌ Don't forget which git commit you used
- ✅ Use `setup_experiment.py` to auto-track git hash
- **Why:** Reproducibility when you want to revisit old runs

### 5. Compare Apples to Apples
- ❌ Don't compare F1 scores from different validation splits
- ✅ Use same seed for train/val split across experiments
- **Why:** Fair comparison between models

### 6. Log Everything
- ❌ Don't rely on terminal output
- ✅ Use `--log_csv` to save metrics
- **Why:** Terminal output disappears, CSVs are permanent

---

## 🚨 Common Issues

### Issue: "No module named 'case_id'"
**Cause:** Dataset doesn't return case_id in batch
**Fix:** Validation predictions use fallback `val_0`, `val_1`, etc.

### Issue: "val_predictions.npz too large"
**Cause:** Large validation images
**Fix:** Predictions are saved as-is; use compressed npz (already done)

### Issue: "Threshold sweep shows no improvement"
**Cause:** Model predictions are poor quality
**Fix:** Check sanity visualization first, may need to retrain

### Issue: "Git dirty flag in metadata"
**Cause:** Uncommitted changes when experiment was created
**Fix:** Commit changes before `setup_experiment.py`

---

## 📚 Related Documentation

- **KAGGLE_SETUP.md** - How to run training on Kaggle
- **LESSONS_LEARNED.md** - Technical issues we solved
- **KAGGLE_RULES_CHEATSHEET.md** - Competition rules quick reference
- **configs/accuracy.yaml** - Training configuration reference

---

## 🎯 Next Steps

After training completes:

1. ✅ Check `train_log.csv` - Did F1 improve over epochs?
2. ✅ Run `visualize_predictions.py` - Do predictions look sane?
3. ✅ Run `sweep_threshold.py` - Can we optimize post-processing?
4. ✅ Create submission with best config
5. ✅ Submit to Kaggle
6. ✅ Compare public LB score with validation F1
7. ✅ Iterate!

**Goal:** Build up a library of experiments with tracked metadata, enabling rapid comparison and ensemble building.

---

**Last updated:** 2025-10-24
**Status:** ✅ Infrastructure complete, ready for iterations
