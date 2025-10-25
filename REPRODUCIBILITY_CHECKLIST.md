# Reproducibility Checklist

**Competition:** RECOD.ai/LUC Scientific Image Forgery Detection
**Purpose:** Ensure all results can be reproduced for Kaggle's code review

---

## Why This Matters

Kaggle competitions require top finishers to **share reproducible code**. If the hosts can't reproduce your results, you may be **disqualified** even if you placed first!

This checklist ensures you meet all reproducibility requirements.

---

## Before Each Training Run

- [ ] **Create experiment directory**
  ```bash
  python scripts/setup_experiment.py --name M1-baseline --config configs/accuracy.yaml
  ```
  - Automatically captures git commit hash
  - Records environment info (Python, PyTorch, GPU)
  - Creates organized directory structure

- [ ] **Commit all code changes**
  ```bash
  git add -A
  git commit -m "Description of changes"
  git push origin main
  ```
  - Ensures git hash in metadata is clean (not "dirty")
  - Allows exact code recreation later

- [ ] **Set random seed**
  - Already done in `src/train.py` with `--seed 42`
  - Ensures reproducible train/val splits

- [ ] **Use version-controlled config**
  - Always use config files (e.g., `configs/accuracy.yaml`)
  - Don't hardcode hyperparameters in code

---

## During Training

- [ ] **Log all metrics**
  ```bash
  python src/train.py \
      --log_csv runs/.../train_log.csv \
      --save_val_preds runs/.../val_predictions.npz
  ```
  - Per-epoch metrics in CSV
  - Validation predictions for post-processing

- [ ] **Monitor git status**
  - Don't make code changes during training
  - If you must change code, commit and record new hash

- [ ] **Save best checkpoint**
  - Automatically done by `src/train.py`
  - Saved as `best_model.pth` in experiment directory

---

## After Training Completes

- [ ] **Capture full environment**
  ```bash
  python scripts/capture_environment.py --output runs/.../environment.json
  ```
  - Captures all Python package versions
  - Records GPU info, CUDA version, etc.

- [ ] **Verify model can be loaded**
  ```python
  import torch
  from model import CMFDNet

  model = CMFDNet(...)
  model.load_state_dict(torch.load('runs/.../best_model.pth'))
  # Should load without errors
  ```

- [ ] **Test inference reproducibility**
  - Run inference twice on same image
  - Outputs should be identical (bit-for-bit)

---

## Before Each Submission

- [ ] **Validate submission format**
  ```bash
  python scripts/validate_submission.py --submission submission.csv
  ```
  - Checks format (columns, RLE encoding, etc.)
  - Prevents DQ from format errors

- [ ] **Record in submissions log**
  - Open `SUBMISSIONS_LOG.md`
  - Fill in all fields:
    - Experiment directory
    - Git commit hash
    - Training command
    - Inference command
    - Post-processing parameters
    - Validation F1
    - Public F1 (after submission)

- [ ] **Save submission file**
  ```bash
  cp submission.csv runs/.../submission_001.csv
  ```
  - Keep all submission files for potential resubmission

---

## For Final Submission (Top 3 Finishers)

If you place in top 3, you'll need to provide:

- [ ] **Complete source code**
  - Entire git repository
  - Exact commit hash used for final submission

- [ ] **Environment specification**
  - `environment.json` from `capture_environment.py`
  - OR `requirements.txt` with exact versions
  - OR Docker image

- [ ] **Training script**
  ```bash
  # Example command that reproduces training
  python src/train.py \
      --config runs/2025-10-24_15-30-00_M1-baseline/config.yaml \
      --data_root /path/to/competition/data \
      --log_csv runs/.../train_log.csv \
      --save_val_preds runs/.../val_predictions.npz \
      --weights_out runs/.../best_model.pth \
      --seed 42
  ```

- [ ] **Inference script**
  ```bash
  # Example command that generates submission
  python scripts/create_submission.py \
      --weights runs/.../best_model.pth \
      --config runs/.../config.yaml \
      --test_dir /path/to/test_images \
      --output submission.csv
  ```

- [ ] **Post-processing details**
  - Threshold values
  - Min area filtering
  - Morphological operations
  - Any ensemble averaging

- [ ] **Hardware used**
  - GPU type (e.g., "Kaggle Tesla T4 x2")
  - CUDA version
  - Training time estimate

- [ ] **Data used**
  - Only competition data allowed
  - No external labeled data
  - Pretrained models OK (DINOv2, etc.)

- [ ] **Random seed**
  - Document seed used (we use 42)
  - Allows exact train/val split reproduction

---

## Reproducibility Testing

Before final submission, test that your results are reproducible:

1. **Clone fresh repo**
   ```bash
   cd /tmp
   git clone https://github.com/Surfrrosa/luc-cmfd.git
   cd luc-cmfd
   git checkout <your_final_commit_hash>
   ```

2. **Setup environment**
   ```bash
   pip install -r requirements.txt
   # Or use your documented environment setup
   ```

3. **Run training from scratch**
   ```bash
   python src/train.py ...  # Use exact command from metadata
   ```

4. **Compare results**
   - Training should reach same validation F1 (±0.01)
   - Same number of epochs before early stopping
   - Model weights may differ slightly (GPU non-determinism)

5. **Run inference**
   ```bash
   python scripts/create_submission.py ...
   ```

6. **Verify submission**
   - Should produce identical CSV (or very similar scores)
   - Format validation passes

---

## What Gets Tracked Automatically

Our infrastructure already tracks:

✅ **Git commit hash** - via `setup_experiment.py`
✅ **Training config** - saved in experiment directory
✅ **Per-epoch metrics** - via `--log_csv`
✅ **Random seed** - via `--seed` argument
✅ **Validation predictions** - via `--save_val_preds`
✅ **Environment info** - via `capture_environment.py`

You just need to:
- Use the tools we built
- Document commands in `SUBMISSIONS_LOG.md`

---

## Common Reproducibility Issues

### Issue 1: "Different results on different GPUs"
**Cause:** GPU-specific optimizations (cuDNN, TensorCores)
**Solution:** Document GPU used, accept minor differences (±0.01 F1)

### Issue 2: "Can't install package versions"
**Cause:** Old package versions deprecated
**Solution:** Use Docker image or Kaggle Notebooks (stable platform)

### Issue 3: "Git hash shows 'dirty'"
**Cause:** Uncommitted changes during training
**Solution:** Commit all changes before `setup_experiment.py`

### Issue 4: "Inference gives different results each time"
**Cause:** Missing random seed or non-deterministic ops
**Solution:** Set all seeds, disable dropout during inference

### Issue 5: "Lost track of which config I used"
**Cause:** Manual config editing without saving
**Solution:** Always use `setup_experiment.py` which copies config

---

## Quick Reference Commands

### Start new experiment
```bash
python scripts/setup_experiment.py --name M1-baseline --config configs/accuracy.yaml --notes "First training run"
```

### Capture environment
```bash
python scripts/capture_environment.py --output runs/.../environment.json
```

### Validate submission
```bash
python scripts/validate_submission.py submission.csv
```

### Check git status
```bash
git status  # Should be clean before training
git log -1  # Shows current commit hash
```

---

## Final Pre-Competition Checklist

Two weeks before competition end:

- [ ] Selected 2 best submissions in Kaggle interface
- [ ] Documented both submissions in `SUBMISSIONS_LOG.md`
- [ ] Tested reproducibility of selected submissions
- [ ] Saved all experiment directories
- [ ] Committed all code to GitHub
- [ ] Captured environment.json for final runs
- [ ] Ready to share code if requested

---

**Remember:** Reproducibility is not optional - it's a requirement for winning!

Use the infrastructure we built, document everything, and you'll be ready for code review.

**Last updated:** 2025-10-24
