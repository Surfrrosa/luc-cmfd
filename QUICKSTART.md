# Quick Start Guide

**For a fresh Claude conversation or new team member**

---

## 1. Read This First

- **CONTEXT.md** - Full project history and decisions
- **configs/accuracy.yaml** - Current recommended config
- **QUALITY_STATUS.md** - What's implemented

---

## 2. Current Status

**We have:** Complete codebase with SOTA features  
**We need:** Train model and submit to Kaggle

---

## 3. Immediate Next Steps

### Step 1: Train Model on Kaggle

1. Go to Kaggle competition page
2. Create new notebook
3. Upload `kaggle_train.ipynb`
4. Attach competition dataset
5. Set GPU ON
6. Run all cells
7. Download `weights/best_model.pth`

### Step 2: Run Inference

```bash
python src/infer.py \
  --weights weights/best_model.pth \
  --image_dir data/recodai-luc-scientific-image-forgery-detection/test_images \
  --config configs/accuracy.yaml \
  --output submission.csv
```

### Step 3: Validate Submission

```bash
python scripts/validate_submission.py submission.csv
```

If validation passes, submit to Kaggle!

---

## 4. Key Commands

```bash
# Validate submission
python scripts/validate_submission.py submission.csv

# Train locally (slow, for testing only)
python train_kaggle.py --batch_size 2 --epochs 2

# Check git status
git status

# Commit changes
git add .
git commit -m "Description"
git push origin main
```

---

## 5. Important Files

```
configs/accuracy.yaml       # Use this config for competition
src/infer.py               # Full inference pipeline
scripts/validate_submission.py  # Check submission format
data/dev_subset.json       # 150-image dev set
CONTEXT.md                 # Full project context
```

---

## 6. Config Choice

- **configs/accuracy.yaml** - Maximum F1, all features ON (RECOMMENDED)
- **configs/safe.yaml** - Fast, guaranteed <2h runtime (fallback)
- **configs/ablation.yaml** - A/B testing framework

---

## 7. Critical Warnings

1. Always validate submission before submitting
2. Use configs/accuracy.yaml (not old fast.yaml)
3. Never commit data/ or weights/ (already in .gitignore)
4. Runtime must be <4h or will timeout
5. No emojis in documentation (user preference)

---

## 8. Competition Details

- **URL:** https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection
- **Task:** Detect copy-move forgeries in biomedical images
- **Metric:** F1 Score
- **Constraint:** <4h GPU runtime

---

## 9. Expected Performance

- **Baseline (decoder only):** ~0.65-0.70 F1
- **With SOTA features:** ~0.70-0.78 F1
- **Runtime:** <3.5h with accuracy.yaml

---

## 10. If Things Break

**Invalid submission:**
```bash
python scripts/validate_submission.py submission.csv
# Fix RLE encoding issues
```

**Runtime too long:**
```bash
# Switch to safe config
python src/infer.py --config configs/safe.yaml ...
```

**Out of memory:**
```bash
# Reduce batch size in config or training script
```

**Import errors:**
```bash
# Check imports use `from corr import ...` not `from .corr import ...`
```

---

**Ready to compete! Good luck!**
