# START HERE - Navigation Guide

**If you're a fresh Claude instance or new team member, read this first.**

---

## Quick Navigation

**Need to get up to speed fast?**
1. Read **QUICKSTART.md** (5 min)
2. Skim **CONTEXT.md** (10 min)
3. Check **QUALITY_STATUS.md** (5 min)

**Need full context?**
1. Read **CONTEXT.md** thoroughly (20 min)
2. Read **CHECKLIST.md** for implementation status
3. Read **IMPROVEMENTS.md** for technical details
4. Check current config: **configs/accuracy.yaml**

---

## File Guide

### Essential Reading
- **CONTEXT.md** - Complete project history, decisions, and context
- **QUICKSTART.md** - Immediate next steps and key commands
- **QUALITY_STATUS.md** - What's implemented and why

### Reference Documentation
- **CHECKLIST.md** - Detailed implementation checklist
- **IMPROVEMENTS.md** - Technical details of SOTA features
- **STATUS.md** - Implementation status summary
- **README.md** - General project overview (slightly outdated)

### Configuration
- **configs/accuracy.yaml** - RECOMMENDED for competition (max F1)
- **configs/safe.yaml** - Fallback (guaranteed <2h runtime)
- **configs/ablation.yaml** - A/B testing framework

### Code
- **src/infer.py** - Full inference pipeline with all features
- **src/model.py** - Architecture (DINOv2 + decoder + strip pooling)
- **src/pm.py** - Deep PatchMatch module
- **src/kp.py** - Keypoint proposals
- **src/post.py** - Component-aware pruning + periodicity detection

### Scripts
- **scripts/validate_submission.py** - Validate RLE format (CRITICAL)
- **train_kaggle.py** - Training script
- **kaggle_train.ipynb** - Kaggle notebook for GPU training

---

## Current Status (Oct 24, 2025)

**Completed:**
- All core modules and SOTA features
- Three config presets
- Validation infrastructure
- Git repository with full documentation
- 150-image dev subset

**Not Yet Done:**
- Model training (no weights yet)
- First competition submission

**Immediate Next Action:**
Train model on Kaggle GPU using kaggle_train.ipynb with configs/accuracy.yaml

---

## Key Facts

- **Competition:** Recod.ai/LUC - Scientific Image Forgery Detection
- **Goal:** Maximum F1 score while staying <4h runtime
- **User Priority:** Quality/accuracy over speed
- **Config to Use:** configs/accuracy.yaml
- **Expected F1:** ~0.70-0.78 with all features
- **Expected Runtime:** <3.5h

---

## Critical Commands

```bash
# Validate submission (ALWAYS run before submitting)
python scripts/validate_submission.py submission.csv

# Run inference with accuracy config
python src/infer.py \
  --weights weights/best_model.pth \
  --image_dir data/test_images \
  --config configs/accuracy.yaml \
  --output submission.csv

# Commit and push changes
git add .
git commit -m "Description"
git push origin main
```

---

## User Preferences

- No emojis in documentation
- Quality/accuracy is priority over speed
- Working as a team for competition
- Detail-oriented and collaborative

---

## Warnings

1. ALWAYS validate submission before submitting to Kaggle
2. Use configs/accuracy.yaml (NOT old fast.yaml or tiny_clones.yaml)
3. Runtime must be <4h or will timeout
4. Never commit data/ or weights/ to git (already in .gitignore)
5. Import structure: use `from corr import ...` not `from .corr import ...`

---

## Success Criteria

**Minimum:** Valid submission, F1 > 0.5, runtime <4h  
**Target:** F1 > 0.70, runtime <3h, top 25% leaderboard  
**Stretch:** F1 > 0.75, top 10% leaderboard

---

**Now go read CONTEXT.md for full details. Good luck!**
