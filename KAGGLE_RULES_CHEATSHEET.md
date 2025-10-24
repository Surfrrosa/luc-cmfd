# Kaggle Competition Rules - Quick Reference

**Competition:** RECOD.ai/LUC Scientific Image Forgery Detection
**Updated:** 2025-10-24

---

## 🚨 CRITICAL LIMITS

### Submission Limits
- **Public Submissions:** 2 per day maximum
- **Final Submissions:** Select 2 for final scoring
- **Total Allowed:** Unlimited (but only 2/day count toward public LB)

**⚠️ STRATEGY:**
- Don't waste daily submissions on untested code!
- Save for tested improvements only
- Use local validation to filter ideas first

---

### Timeline

| Date | Event |
|------|-------|
| Start | Competition open |
| 2 weeks before end | Final submission window opens |
| Final day | Select your best 2 submissions |
| After close | Private leaderboard revealed |

**⚠️ WARNING:** Submissions after final deadline = DQ!

---

## 📋 Submission Format (STRICT!)

### Required Columns
```csv
case_id,annotation
12345,"150 3 153 3 156 3 ..."
12346,"authentic"
```

### Rules:
1. **Header:** Must be `case_id,annotation` (exact)
2. **For forged images:** RLE-encoded mask string
3. **For authentic images:** Literal string `"authentic"`
4. **No missing rows:** Every test image must have entry
5. **No duplicates:** One row per case_id

### RLE Format:
```
"start1 length1 start2 length2 ..."
```
- Pixel positions are 1-indexed (not 0!)
- Run-length encoded in row-major order
- Use our `rle_encode()` function (already built)

---

## ✅ Allowed

### Data & Pretraining
- ✅ **External pretrained models** (DINOv2, ResNet, etc.)
- ✅ **ImageNet** and similar pretrained weights
- ✅ **Public datasets** for pretraining
- ✅ **Self-supervised learning** on competition data

### Techniques
- ✅ **Ensembles** (multiple models)
- ✅ **Test-time augmentation** (TTA)
- ✅ **Post-processing** (any method)
- ✅ **Domain knowledge** (gel band priors, etc.)

### Code & Collaboration
- ✅ **Open-source libraries** (PyTorch, scikit-learn, etc.)
- ✅ **Team size:** Up to 4 members (check current rules!)
- ✅ **Public code sharing** (but not trained weights on test set)

---

## ❌ NOT Allowed

### Data
- ❌ **Manual labeling** of test set
- ❌ **Using test set** for training (obviously!)
- ❌ **External labeled data** that overlaps with test set

### Cheating
- ❌ **Multiple accounts** to bypass submission limits
- ❌ **Private sharing** of test set predictions between teams
- ❌ **Reverse engineering** test set labels from LB

### Technical
- ❌ **Hardcoding** test set predictions
- ❌ **Using competition platform resources** for cryptocurrency mining (seriously, it's in the rules)

---

## 🎯 Scoring

### Metric: **Pixel-Level F1 Score**

```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

Where:
- **TP:** Forged pixels correctly identified
- **FP:** Authentic pixels incorrectly marked as forged
- **FN:** Forged pixels missed

### Leaderboard Types:

**Public LB:**
- Computed on ~30% of test set
- Updates immediately after submission
- Visible during competition
- **CAN BE MISLEADING!** (Overfitting risk)

**Private LB:**
- Computed on remaining ~70% of test set
- **Only revealed after competition ends**
- **This determines final ranking!**
- Selected from your chosen 2 final submissions

---

## 🎓 Best Practices (Avoid Common Mistakes)

### Submission Strategy
1. **Validate locally first** - Don't submit untested ideas
2. **Track what you submit** - Keep a log with config/score
3. **2 submissions/day** - Use them wisely!
4. **Select finals carefully** - Pick diverse submissions (different approaches)

### Avoiding Overfitting Public LB
- ❌ Don't optimize purely for public LB score
- ✅ Trust your local validation
- ✅ Keep some ideas "un-tested" for final week
- ✅ Use k-fold validation for robust estimates

### Team Merge Rules
- Merging teams = combine submission limits
- Must merge before deadline (usually 7 days before end)
- Can't unmerge after merging

---

## 📊 Submission Tracking Template

Keep this log:

```
Date       | Config      | Val F1 | Public F1 | Notes
-----------|-------------|--------|-----------|------------------
2025-10-24 | M1-baseline | 0.55   | ???       | First submission
2025-10-26 | M1-tta      | 0.58   | ???       | Added TTA
2025-10-27 | M2-ensemble | 0.62   | ???       | 2-model ensemble
```

**Track:**
- What changed
- Local validation F1
- Public LB F1
- Any surprises (big gap = overfitting!)

---

## ⚠️ Common Disqualification Reasons

1. **Wrong submission format** → Immediate 0 score
2. **Missing test images** in submission → DQ
3. **Submitting after deadline** → DQ
4. **Using multiple accounts** → Permanent ban
5. **Not selecting final submissions** → Random selection (risky!)

---

## 🔍 Pre-Submission Checklist

Before EVERY submission:

- [ ] Ran `scripts/validate_submission.py` (we'll create this)
- [ ] Checked: All test images present?
- [ ] Checked: Format matches sample_submission.csv?
- [ ] Checked: No NaN or empty annotations?
- [ ] Checked: "authentic" for no-forgery cases (not empty string)?
- [ ] Saved config/code used for this submission
- [ ] Logged in tracking sheet

---

## 🎯 Target Timeline

| Week | Submissions | Goal |
|------|-------------|------|
| 1 | 2-3 | Baseline + TTA |
| 2-3 | 4-6 | Hyperparameter sweeps |
| 4-8 | 8-12 | Ensembles, improvements |
| 9-11 | 4-6 | Final optimizations |
| 12 | 2 | Select best 2 for finals |

**Total:** ~20-30 submissions over 3 months (well within limits)

---

## 📞 Where to Get Help

- **Discussion Forum:** Check for updates/clarifications
- **Host Q&A:** Watch for official responses to rule questions
- **Sample Submission:** Always available for format reference

---

## 🚀 Emergency Contacts

**If something seems wrong:**
1. Check Discussion forum first
2. Re-read Overview and Rules tabs
3. Contact competition hosts (via Kaggle messaging)
4. Don't assume - ask!

---

**Remember:**
- 🎯 Quality > Quantity (2 good submissions beats 30 bad ones)
- 📊 Local validation is your best friend
- 🤝 Private LB is what matters
- ⚡ Don't waste submissions on untested code!

**Good luck! 🍀**
