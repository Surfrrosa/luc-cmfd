# LUC-CMFD Competition Improvements

## Implemented SOTA Features ✅

### 1. Deep PatchMatch (HIGH ROI) ✅
**File:** `src/pm.py`

**What it does:**
- Multi-scale differentiable PatchMatch for robust source-target correspondence
- 3-4 iterations per scale with random search + propagation
- Top-K proposal extraction with score thresholding
- Exclusion radius to avoid trivial matches

**Based on:** "Image Copy-Move Forgery Detection via Deep PatchMatch and Pairwise Ranking Learning" (2024)

**Expected Impact:**
- Better separation of source/target regions
- Fewer false matches on hard backgrounds
- Improved generalization on subtle blends

**API:**
```python
from pm import pm_proposals

results = pm_proposals(
    feats,          # (B, C, H, W) normalized features
    iters=3,        # PatchMatch iterations
    topk=50,        # Top correspondences
    score_thr=0.2   # Minimum score
)
# Returns: {'qxy': query positions, 'txy': target positions, 'score': scores}
```

---

### 2. Strip Pooling in Decoder ✅
**File:** `src/model.py` (StripPooling + TinyDecoder)

**What it does:**
- Horizontal/vertical strip pooling for global context
- Helps with mask continuity for thin/fragmented regions
- Inspired by CMFDFormer's PCSD module

**Based on:** CMFDFormer (2023) continual learning approach

**Expected Impact:**
- Better handling of thin gel bands
- Improved connectivity for fragmented forgeries
- Minimal runtime overhead (~2-3%)

**Usage:**
```python
model = CMFDNet(
    backbone='dinov2_vits14',
    use_strip_pool=True  # Enable strip pooling
)
```

---

### 3. Component-Aware Pruning ✅
**File:** `src/post.py`

**What it does:**
- Score each component: `score = α·mean(corr) + β·inlier_ratio + γ·plausibility`
- Size-adaptive thresholds (small components need higher scores)
- Keeps legitimate small multi-clone regions
- Drops periodic-texture false positives

**New Functions:**
- `score_component()`: Multi-factor component scoring
- `component_aware_pruning()`: Size-adaptive component filtering
- `detect_periodicity()`: FFT-based periodic texture detection
- `is_band_shaped()`: Gel band detection for plausibility priors

**Expected Impact:**
- Precision boost on periodic backgrounds
- Better handling of multi-clone images
- Domain-aware filtering for biomedical images

**Usage:**
```python
from post import component_aware_pruning, detect_periodicity

# Prune low-quality components
cleaned_mask = component_aware_pruning(
    mask,
    corr_map=correlation_map,
    min_score=0.3,
    size_adaptive=True
)

# Detect periodic textures
is_periodic, strength = detect_periodicity(image_gray)
```

---

## Ready to Implement (Prioritized)

### 4. Rotation/Scale Stabilization (CHEAP, RELIABLE)
**Files to update:** `src/infer.py`

**What to add:**
- TTA with rot90 (0°/90°/180°/270°)
- Log-polar pooling on correlation maps
- Cheap signal-processing trick for rotation/scale invariance

**Config:**
```python
CONFIG = {
    'tta_rot90': True,
    'log_polar_pool': {
        'enable': True,
        'radius_bins': 24,
        'angle_bins': 32
    }
}
```

---

### 5. Keypoint Proposal Gate (SIFT/ORB)
**New file:** `src/kp.py`

**What it does:**
- Fast SIFT/ORB proposals to seed RANSAC neighborhoods
- Adds rotation/scale invariance
- Narrows dense search, reduces spurious geometry

**Expected Impact:**
- Catches rotated/scaled clones missed by dense correlation
- Faster RANSAC convergence
- Hybrid keypoint+deep approach

---

### 6. Upgraded Synthetic Data Generation
**File to update:** `synth/synth.py`

**Improvements:**
- Wider rotation range: ±30°
- Scale range: 0.7-1.3
- Intensity/histogram matching source→target
- JPEG-like compression artifacts (quality 50-90)
- Elastic warping
- Splicing-style composites
- Panel-level duplicates

---

### 7. Domain-Specific Forensics (BIOMEDICAL)

#### A. Scale Bar Detection
**New file:** `src/scalebar.py`

**What it does:**
- Auto-detect scale bars in scientific images
- OCR to read length text
- Compare bar-pixel length vs. stated units
- Flag inconsistencies as priors

**Why it matters:**
- Common fraud pattern: mismatched scale bars after copy-move
- Cheap precision boost on scientific figures
- Few competitors will implement this

---

#### B. Panel Splitting & Cross-Panel Correlation
**New file:** `src/panel.py`

**What it does:**
- Detect panel borders (whitespace/lines)
- Run cross-panel correlation
- Catch panel-to-panel duplication (common in western blots)

**Why it matters:**
- Many scientific frauds duplicate across panels
- Directly targets domain-specific patterns

---

#### C. Gel Band Heuristics
**Already in:** `src/post.py` (`is_band_shaped()`)

**Enhancements needed:**
- Prefer band-shaped components with parallel orientation
- Penalize blob-y masks unless geometry is strong
- PSF/blur consistency checks around pasted regions

---

## Integration Plan

### Phase 1: Core Improvements (Current Sprint)
1. ✅ Deep PatchMatch
2. ✅ Strip Pooling
3. ✅ Component-Aware Pruning
4. 🔄 Update inference pipeline to use all three
5. 🔄 Rotation/scale stabilization (TTA + log-polar)

### Phase 2: Enhanced Synthetics
1. Upgrade `synth/synth.py` with advanced augmentations
2. Retrain with improved data

### Phase 3: Domain-Specific Add-Ons
1. Keypoint proposals (kp.py)
2. Periodicity-aware masking
3. Scale bar forensics (if time permits)
4. Panel splitting (if time permits)

---

## A/B Testing Protocol

For each new feature:
1. **Baseline:** Current model on dev set
2. **+Feature:** Add feature, measure ΔF1 and Δruntime
3. **Keep if:** ΔF1 ≥ +0.01 AND Δruntime ≤ +10%

**Dev Set:** Hold out 200 diverse samples (100 authentic, 100 forged)

---

## Configuration Schema

```python
CONFIG = {
    # Core model
    'backbone': 'dinov2_vits14',
    'freeze_backbone': True,
    'use_decoder': True,
    'use_strip_pool': False,  # Toggle strip pooling

    # Deep PatchMatch
    'use_patchmatch': False,  # Toggle PM proposals
    'pm_iters': 3,
    'pm_topk': 50,
    'pm_score_thr': 0.2,

    # TTA & Stabilization
    'tta_rot90': False,
    'log_polar_pool': {
        'enable': False,
        'radius_bins': 24,
        'angle_bins': 32
    },

    # Post-processing
    'component_pruning': {
        'enable': True,
        'min_score': 0.3,
        'size_adaptive': True,
        'check_periodicity': True
    },

    # Keypoints
    'use_keypoints': False,
    'keypoint_type': 'ORB',  # 'SIFT' or 'ORB'

    # Domain-specific
    'gel_band_prior': False,
    'scale_bar_check': False,
    'panel_analysis': False
}
```

---

## Next Steps

**Immediate (today):**
1. Create updated `infer.py` that integrates PM + strip pooling + component pruning
2. Add TTA rotation stabilization
3. Test on validation set

**Short-term (this week):**
1. Upgrade synthetic data
2. Implement keypoint proposals
3. Run A/B tests on all toggles

**Optional (if time/performance budget allows):**
1. Scale bar forensics
2. Panel splitting
3. Advanced gel heuristics

---

## Expected F1 Improvements

Based on literature and conservative estimates:

| Feature | Expected ΔF1 | Runtime Δ |
|---------|-------------|-----------|
| Deep PatchMatch | +0.02 to +0.04 | +15% |
| Strip Pooling | +0.01 to +0.02 | +2% |
| Component Pruning | +0.01 to +0.03 | <1% |
| TTA Rot90 | +0.01 to +0.02 | +4x (test only) |
| Keypoint Proposals | +0.01 to +0.02 | +5% |
| **Total (conservative)** | **+0.06 to +0.13** | **+30-40%** |

With selective toggling based on A/B tests, we can likely achieve **+0.08 F1** with **<20% runtime overhead**.

---

## References

1. Deep PatchMatch: https://arxiv.org/abs/2404.17310
2. CMFDFormer: https://arxiv.org/abs/2311.13263
3. Scale Bar Detection: https://arxiv.org/html/2510.11260v1
4. Scientific Image Forensics: https://farid.berkeley.edu/downloads/publications/acm06a.pdf
5. Periodic Texture CMFD: https://www.researchgate.net/publication/323276647
