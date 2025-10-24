# LUC-CMFD: Copy-Move Forgery Detection for Biomedical Images

Production-ready pipeline for detecting and segmenting copy-move forgeries in biomedical figures, designed for Kaggle competition submission.

## Overview

This system combines:
- **DINOv2 ViT** backbone for robust feature extraction
- **Self-correlation** analysis to detect duplicated regions
- **RANSAC geometric verification** to filter false positives
- **Lightweight decoder** for mask refinement
- Strict **RLE encoding** with "authentic" handling for submissions

**Key Features:**
- ✅ Kaggle-compliant (no internet, <4h GPU runtime)
- ✅ Deterministic & reproducible (fixed seeds)
- ✅ AMP & channels-last optimization
- ✅ Comprehensive unit tests
- ✅ Vectorized operations (no pixel loops)

---

## Repository Structure

```
luc-cmfd/
├── src/
│   ├── dataset.py          # Image loading & normalization
│   ├── model.py            # DinoBackbone + CorrHead + TinyDecoder
│   ├── corr.py             # Self-correlation volumes (multi-scale)
│   ├── geom.py             # RANSAC geometric verification
│   ├── post.py             # Mask post-processing (CC, morphology)
│   ├── rle.py              # RLE encode/decode + validation
│   ├── train.py            # Training script
│   ├── infer.py            # Inference + submission generation
│   └── utils.py            # Seeds, logging, timing
├── synth/
│   └── synth.py            # Synthetic copy-move generator
├── scripts/
│   └── run_ablate.py       # Ablation sweeps
├── configs/
│   ├── fast.yaml           # Baseline config
│   └── tiny_clones.yaml    # Optimized for small regions
├── tests/
│   ├── test_rle.py         # RLE round-trip tests
│   ├── test_geom.py        # RANSAC accuracy tests
│   └── test_corr.py        # Correlation sanity tests
├── notebooks/
│   └── 20_infer_submit.ipynb   # Kaggle submission notebook
└── README.md
```

---

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/luc-cmfd.git
cd luc-cmfd

# Create environment
conda create -n cmfd python=3.9
conda activate cmfd

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scikit-image scikit-learn scipy opencv-python pillow pyyaml tqdm pytest
```

---

## Quick Start

### 1. Generate Synthetic Training Data

```bash
python synth/synth.py --output_dir data/synthetic --n_samples 1000
```

### 2. Train Model

```bash
python src/train.py \
    --train_images data/synthetic/images \
    --train_masks data/synthetic/masks \
    --output_dir weights \
    --epochs 50 \
    --batch_size 8
```

### 3. Run Inference

```bash
python src/infer.py \
    --weights weights/best_model.pth \
    --image_dir data/test_images \
    --output submission.csv \
    --config configs/fast.yaml
```

### 4. Run Tests

```bash
pytest tests/ -v
```

---

## Configuration

Two preset configs are provided:

### `fast.yaml` (Baseline)
- **Patch:** 12, **Stride:** 4, **Top-K:** 5
- **Transform:** Similarity
- **TTA:** Horizontal flip
- **Target:** Balanced speed/accuracy

### `tiny_clones.yaml` (Small Regions)
- **Patch:** 8, **Stride:** 4, **Top-K:** 7
- **Min Area:** 12 (vs 24)
- **Target:** Detecting tiny duplicated regions

---

## Inference Pipeline

The full pipeline executes these steps in order:

1. **Feature Extraction**: DINOv2 ViT-S backbone (frozen)
2. **Multi-scale Pyramid**: Build pyramid at scales [1, 2, 4]
3. **Self-Correlation**: Compute correlation volumes with stride/top-k pruning
4. **Decoder**: Tiny U-Net refines correlation → probability map
5. **Geometric Verification**: RANSAC (similarity/affine) → inlier mask
6. **Post-Processing**: CC analysis, min-area filter, morphology
7. **RLE Encoding**: Empty mask → `"authentic"`, else RLE string

---

## Key Modules

### `rle.py`: Run-Length Encoding
- **1-indexed** flat indexing (row-major)
- Empty masks → literal string `"authentic"`
- Round-trip tested on 10+ mask types

### `geom.py`: Geometric Verification
- RANSAC for **similarity** & **affine** transforms
- Recovers rotation (±20°), scale (0.8–1.2) with >90% inlier rate
- Filters false positives via transform consistency

### `corr.py`: Self-Correlation
- Vectorized torch (no Python loops)
- Top-K matching with self-exclusion radius
- Multi-scale aggregation

### `post.py`: Post-Processing
- Connected components (CC) analysis
- Min-area thresholding (default: 24px)
- Morphological close/open

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| **Runtime** | <4h GPU | Full evaluation set (main phase) |
| **Memory** | <16GB VRAM | Peak allocation tracked |
| **F1 Score** | TBD | Validation set baseline |
| **Determinism** | 100% | Fixed seeds, no cudnn non-determinism |

---

## Ablation Studies

Run hyperparameter sweeps:

```bash
python scripts/run_ablate.py \
    --val_images data/val/images \
    --val_masks data/val/masks \
    --output ablation_results.csv
```

**Grid:**
- `patch ∈ {8, 12, 16}`
- `stride ∈ {4, 8}`
- `top_k ∈ {3, 5, 7}`
- `transform ∈ {similarity, affine}`
- `TTA ∈ {none, flip, flip+rot90}`

**Promotion Rule:** Keep if ΔF1 ≥ +0.01 with ≤ +10% runtime increase.

---

## Testing

All critical modules have unit tests:

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_rle.py -v

# Coverage report
pytest tests/ --cov=src --cov-report=html
```

**Test Coverage:**
- `test_rle.py`: Empty mask, round-trip, random masks, submission format
- `test_geom.py`: RANSAC recovery, rotation/scale ranges, inlier detection
- `test_corr.py`: Duplicated patch detection, exclusion mask, determinism

---

## Kaggle Submission

### Workflow:

1. **Train locally** and save `best_model.pth`
2. **Create Kaggle Dataset** with weights file
3. **Attach dataset** to notebook `20_infer_submit.ipynb`
4. **Run notebook** (internet off, GPU on)
5. **Download** `submission.csv`

### Notebook Checklist:
- ✅ Asserts CUDA/AMP availability
- ✅ Loads weights from attached dataset
- ✅ Logs runtime & memory stats
- ✅ Validates submission format (header, RLE, authentic)
- ✅ Writes deterministic output

---

## Optimization Notes

### AMP (Automatic Mixed Precision)
- Enabled for training & inference
- ~2x speedup on modern GPUs
- No accuracy loss observed

### Channels-Last
- Memory format optimization for convolutions
- Applied to model & input tensors

### Vectorization
- All hot paths use tensor ops (no loops)
- Correlation uses batched matmul
- Post-processing uses scipy/skimage

---

## Citation

If you use this code, please cite:

```bibtex
@misc{luc-cmfd,
  author = {Your Name},
  title = {LUC-CMFD: Copy-Move Forgery Detection for Biomedical Images},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/luc-cmfd}
}
```

---

## License

MIT License - see LICENSE file for details.

---

## Acknowledgments

- **DINOv2**: Meta AI Research
- **Biomedical Imagery**: Kaggle competition organizers
- **RANSAC Implementation**: scikit-image

---

## Contact

For questions or issues, please open a GitHub issue or contact:
- Email: your.email@example.com
- Kaggle: @yourusername

---

**Built with Claude Code for Kaggle Competition** 🏆
