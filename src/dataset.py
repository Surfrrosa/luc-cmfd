"""Dataset and data loading utilities for CMFD - Updated for Kaggle data structure."""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Callable, List, Dict, Any, Union
import re

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image

# Optional: set to False if you want grayscale-as-1ch
DEFAULT_TO_RGB = True

def _pad_to(img_t: torch.Tensor, target_h: int, target_w: int, mode: str = "constant", value: float = 0.0) -> torch.Tensor:
    """
    Pad (C,H,W) tensor to (C,target_h,target_w).

    Args:
        img_t: Input tensor (C,H,W)
        target_h: Target height
        target_w: Target width
        mode: Padding mode ('constant', 'reflect', etc.)
        value: Fill value for constant padding

    Returns:
        Padded tensor (C,target_h,target_w)
    """
    _, h, w = img_t.shape
    pad_h = target_h - h
    pad_w = target_w - w
    # pad format is (left, right, top, bottom)
    return F.pad(img_t, (0, pad_w, 0, pad_h), mode=mode, value=value)

def _imread(path: Union[str, Path]) -> np.ndarray:
    """Read image with PIL, return np.uint8 (H,W,3) or (H,W,1)."""
    p = Path(path)
    img = Image.open(p).convert("RGB" if DEFAULT_TO_RGB else "L")
    arr = np.array(img)
    if arr.ndim == 2:  # (H,W) -> (H,W,1)
        arr = arr[:, :, None]
    return arr

def _to_tensor_image(arr: np.ndarray) -> torch.Tensor:
    """(H,W,C) uint8 -> float32 tensor (C,H,W) in [0,1]."""
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return t

def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Ensure mask is (H,W) binary {0,1}.
    Handles (C,H,W) or (H,W,C) by merging channels with max/any.
    Thresholds non-binary masks >0 -> 1.
    """
    m = mask
    # squeeze singleton dims
    while m.ndim > 2 and 1 in m.shape:
        m = m.squeeze()
    # channel-first (C,H,W) -> (H,W)
    if m.ndim == 3:
        # Accept both (C,H,W) and (H,W,C)
        if m.shape[0] in (2, 3, 4):         # (C,H,W)
            m = m.max(axis=0)
        elif m.shape[-1] in (2, 3, 4):      # (H,W,C)
            m = m.max(axis=-1)
        else:
            # Fallback: max over all extra dims
            m = m.reshape(m.shape[-2], m.shape[-1]).copy()
    # float/255 handling
    if m.dtype != np.uint8:
        m = (m > 0).astype(np.uint8)
    else:
        # Sometimes masks are {0,255}
        if m.max() > 1:
            m = (m > 0).astype(np.uint8)
    return m

def _to_tensor_mask(m: np.ndarray) -> torch.Tensor:
    """(H,W) uint8 {0,1} -> float32 tensor (1,H,W) ∈ {0,1}."""
    t = torch.from_numpy(m.astype(np.uint8)).unsqueeze(0).float()
    return t

def _stem(path: Union[str, Path]) -> str:
    return Path(path).stem

def _extract_case_id(name: str) -> str:
    """
    Extract a robust case_id from a filename like '12345.png' or 'case_00123.png'.
    We'll use the stem as default; if digits exist, prefer them.
    """
    stem = Path(name).stem
    digits = re.findall(r"\d+", stem)
    return digits[-1] if digits else stem

class CMFDDataset(Dataset):
    """
    Reads directly from the original Kaggle structure:

      root/
        train_images/
          authentic/   *.png|jpg
          forged/      *.png|jpg
        train_masks/   *.npy (mask per forged image)
        test_images/   *.png|jpg

    For training:
      - forged images have masks in train_masks (npy).
      - authentic images have no mask => zero mask returned.

    For test/eval:
      - Masks not available; dataset returns zeros mask (ignored by inference).
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",                # "train" or "test"
        transform: Optional[Callable] = None,
        mask_merge: str = "max",             # "max" (merge channels) or "first"
        image_exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
        preload_index: bool = True,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.mask_merge = mask_merge
        self.image_exts = image_exts

        if split == "train":
            self.img_dir_auth = self.root / "train_images" / "authentic"
            self.img_dir_forg = self.root / "train_images" / "forged"
            self.mask_dir = self.root / "train_masks"
            self.items = self._build_train_index(preload_index)
        elif split == "test":
            self.img_dir_test = self.root / "test_images"
            self.items = self._build_test_index(preload_index)
        else:
            raise ValueError(f"Unknown split={split}")

    def _is_image(self, p: Path) -> bool:
        return p.suffix.lower() in self.image_exts

    def _build_train_index(self, preload: bool) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []

        # Authentic images (no masks)
        if self.img_dir_auth.exists():
            for p in sorted(self.img_dir_auth.iterdir()):
                if p.is_file() and self._is_image(p):
                    cid = _extract_case_id(p.name)
                    items.append({
                        "image_path": p,
                        "mask_path": None,     # no mask
                        "case_id": cid,
                        "is_forged": False
                    })

        # Forged images (have .npy masks)
        if self.img_dir_forg.exists():
            for p in sorted(self.img_dir_forg.iterdir()):
                if p.is_file() and self._is_image(p):
                    cid = _extract_case_id(p.name)
                    mpath = self._match_mask_for_case(cid)
                    items.append({
                        "image_path": p,
                        "mask_path": mpath,    # may be None if missing; we handle gracefully
                        "case_id": cid,
                        "is_forged": True
                    })
        if preload and len(items) == 0:
            raise FileNotFoundError(f"No training items found under {self.root}")
        return items

    def _build_test_index(self, preload: bool) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        if not self.img_dir_test.exists():
            raise FileNotFoundError(f"Missing test_images at {self.img_dir_test}")
        for p in sorted(self.img_dir_test.iterdir()):
            if p.is_file() and self._is_image(p):
                cid = _extract_case_id(p.name)
                items.append({
                    "image_path": p,
                    "mask_path": None,
                    "case_id": cid,
                    "is_forged": None
                })
        if preload and len(items) == 0:
            raise FileNotFoundError(f"No test items found under {self.root}")
        return items

    def _match_mask_for_case(self, case_id: str) -> Optional[Path]:
        """Find mask .npy by case_id. Tries exact match first, then fuzzy."""
        if not (self.root / "train_masks").exists():
            return None
        # exact match: <case_id>.npy
        exact = self.root / "train_masks" / f"{case_id}.npy"
        if exact.exists():
            return exact
        # fuzzy: pick first .npy that contains the id token
        for p in (self.root / "train_masks").glob("*.npy"):
            if case_id in p.stem:
                return p
        return None

    def __len__(self) -> int:
        return len(self.items)

    def _load_mask(self, path: Optional[Path], image_hw: Tuple[int, int]) -> np.ndarray:
        H, W = image_hw
        if path is None:
            return np.zeros((H, W), dtype=np.uint8)
        # memory-map read to save RAM where possible
        m = np.load(path, allow_pickle=False, mmap_mode="r")
        m = np.array(m)  # ensure we have a real ndarray (drop memmap view)
        # merge channels
        if m.ndim == 3:
            if self.mask_merge == "first":
                if m.shape[0] in (2, 3, 4):    # (C,H,W)
                    m = m[0]
                else:                        # (H,W,C)
                    m = m[..., 0]
            else:  # "max" = union of regions
                if m.shape[0] in (2, 3, 4):    # (C,H,W)
                    m = m.max(axis=0)
                else:                        # (H,W,C)
                    m = m.max(axis=-1)
        m = _normalize_mask(m)               # -> (H,W) {0,1}
        # If mask dims don't match image, resize by nearest (rare but safe)
        if (m.shape[0] != H) or (m.shape[1] != W):
            m_img = Image.fromarray(m.astype(np.uint8))
            m = np.array(m_img.resize((W, H), Image.NEAREST))
            m = (m > 0).astype(np.uint8)
        return m

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        ipath: Path = item["image_path"]
        mpath: Optional[Path] = item["mask_path"]
        case_id: str = item["case_id"]
        is_forged = item["is_forged"]

        img = _imread(ipath)                         # (H,W,1or3) uint8
        H, W = img.shape[:2]
        mask = self._load_mask(mpath, (H, W))        # (H,W) {0,1}

        # optional transforms (augmentations); expect callable(dict)->dict
        sample = {
            "image": img,               # numpy
            "mask": mask,               # numpy
            "case_id": case_id,
            "is_forged": is_forged,
            "original_size": (H, W)
        }

        if self.transform is not None:
            sample = self.transform(sample)

        # to tensors for training
        if isinstance(sample["image"], np.ndarray):
            sample["image"] = _to_tensor_image(sample["image"])     # (C,H,W)
        if isinstance(sample["mask"], np.ndarray):
            sample["mask"] = _to_tensor_mask(sample["mask"])        # (1,H,W) float {0,1}

        return sample


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Padding-based collate function that preserves aspect ratios.

    Pads all images/masks in a batch to the max(H), max(W) within the batch,
    rounded up to multiple of 32 for ViT/stride alignment.

    This avoids distortion of biomedical structures and keeps features aligned.

    Args:
        batch: List of sample dicts from __getitem__

    Returns:
        Batched dictionary with padded 'image', 'mask', and metadata
    """
    # Find target size (max dimensions in batch)
    hs = [item['image'].shape[1] for item in batch]
    ws = [item['image'].shape[2] for item in batch]
    Ht = max(hs)
    Wt = max(ws)

    # Pad to multiple of 224 (LCM of 14 and 32)
    # - DINOv2 patch_embed requires multiples of patch_size (14)
    # - Decoder/stride may require multiples of 32
    # - LCM(14, 32) = 224 satisfies both
    mul = 224
    Ht = ((Ht + mul - 1) // mul) * mul
    Wt = ((Wt + mul - 1) // mul) * mul

    # Pad images & masks
    padded_imgs = []
    padded_msks = []

    for item in batch:
        img = item['image'].contiguous()  # (C,H,W)
        msk = item['mask'].contiguous()   # (1,H,W)

        img_p = _pad_to(img, Ht, Wt, mode="constant", value=0.0)
        msk_p = _pad_to(msk, Ht, Wt, mode="constant", value=0.0)

        padded_imgs.append(img_p)
        padded_msks.append(msk_p)

    images = torch.stack(padded_imgs, dim=0)   # (B,C,Ht,Wt)
    masks = torch.stack(padded_msks, dim=0)    # (B,1,Ht,Wt)

    return {
        'image': images,
        'mask': masks,
        'case_id': [item['case_id'] for item in batch],
        'is_forged': [item['is_forged'] for item in batch],
        'original_size': [item['original_size'] for item in batch]
    }
