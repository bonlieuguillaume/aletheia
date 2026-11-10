"""SAR image dataset.

Expected layout (one sub-folder per class):

    data_root/
        train/real/*.tif|png|npy ...
        train/fake/...
        val/real/...
        val/fake/...
        test/...            (optional)

Forensic note: resizing / JPEG re-compression destroy the low-level traces
that betray a manipulation, so we *crop* to `img_size` instead of resizing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".npy"}


def load_sar(path: Path) -> np.ndarray:
    """Load a SAR image as a float32 2-D array (amplitude/intensity)."""
    ext = path.suffix.lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext in {".tif", ".tiff"}:
        import tifffile

        arr = tifffile.imread(path)
    else:
        arr = np.asarray(Image.open(path))

    arr = arr.astype(np.float32)
    if np.iscomplexobj(arr):  # SLC data -> amplitude
        arr = np.abs(arr)
    if arr.ndim == 3:  # (H, W, C) or (C, H, W) -> single channel
        arr = arr.mean(axis=-1 if arr.shape[-1] <= 4 else 0)
    return arr


def normalize_sar(arr: np.ndarray, log_transform: bool = True) -> np.ndarray:
    """Log-compress the heavy-tailed SAR dynamic range then standardise."""
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if log_transform:
        arr = np.log1p(np.clip(arr, 0, None))
    std = arr.std()
    return (arr - arr.mean()) / (std if std > 1e-6 else 1.0)


def crop_or_pad(arr: np.ndarray, size: int, random: bool) -> np.ndarray:
    h, w = arr.shape
    if h < size or w < size:
        ph, pw = max(0, size - h), max(0, size - w)
        arr = np.pad(arr, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)), mode="reflect")
        h, w = arr.shape
    if random:
        top = np.random.randint(0, h - size + 1)
        left = np.random.randint(0, w - size + 1)
    else:
        top, left = (h - size) // 2, (w - size) // 2
    return arr[top : top + size, left : left + size]


class SARDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        class_names: list[str],
        img_size: int = 256,
        train: bool = False,
        log_transform: bool = True,
    ):
        self.root = Path(root) / split
        self.class_names = class_names
        self.img_size = img_size
        self.train = train
        self.log_transform = log_transform

        self.samples: list[tuple[Path, int]] = []
        for label, cls in enumerate(class_names):
            cls_dir = self.root / cls
            if not cls_dir.is_dir():
                raise FileNotFoundError(f"Missing class folder: {cls_dir}")
            files = sorted(p for p in cls_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
            self.samples += [(p, label) for p in files]
        if not self.samples:
            raise RuntimeError(f"No images found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def class_counts(self) -> list[int]:
        counts = [0] * len(self.class_names)
        for _, y in self.samples:
            counts[y] += 1
        return counts

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        arr = normalize_sar(load_sar(path), self.log_transform)
        arr = crop_or_pad(arr, self.img_size, random=self.train)
        if self.train:
            # Geometric augmentations only: they keep pixel statistics intact.
            if np.random.rand() < 0.5:
                arr = arr[:, ::-1]
            if np.random.rand() < 0.5:
                arr = arr[::-1, :]
            arr = np.rot90(arr, k=np.random.randint(4))
        x = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0)  # (1, H, W)
        return x, label
