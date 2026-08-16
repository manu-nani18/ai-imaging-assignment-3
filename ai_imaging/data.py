from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def image_paths(data_dir: Path, split: str) -> list[Path]:
    paths = sorted((data_dir / split / "images").glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG images found in {data_dir / split / 'images'}")
    return paths


def paired_paths(data_dir: Path, split: str) -> list[tuple[Path, Path]]:
    pairs = []
    for image_path in image_paths(data_dir, split):
        mask_path = data_dir / split / "masks" / image_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {image_path.name}: {mask_path}")
        pairs.append((image_path, mask_path))
    return pairs


def load_grayscale(path: Path, size: int = 256) -> np.ndarray:
    image = Image.open(path).convert("L").resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def load_mask(path: Path, size: int = 256) -> np.ndarray:
    mask = Image.open(path).convert("L").resize((size, size), Image.Resampling.NEAREST)
    return (np.asarray(mask) > 0).astype(np.float32)


class NucleiDataset:
    """Lazy PyTorch dataset; imports torch only when samples are requested."""

    def __init__(self, data_dir: Path, split: str, size: int = 256, augment: bool = False):
        self.pairs = paired_paths(data_dir, split)
        self.size = size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        import torch

        image_path, mask_path = self.pairs[index]
        image = load_grayscale(image_path, self.size)
        mask = load_mask(mask_path, self.size)
        if self.augment:
            if random.random() < 0.5:
                image, mask = np.fliplr(image).copy(), np.fliplr(mask).copy()
            if random.random() < 0.5:
                image, mask = np.flipud(image).copy(), np.flipud(mask).copy()
        return torch.from_numpy(image[None]), torch.from_numpy(mask[None]), image_path.stem

