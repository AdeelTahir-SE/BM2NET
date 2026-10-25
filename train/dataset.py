"""
BodyM PyTorch Dataset.

Constructs the 3×640×960 composite input tensor:
    Channel 0: Frontal mask (640×480) || Lateral mask (640×480) side-by-side
    Channel 1: Height ξ — constant image (normalized height broadcast to 640×960)
    Channel 2: Weight ω — constant image (normalized weight broadcast to 640×960)
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from train.config import TrainConfig
from train.utils import load_hwg_stats, load_measurement_stats, load_pixel_stats


class BodyMDataset(Dataset):
    """
    PyTorch Dataset for the BodyM dataset.

    Each sample returns:
        input_tensor:   (3, 640, 960) composite tensor
        targets:        (14,) z-normalized body measurements
    """

    def __init__(
        self,
        manifest_path: str | Path,
        data_root: str | Path,
        config: TrainConfig,
        augment: bool = False,
    ):
        self.config = config
        self.augment = augment
        self.data_root = Path(data_root)

        # Load manifest
        self.manifest = pd.read_csv(manifest_path)

        # Load normalization statistics
        self.pixel_mean, self.pixel_std = load_pixel_stats(self.data_root)
        self.hwg_stats = load_hwg_stats(self.data_root)
        self.measurement_stats = load_measurement_stats(self.data_root)

    def __len__(self) -> int:
        return len(self.manifest)

    def _load_mask(self, path: str) -> np.ndarray:
        """Load a silhouette mask and resize to (img_height, img_width)."""
        img = Image.open(path).convert("L")
        img = img.resize(
            (self.config.img_width, self.config.img_height),
            Image.BILINEAR,
        )
        return np.array(img, dtype=np.float32) / 255.0

    def _normalize_scalar(self, value: float, stat_key: str) -> float:
        """Z-score normalize a scalar using precomputed stats."""
        mean = self.hwg_stats[stat_key]["mean"]
        std = self.hwg_stats[stat_key]["std"]
        return (value - mean) / std

    def _normalize_measurements(self, row: pd.Series) -> torch.Tensor:
        """Extract and z-normalize the 14 body measurements."""
        values = []
        for name in self.config.measurement_names:
            val = row[name]
            mean = self.measurement_stats[name]["mean"]
            std = self.measurement_stats[name]["std"]
            values.append((val - mean) / std)
        return torch.tensor(values, dtype=torch.float32)

    def _apply_augmentation(
        self,
        frontal: np.ndarray,
        lateral: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply consistent augmentations to both views."""
        # Random horizontal flip (flip both or neither)
        if np.random.random() < self.config.random_hflip:
            frontal = np.fliplr(frontal).copy()
            lateral = np.fliplr(lateral).copy()

        return frontal, lateral

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.manifest.iloc[idx]

        # ── Load silhouette masks ──────────────────────────────────────
        frontal = self._load_mask(row["frontal_path"])  # (640, 480)
        lateral = self._load_mask(row["lateral_path"])  # (640, 480)

        # ── Augmentation (train only) ──────────────────────────────────
        if self.augment:
            frontal, lateral = self._apply_augmentation(frontal, lateral)

        # ── Channel 0: Silhouettes side-by-side ────────────────────────
        # Normalize pixel values
        frontal = (frontal - self.pixel_mean) / self.pixel_std
        lateral = (lateral - self.pixel_mean) / self.pixel_std

        # Concatenate horizontally: (640, 480) || (640, 480) → (640, 960)
        ch0 = np.concatenate([frontal, lateral], axis=1)

        # ── Channel 1: Height constant image ──────────────────────────
        height_norm = self._normalize_scalar(row["height_cm"], "height_cm")
        ch1 = np.full(
            (self.config.input_height, self.config.input_width),
            height_norm,
            dtype=np.float32,
        )

        # ── Channel 2: Weight constant image ──────────────────────────
        weight_norm = self._normalize_scalar(row["weight_kg"], "weight_kg")
        ch2 = np.full(
            (self.config.input_height, self.config.input_width),
            weight_norm,
            dtype=np.float32,
        )

        # ── Stack to 3×640×960 ─────────────────────────────────────────
        input_tensor = torch.from_numpy(
            np.stack([ch0, ch1, ch2], axis=0)
        )  # (3, 640, 960)

        # ── Targets ────────────────────────────────────────────────────
        targets = self._normalize_measurements(row)  # (14,)

        return input_tensor, targets


def build_dataloaders(config: TrainConfig):
    """Create train, testA, and testB dataloaders.

    Returns:
        (train_loader, testA_loader, testB_loader)
    """
    data_root = Path(config.data_root)

    train_dataset = BodyMDataset(
        manifest_path=data_root / "train_manifest.csv",
        data_root=data_root,
        config=config,
        augment=config.augment,
    )

    testA_dataset = BodyMDataset(
        manifest_path=data_root / "testA_manifest.csv",
        data_root=data_root,
        config=config,
        augment=False,
    )

    testB_dataset = BodyMDataset(
        manifest_path=data_root / "testB_manifest.csv",
        data_root=data_root,
        config=config,
        augment=False,
    )

    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
        persistent_workers=config.num_workers > 0,
    )

    testA_loader = DataLoader(
        testA_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
    )

    testB_loader = DataLoader(
        testB_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
    )

    return train_loader, testA_loader, testB_loader
