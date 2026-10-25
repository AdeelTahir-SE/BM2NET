"""
BodyM data preprocessing script.

Builds unified CSV manifests per split and computes normalization statistics.

Outputs (all written to ../processed-bodym/):
    - train_manifest.csv, testA_manifest.csv, testB_manifest.csv
    - pixel_stats.json      (silhouette pixel mean/std)
    - hwg_stats.json         (height_cm, weight_kg mean/std)
    - measurement_stats.json (per-measurement mean/std)

Usage:
    python preprocessing/preprocess.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = PROJECT_ROOT / "bodym"
OUT_ROOT = PROJECT_ROOT / "processed-bodym"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "testA", "testB"]

MEASUREMENT_COLS = [
    "ankle", "arm-length", "bicep", "calf", "chest", "forearm",
    "height", "hip", "leg-length", "shoulder-breadth",
    "shoulder-to-crotch", "thigh", "waist", "wrist",
]


# ── Step 1: Build manifests ────────────────────────────────────────────────

def build_manifest(split: str) -> pd.DataFrame:
    """Merge the three CSVs and verify images exist."""
    split_dir = RAW_ROOT / split

    # Load CSVs
    photo_map = pd.read_csv(split_dir / "subject_to_photo_map.csv")
    measurements = pd.read_csv(split_dir / "measurements.csv")
    hwg = pd.read_csv(split_dir / "hwg_metadata.csv")

    # Merge: photo_map ← measurements ← hwg (all on subject_id)
    df = photo_map.merge(measurements, on="subject_id", how="inner")
    df = df.merge(hwg, on="subject_id", how="inner")

    # Build image paths and verify they exist
    mask_dir = split_dir / "mask"
    mask_left_dir = split_dir / "mask_left"

    df["frontal_path"] = df["photo_id"].apply(
        lambda pid: str(mask_dir / f"{pid}.png")
    )
    df["lateral_path"] = df["photo_id"].apply(
        lambda pid: str(mask_left_dir / f"{pid}.png")
    )

    # Verify existence
    frontal_exists = df["frontal_path"].apply(lambda p: Path(p).exists())
    lateral_exists = df["lateral_path"].apply(lambda p: Path(p).exists())
    valid = frontal_exists & lateral_exists
    dropped = (~valid).sum()
    if dropped > 0:
        print(f"  WARNING: Dropping {dropped} rows with missing images in {split}")
    df = df[valid].reset_index(drop=True)

    print(f"  {split}: {len(df)} samples "
          f"({df['subject_id'].nunique()} subjects)")
    return df


def save_manifests():
    """Build and save a manifest CSV for each split."""
    print("Step 1: Building manifests...")
    for split in SPLITS:
        df = build_manifest(split)
        out_path = OUT_ROOT / f"{split}_manifest.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved {out_path}")
    print()


# ── Step 2: Compute pixel statistics ──────────────────────────────────────

def compute_pixel_stats(max_samples: int = 500):
    """
    Compute mean and std of silhouette mask pixels from a random sample
    of the training set. Pixel values are scaled to [0, 1].
    """
    print("Step 2: Computing pixel statistics...")
    manifest = pd.read_csv(OUT_ROOT / "train_manifest.csv")

    # Sample to speed up computation
    if len(manifest) > max_samples:
        manifest = manifest.sample(n=max_samples, random_state=42)

    pixel_sums = 0.0
    pixel_sq_sums = 0.0
    pixel_count = 0

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="  Pixel stats"):
        for path_col in ["frontal_path", "lateral_path"]:
            img = np.array(Image.open(row[path_col]).resize((480, 640)),
                           dtype=np.float32) / 255.0
            pixel_sums += img.sum()
            pixel_sq_sums += (img ** 2).sum()
            pixel_count += img.size

    mean = float(pixel_sums / pixel_count)
    std = float(np.sqrt(pixel_sq_sums / pixel_count - mean ** 2))

    stats = {"mean": round(mean, 6), "std": round(std, 6)}
    out_path = OUT_ROOT / "pixel_stats.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Pixel stats: mean={stats['mean']}, std={stats['std']}")
    print(f"  Saved {out_path}\n")
    return stats


# ── Step 3: Compute height/weight statistics ──────────────────────────────

def compute_hwg_stats():
    """Compute mean and std for height_cm and weight_kg from training set."""
    print("Step 3: Computing height/weight statistics...")
    manifest = pd.read_csv(OUT_ROOT / "train_manifest.csv")

    # Deduplicate by subject_id to avoid bias from subjects with many photos
    subjects = manifest.drop_duplicates(subset="subject_id")

    stats = {}
    for col in ["height_cm", "weight_kg"]:
        vals = subjects[col].dropna()
        stats[col] = {
            "mean": round(float(vals.mean()), 4),
            "std": round(float(vals.std()), 4),
        }
        print(f"  {col}: mean={stats[col]['mean']}, std={stats[col]['std']}")

    out_path = OUT_ROOT / "hwg_stats.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved {out_path}\n")
    return stats


# ── Step 4: Compute measurement statistics ────────────────────────────────

def compute_measurement_stats():
    """Compute mean and std for each of the 14 body measurements."""
    print("Step 4: Computing measurement statistics...")
    manifest = pd.read_csv(OUT_ROOT / "train_manifest.csv")

    # Deduplicate by subject_id (measurements are per-subject, not per-photo)
    subjects = manifest.drop_duplicates(subset="subject_id")

    stats = {}
    for col in MEASUREMENT_COLS:
        vals = subjects[col].dropna()
        stats[col] = {
            "mean": round(float(vals.mean()), 4),
            "std": round(float(vals.std()), 4),
        }
        print(f"  {col}: mean={stats[col]['mean']:.2f}, std={stats[col]['std']:.2f}")

    out_path = OUT_ROOT / "measurement_stats.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved {out_path}\n")
    return stats


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("BodyM Preprocessing Pipeline")
    print("=" * 60)
    print(f"Raw data:   {RAW_ROOT}")
    print(f"Output dir: {OUT_ROOT}\n")

    save_manifests()
    compute_pixel_stats()
    compute_hwg_stats()
    compute_measurement_stats()

    print("=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
