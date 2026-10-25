"""
BM2-Net utility functions.

Helpers for normalization, checkpointing, and logging.
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ── Logging ────────────────────────────────────────────────────────────────

def setup_logger(name: str, log_dir: Optional[Path] = None) -> logging.Logger:
    """Configure a logger with console and optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"{name}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ── Reproducibility ───────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ── Normalization stats loading ───────────────────────────────────────────

def load_json_stats(path: Path) -> dict:
    """Load a JSON statistics file."""
    with open(path, "r") as f:
        return json.load(f)


def load_pixel_stats(data_root: Path) -> Tuple[float, float]:
    """Return (mean, std) for silhouette pixel values."""
    stats = load_json_stats(data_root / "pixel_stats.json")
    return stats["mean"], stats["std"]


def load_hwg_stats(data_root: Path) -> Dict[str, Dict[str, float]]:
    """Return height/weight normalization stats."""
    return load_json_stats(data_root / "hwg_stats.json")


def load_measurement_stats(data_root: Path) -> Dict[str, Dict[str, float]]:
    """Return per-measurement normalization stats."""
    return load_json_stats(data_root / "measurement_stats.json")


# ── Measurement normalization ─────────────────────────────────────────────

def normalize_measurements(
    values: torch.Tensor,
    stats: Dict[str, Dict[str, float]],
    measurement_names: list,
) -> torch.Tensor:
    """Z-score normalize measurement values.

    Args:
        values: Tensor of shape (num_measurements,) or (B, num_measurements)
        stats: Dict mapping measurement name → {"mean": ..., "std": ...}
        measurement_names: Ordered list of measurement names

    Returns:
        Normalized tensor of same shape.
    """
    means = torch.tensor([stats[m]["mean"] for m in measurement_names],
                         dtype=values.dtype, device=values.device)
    stds = torch.tensor([stats[m]["std"] for m in measurement_names],
                        dtype=values.dtype, device=values.device)
    return (values - means) / stds


def denormalize_measurements(
    values: torch.Tensor,
    stats: Dict[str, Dict[str, float]],
    measurement_names: list,
) -> torch.Tensor:
    """Reverse z-score normalization back to original cm scale.

    Args:
        values: Normalized tensor of shape (num_measurements,) or (B, num_measurements)
        stats: Dict mapping measurement name → {"mean": ..., "std": ...}
        measurement_names: Ordered list of measurement names

    Returns:
        De-normalized tensor in cm.
    """
    means = torch.tensor([stats[m]["mean"] for m in measurement_names],
                         dtype=values.dtype, device=values.device)
    stds = torch.tensor([stats[m]["std"] for m in measurement_names],
                        dtype=values.dtype, device=values.device)
    return values * stds + means


# ── Checkpointing ─────────────────────────────────────────────────────────

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_mae: float,
    scaler=None,
) -> None:
    """Save a training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_mae": best_mae,
    }
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()
    torch.save(state, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    scaler=None,
) -> dict:
    """Load a training checkpoint.

    Returns:
        Dict with 'epoch' and 'best_mae' keys.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and state.get("scheduler_state_dict"):
        scheduler.load_state_dict(state["scheduler_state_dict"])
    if scaler is not None and state.get("scaler_state_dict"):
        scaler.load_state_dict(state["scaler_state_dict"])
    return {"epoch": state["epoch"], "best_mae": state["best_mae"]}
