"""
BM2-Net training configuration.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class TrainConfig:
    """All hyperparameters and paths for BM2-Net training."""

    # ── Paths (relative defaults for Windows, Linux, Colab) ────────────
    project_root: str = str(Path(__file__).resolve().parent.parent)
    data_root: str = ""
    raw_data_root: str = ""
    checkpoint_dir: str = ""
    log_dir: str = ""

    def __post_init__(self):
        root = Path(self.project_root)
        if not self.data_root:
            self.data_root = str(root / "processed-bodym")
        if not self.raw_data_root:
            self.raw_data_root = str(root / "bodym")
        if not self.checkpoint_dir:
            self.checkpoint_dir = str(root / "train" / "checkpoints")
        if not self.log_dir:
            self.log_dir = str(root / "train" / "logs")

    # ── Image / Input ──────────────────────────────────────────────────
    img_height: int = 640          # Height of each view after resize
    img_width: int = 480           # Width of each view after resize
    input_channels: int = 3       # Ch0=silhouettes, Ch1=height, Ch2=weight
    input_height: int = 640       # Final composite tensor height
    input_width: int = 960        # Final composite tensor width (480+480)

    # ── Architecture ───────────────────────────────────────────────────
    backbone: str = "mnasnet1_0"  # MNASNet with depth multiplier 1
    backbone_out_channels: int = 1280  # MNASNet1_0 feature channels
    d_model: int = 256            # Token embedding dimension
    n_heads: int = 8              # Multi-head attention heads
    mlp_hidden: int = 128         # MLP head hidden layer size
    dropout: float = 0.1          # Dropout rate

    # ── Measurements ───────────────────────────────────────────────────
    num_measurements: int = 14
    measurement_names: List[str] = field(default_factory=lambda: [
        "ankle", "arm-length", "bicep", "calf", "chest", "forearm",
        "height", "hip", "leg-length", "shoulder-breadth",
        "shoulder-to-crotch", "thigh", "waist", "wrist",
    ])

    # ── Training ───────────────────────────────────────────────────────
    epochs: int = 100
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine"     # "cosine" or "step"
    warmup_epochs: int = 5
    grad_clip_norm: float = 1.0
    eval_every: int = 5           # Evaluate on val set every N epochs
    save_every: int = 10          # Checkpoint every N epochs

    # ── DataLoader ─────────────────────────────────────────────────────
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2

    # ── Augmentation ───────────────────────────────────────────────────
    augment: bool = True
    random_hflip: float = 0.5     # probability
    random_affine_degrees: float = 5.0
    random_affine_translate: float = 0.05
    random_erasing_prob: float = 0.1

    # ── Misc ───────────────────────────────────────────────────────────
    seed: int = 42
    device: str = "cuda"          # "cuda" or "cpu"
    mixed_precision: bool = True  # Use AMP

    @property
    def data_root_path(self) -> Path:
        return Path(self.data_root)

    @property
    def checkpoint_path(self) -> Path:
        p = Path(self.checkpoint_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def log_path(self) -> Path:
        p = Path(self.log_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p
