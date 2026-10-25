"""
BM2-Net evaluation script.

Evaluates a trained model on testA and/or testB splits.

Usage:
    python -m train.evaluate --checkpoint train/checkpoints/best_model.pt
    python -m train.evaluate --checkpoint train/checkpoints/best_model.pt --split testB
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.amp import autocast

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from train.config import TrainConfig
from train.dataset import BodyMDataset
from train.model import build_model
from train.utils import (
    denormalize_measurements,
    load_checkpoint,
    load_measurement_stats,
    set_seed,
    setup_logger,
)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    measurement_stats: dict,
    measurement_names: list,
    use_amp: bool = True,
) -> dict:
    """
    Run full evaluation and return detailed metrics.

    Returns:
        dict with keys:
            - mae_overall: float (cm)
            - mae_per_measurement: dict[str, float]
            - mse_overall: float
            - predictions: Tensor (N, 14) in cm
            - targets: Tensor (N, 14) in cm
    """
    model.eval()
    all_preds = []
    all_targets = []

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        with autocast(device_type=device.type, enabled=use_amp):
            preds = model(inputs)

        # De-normalize to cm
        preds_cm = denormalize_measurements(preds, measurement_stats, measurement_names)
        targets_cm = denormalize_measurements(targets, measurement_stats, measurement_names)

        all_preds.append(preds_cm.cpu())
        all_targets.append(targets_cm.cpu())

    all_preds = torch.cat(all_preds, dim=0)      # (N, 14)
    all_targets = torch.cat(all_targets, dim=0)   # (N, 14)

    errors = (all_preds - all_targets).abs()      # (N, 14)
    mae_per = errors.mean(dim=0)                  # (14,)
    mae_overall = mae_per.mean().item()
    mse_overall = ((all_preds - all_targets) ** 2).mean().item()

    mae_dict = {
        name: round(mae_per[i].item(), 3)
        for i, name in enumerate(measurement_names)
    }

    return {
        "mae_overall": round(mae_overall, 3),
        "mae_per_measurement": mae_dict,
        "mse_overall": round(mse_overall, 3),
        "predictions": all_preds,
        "targets": all_targets,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate BM2-Net")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--split", type=str, default="both",
        choices=["testA", "testB", "both"],
        help="Which test split(s) to evaluate on",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no_amp", action="store_true")
    args = parser.parse_args()

    config = TrainConfig()
    if args.batch_size:
        config.batch_size = args.batch_size

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    use_amp = not args.no_amp

    logger = setup_logger("evaluate")
    set_seed(config.seed)

    # ── Load model ─────────────────────────────────────────────────────
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model = build_model(config).to(device)
    ckpt_info = load_checkpoint(Path(args.checkpoint), model)
    logger.info(f"Checkpoint from epoch {ckpt_info['epoch']+1}, "
                f"best MAE: {ckpt_info['best_mae']:.3f} cm")

    # ── Load stats ─────────────────────────────────────────────────────
    data_root = Path(config.data_root)
    measurement_stats = load_measurement_stats(data_root)

    # ── Evaluate ───────────────────────────────────────────────────────
    splits = ["testA", "testB"] if args.split == "both" else [args.split]

    for split in splits:
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluating on {split}")
        logger.info(f"{'='*50}")

        dataset = BodyMDataset(
            manifest_path=data_root / f"{split}_manifest.csv",
            data_root=data_root,
            config=config,
            augment=False,
        )

        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )

        logger.info(f"Samples: {len(dataset)}")

        results = evaluate_model(
            model, dataloader, device,
            measurement_stats, config.measurement_names,
            use_amp=use_amp,
        )

        logger.info(f"\nOverall MAE: {results['mae_overall']:.3f} cm")
        logger.info(f"Overall MSE: {results['mse_overall']:.3f}")
        logger.info(f"\nPer-measurement MAE (cm):")
        logger.info(f"{'─'*40}")
        for name, mae in results["mae_per_measurement"].items():
            logger.info(f"  {name:<22s} {mae:>7.3f}")
        logger.info(f"{'─'*40}")
        logger.info(f"  {'MEAN':<22s} {results['mae_overall']:>7.3f}")


if __name__ == "__main__":
    main()
