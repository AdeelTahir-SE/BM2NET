"""
BM2-Net training script.

Usage:
    python -m train.train
    python -m train.train --epochs 200 --batch_size 8 --lr 3e-4
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from train.config import TrainConfig
from train.dataset import build_dataloaders
from train.model import build_model
from train.utils import (
    denormalize_measurements,
    load_measurement_stats,
    save_checkpoint,
    set_seed,
    setup_logger,
)


def parse_args() -> TrainConfig:
    """Parse CLI arguments and override config defaults."""
    parser = argparse.ArgumentParser(description="Train BM2-Net")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable mixed precision training")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    args = parser.parse_args()
    config = TrainConfig()

    # Override config with any provided CLI args
    for key, value in vars(args).items():
        if key == "no_amp":
            if value:
                config.mixed_precision = False
        elif key == "resume":
            pass  # handled separately
        elif value is not None and hasattr(config, key):
            setattr(config, key, value)

    return config, args.resume


def build_optimizer_and_scheduler(model: nn.Module, config: TrainConfig):
    """Create AdamW optimizer and cosine annealing scheduler with warmup."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # Cosine annealing with warmup
    # During warmup, linearly increase LR from 0 to config.lr
    warmup_steps = config.warmup_epochs
    total_steps = config.epochs

    def lr_lambda(epoch):
        if epoch < warmup_steps:
            return (epoch + 1) / warmup_steps
        else:
            import math
            progress = (epoch - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    measurement_stats: dict,
    measurement_names: list,
    use_amp: bool = True,
) -> dict:
    """
    Evaluate model on a dataset.

    Returns:
        dict with keys: 'loss', 'mae_overall', 'mae_per_measurement'
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    num_batches = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        with autocast(device_type=device.type, enabled=use_amp):
            preds = model(inputs)
            loss = criterion(preds, targets)

        total_loss += loss.item()
        num_batches += 1

        # De-normalize for MAE in cm
        preds_cm = denormalize_measurements(preds, measurement_stats, measurement_names)
        targets_cm = denormalize_measurements(targets, measurement_stats, measurement_names)

        all_preds.append(preds_cm.cpu())
        all_targets.append(targets_cm.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Per-measurement MAE (in cm)
    mae_per = (all_preds - all_targets).abs().mean(dim=0)
    mae_overall = mae_per.mean().item()

    mae_dict = {
        name: round(mae_per[i].item(), 3)
        for i, name in enumerate(measurement_names)
    }

    return {
        "loss": total_loss / max(num_batches, 1),
        "mae_overall": round(mae_overall, 3),
        "mae_per_measurement": mae_dict,
    }


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    config: TrainConfig,
    epoch: int,
    logger,
) -> float:
    """Train for one epoch. Returns average training loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(
        dataloader,
        desc=f"Epoch {epoch+1}/{config.epochs}",
        leave=False,
    )

    for inputs, targets in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        with autocast(device_type=device.type, enabled=config.mixed_precision):
            preds = model(inputs)
            loss = criterion(preds, targets)

        if config.mixed_precision:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def main():
    config, resume_path = parse_args()

    # Setup
    set_seed(config.seed)
    logger = setup_logger("train", config.log_path)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info("BM2-Net Training")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {config.epochs}, Batch size: {config.batch_size}")
    logger.info(f"LR: {config.lr}, Weight decay: {config.weight_decay}")
    logger.info(f"d_model: {config.d_model}, n_heads: {config.n_heads}")
    logger.info(f"Mixed precision: {config.mixed_precision}")

    # ── Data ───────────────────────────────────────────────────────────
    logger.info("Loading datasets...")
    train_loader, testA_loader, testB_loader = build_dataloaders(config)
    logger.info(f"Train: {len(train_loader.dataset)} samples, "
                f"{len(train_loader)} batches")
    logger.info(f"TestA: {len(testA_loader.dataset)} samples")
    logger.info(f"TestB: {len(testB_loader.dataset)} samples")

    # Load measurement stats for de-normalization
    measurement_stats = load_measurement_stats(Path(config.data_root))

    # ── Model ──────────────────────────────────────────────────────────
    logger.info("Building model...")
    model = build_model(config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {num_params:,}")

    # ── Training setup ─────────────────────────────────────────────────
    criterion = nn.MSELoss()
    optimizer, scheduler = build_optimizer_and_scheduler(model, config)
    scaler = GradScaler(enabled=config.mixed_precision)
    best_mae = float("inf")
    start_epoch = 0

    # Resume from checkpoint
    if resume_path:
        from train.utils import load_checkpoint
        logger.info(f"Resuming from {resume_path}")
        ckpt_info = load_checkpoint(
            Path(resume_path), model, optimizer, scheduler, scaler
        )
        start_epoch = ckpt_info["epoch"] + 1
        best_mae = ckpt_info["best_mae"]
        logger.info(f"Resumed at epoch {start_epoch}, best MAE: {best_mae:.3f}")

    # ── Training loop ──────────────────────────────────────────────────
    logger.info("Starting training...")
    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler, config, epoch, logger,
        )

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch+1}/{config.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Evaluate periodically
        if (epoch + 1) % config.eval_every == 0 or epoch == config.epochs - 1:
            eval_results = evaluate(
                model, testA_loader, criterion, device,
                measurement_stats, config.measurement_names,
                use_amp=config.mixed_precision,
            )

            logger.info(
                f"  TestA — Loss: {eval_results['loss']:.4f} | "
                f"MAE: {eval_results['mae_overall']:.3f} cm"
            )
            for name, mae in eval_results["mae_per_measurement"].items():
                logger.info(f"    {name}: {mae:.3f} cm")

            # Save best model
            if eval_results["mae_overall"] < best_mae:
                best_mae = eval_results["mae_overall"]
                save_checkpoint(
                    config.checkpoint_path / "best_model.pt",
                    model, optimizer, scheduler, epoch, best_mae, scaler,
                )
                logger.info(f"  ★ New best MAE: {best_mae:.3f} cm — saved!")

        # Periodic checkpoint
        if (epoch + 1) % config.save_every == 0:
            save_checkpoint(
                config.checkpoint_path / f"epoch_{epoch+1}.pt",
                model, optimizer, scheduler, epoch, best_mae, scaler,
            )

    # ── Final save ─────────────────────────────────────────────────────
    save_checkpoint(
        config.checkpoint_path / "final_model.pt",
        model, optimizer, scheduler, config.epochs - 1, best_mae, scaler,
    )

    logger.info("=" * 60)
    logger.info(f"Training complete! Best TestA MAE: {best_mae:.3f} cm")
    logger.info(f"Checkpoints saved to: {config.checkpoint_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
