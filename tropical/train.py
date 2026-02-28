from __future__ import annotations

import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tropical.config import TropicalConfig
from tropical.data import TranscriptDataset
from tropical.model import Tropical


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _get_lr(step: int, config: TropicalConfig) -> float:
    """Linear warmup + cosine decay schedule."""
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    if step >= config.max_steps:
        return config.min_lr
    progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_lr + (config.learning_rate - config.min_lr) * cosine


def _freeze_for_stage(model: Tropical, stage: int) -> None:
    """Freeze parameters based on training stage."""
    if stage == 1:
        # Freeze protein encoder, TE conditioner, cross-attention, adaLN for cross-attention
        for name, param in model.named_parameters():
            if any(k in name for k in [
                "protein_encoder", "te_conditioner", "cross_attn", "adaln_ca",
            ]):
                param.requires_grad = False
    elif stage == 2:
        # Freeze TE conditioner only
        for param in model.te_conditioner.parameters():
            param.requires_grad = False
    # Stage 3: all trainable (no freezing)


def _save_checkpoint(
    model: Tropical,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: TropicalConfig,
) -> None:
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"stage{config.stage}_step{step}.pt"
    torch.save(
        {
            "step": step,
            "stage": config.stage,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        path,
    )
    print(f"Saved checkpoint: {path}")


@torch.no_grad()
def _evaluate(
    model: Tropical, val_loader: DataLoader, device: torch.device, max_batches: int = 50
) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(
            idx=batch["input_ids"],
            protein_ids=batch["protein_ids"],
            protein_pad_mask=batch["protein_pad_mask"],
            te_values=batch["te_values"],
            te_mask=batch["te_mask"],
        )
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), batch["labels"].view(-1), ignore_index=-100
        )
        total_loss += loss.item()
        n += 1
    model.train()
    return total_loss / max(n, 1)


def train(config: TropicalConfig) -> None:
    device = _get_device()
    print(f"Device: {device}")
    print(f"Stage: {config.stage}")

    # Build model
    model = Tropical(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")

    # Resume from checkpoint
    if config.resume_from:
        print(f"Resuming from: {config.resume_from}")
        ckpt = torch.load(config.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

    # Stage-specific freezing (after loading weights)
    _freeze_for_stage(model, config.stage)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    # Dataset + dataloader
    train_dataset = TranscriptDataset(config, split="train")
    val_dataset = TranscriptDataset(config, split="val")
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=min(2, os.cpu_count() or 1),
        pin_memory=(device.type == "cuda"),
    )

    # Optimizer
    param_groups = [
        {"params": [p for p in model.parameters() if p.requires_grad and p.dim() >= 2],
         "weight_decay": config.weight_decay},
        {"params": [p for p in model.parameters() if p.requires_grad and p.dim() < 2],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=config.learning_rate, betas=(0.9, 0.95))

    # Resume optimizer state
    start_step = 0
    if config.resume_from:
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt.get("step", 0)
        print(f"Resuming from step {start_step}")

    # Training loop
    model.train()
    data_iter = iter(train_loader)
    t0 = time.time()

    for step in range(start_step, config.max_steps):
        # Get next batch (cycle through data)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        batch = {k: v.to(device) for k, v in batch.items()}

        # LR schedule
        lr = _get_lr(step, config)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Forward
        logits = model(
            idx=batch["input_ids"],
            protein_ids=batch["protein_ids"],
            protein_pad_mask=batch["protein_pad_mask"],
            te_values=batch["te_values"],
            te_mask=batch["te_mask"],
        )
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), batch["labels"].view(-1), ignore_index=-100
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        # Logging
        if step % config.log_interval == 0:
            dt = time.time() - t0
            print(f"step {step:>6d} | loss {loss.item():.4f} | lr {lr:.2e} | {dt:.1f}s")
            t0 = time.time()

        # Evaluation
        if step > 0 and step % config.eval_interval == 0:
            val_loss = _evaluate(model, val_loader, device)
            print(f"step {step:>6d} | val_loss {val_loss:.4f}")

        # Save checkpoint
        if step > 0 and step % config.save_interval == 0:
            _save_checkpoint(model, optimizer, step, config)

    # Final save
    _save_checkpoint(model, optimizer, config.max_steps, config)
    print("Training complete.")


if __name__ == "__main__":
    from tropical.cli import app

    app()
