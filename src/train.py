"""
Training logic for Flow Matching in SerpentFlow.

This module implements:
    - optimizer setup
    - learning rate scheduling (warmup + cosine decay)
    - checkpointing (best model based on training loss)
    - support for resuming training

Main function:
    train_flow_matching(...)
"""

import os
import math
import torch
from torch.utils.data import DataLoader
from flow_matching.path import CondOTProbPath
from utils.training_utils import train_one_epoch, NativeScalerWithGradNormCount


def train_flow_matching(
    model,
    dataset,
    name,
    ckpt=None,
    epochs=200,
    batch_size=64,
    lr=1e-4,
    accum_iter=1,
    device="cuda",
    betas=(0.9, 0.95),
    save_dir="checkpoints",
    mask=None,
    **args
):
    """
    Train a Flow Matching model with cosine learning rate scheduling and checkpointing.

    Args:
        model (EMA): EMA-wrapped model
        dataset (Dataset): SerpentFlowDataset
        name (str): experiment identifier
        ckpt (dict or None): checkpoint for resuming training
        epochs (int): number of training epochs
        batch_size (int): batch size
        lr (float): learning rate
        accum_iter (int): gradient accumulation steps
        device (str): device identifier
        betas (tuple): AdamW betas
        save_dir (str): checkpoint directory

    Returns:
        EMA: trained model
    """

    # ---------------------------
    # Directories and device
    # ---------------------------
    os.makedirs(save_dir, exist_ok=True)
    best_ckpt = os.path.join(save_dir, f"fm_{name}_best.pth")
    device = torch.device(device)

    model = model.to(device)
    model.train(True)

    # ---------------------------
    # DataLoader
    # ---------------------------
    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    # ---------------------------
    # Optimizer
    # ---------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas)

    # ---------------------------
    # Scheduler (warmup + cosine)
    # ---------------------------
    total_steps = epochs * len(dl) // accum_iter
    warmup_steps = 5000

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    loss_scaler = NativeScalerWithGradNormCount()
    path = CondOTProbPath()

    # ---------------------------
    # Resume training if needed
    # ---------------------------
    if ckpt is not None:
        print("[INFO] Resuming training from checkpoint.")
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt["loss"]
        lr_schedule = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda,
            last_epoch=start_epoch - 1
        )
        torch.cuda.empty_cache()
    else:
        start_epoch = 1
        best_loss = float("inf")
        lr_schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Try loading scheduler state (if available)
    if ckpt is not None:
        try:
            lr_schedule.load_state_dict(ckpt["lr_schedule"])
        except KeyError:
            print("[WARNING] LR scheduler state not found. Restarting scheduler.")
            best_loss = float("inf")

    # ---------------------------
    # Training loop
    # ---------------------------
    print("Starting Flow Matching training...")
    
    mask = None
    for epoch in range(start_epoch, epochs + 1):

        epoch_loss = train_one_epoch(
            model=model,
            data_loader=dl,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            device=device,
            epoch=epoch,
            accum_iter=accum_iter,
            mask=mask,
            loss_scaler=loss_scaler,
            path=path
        )


        print(f"[Epoch {epoch:03d}/{epochs}] Loss: {epoch_loss:.6f}")

        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            model.eval()

            torch.save({
                "ema": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_schedule": lr_schedule.state_dict(),
                "epoch": epoch,
                "loss": best_loss
            }, best_ckpt)

            print(f"[INFO] ✅ New best model saved (loss={best_loss:.6f})")

            torch.cuda.empty_cache()
            model.train(True)

    print(f"[INFO] Training complete. Best loss = {best_loss:.6f}")
    return model
