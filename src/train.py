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
from utils.inference_utils import quick_si_sde_sample, save_channels_subplot
from utils.training_utils import train_one_epoch, NativeScalerWithGradNormCount, train_one_epoch_si_sde, vel_loss, eta_loss
import time

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
    sde=False,
    path=CondOTProbPath(),
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
        pin_memory=True,
        num_workers=8,
        persistent_workers=True
    )
    print(f"DataLoader ready with {len(dl)} iterations per epoch. Batch size: {batch_size}")

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
    
    for epoch in range(start_epoch, epochs + 1):
        s_time = time.time()
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


        print(f"[Epoch {epoch:03d}/{epochs}] Loss: {epoch_loss:.6f}, Time: {time.time()-s_time:.2f} sec")

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

def train_si_sde(
    vel_model,
    denoise_model,
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
    num_workers=8,
    num_steps=50,
    path=CondOTProbPath(),
    skewed_timesteps=True,
    **args
):
    """
    Train velocity + denoiser models for SDE stochastic interpolants.
    
    Args:
        vel_model (EMA): EMA-wrapped model predicting velocity
        denoise_model (EMA): EMA-wrapped model predicting eta_z
        dataset (Dataset)
        name (str): experiment name for saving checkpoints
        ckpt (dict, optional): checkpoint for resuming
        ...
    
    Returns:
        Tuple[EMA, EMA]: trained (vel_model, denoise_model)
    """

    device = torch.device(device)
    os.makedirs(save_dir, exist_ok=True)
    best_ckpt_path = os.path.join(save_dir, f"si_sde_{name}_best.pth")

    # ---------------------------
    # Move models to device
    # ---------------------------
    vel_model = vel_model.to(device)
    vel_model.train(True)
    denoise_model = denoise_model.to(device)
    denoise_model.train(True)

    # ---------------------------
    # DataLoader
    # ---------------------------
    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True
    )
    print(f"[SI-SDE] DataLoader ready: {len(dl)} iterations per epoch, batch size={batch_size}")
    # ---------------------------
    # Optimizers
    # ---------------------------
    optimizer = torch.optim.AdamW(
        list(vel_model.parameters()) + list(denoise_model.parameters()),
        lr=lr,
        betas=betas
    )

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
    lr_schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---------------------------
    # Resume checkpoint
    # ---------------------------
    start_epoch = 1
    best_loss = float("inf")
    if ckpt is not None:
        print(f"[SI-SDE] Resuming from checkpoint")
        vel_model.load_state_dict(ckpt["vel_model"])
        denoise_model.load_state_dict(ckpt["denoise_model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_schedule.load_state_dict(ckpt["lr_schedule"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("loss", float("inf"))
        torch.cuda.empty_cache()

    # ---------------------------
    # Loss scaler
    # ---------------------------
    loss_scaler = NativeScalerWithGradNormCount()

    # ---------------------------
    # Training loop
    # ---------------------------
    print("[SI-SDE] Starting training...")
    for epoch in range(start_epoch, epochs + 1):
        s_time = time.time()

        epoch_loss = train_one_epoch_si_sde(
            vel_model=vel_model,
            denoise_model=denoise_model,
            data_loader=dl,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            path=path,
            mask=mask,
            accum_iter=accum_iter,
            skewed_timesteps=skewed_timesteps
        )

        print(f"[Epoch {epoch:03d}/{epochs}] Loss={epoch_loss:.6f}, Time={time.time()-s_time:.2f}s")

        # ---------------------------
        # Save best checkpoint
        # ---------------------------
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            vel_model.eval()
            denoise_model.eval()

            ckpt_dict = {
                "vel_model": vel_model.state_dict(),
                "denoise_model": denoise_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_schedule": lr_schedule.state_dict(),
                "epoch": epoch,
                "loss": best_loss
            }
            torch.save(ckpt_dict, best_ckpt_path)
            print(f"[SI-SDE] ✅ New best checkpoint saved: {best_ckpt_path} (loss={best_loss:.6f})")

            vel_model.train(True)
            denoise_model.train(True)

        # Step scheduler
        lr_schedule.step()

    print("[SI-SDE] Training complete. Best loss = {:.6f}".format(best_loss))

    run_check=False
    if run_check:
        # =====================================================
        # 🔍 Quick sampling sanity check from pure noise
        # =====================================================
        print("[SI-SDE] Running quick sampling sanity check...")

        vel_model.eval()
        denoise_model.eval()
        for steps in range(1, 10):
            sample = dataset[0]["noisy"].to(device)
            stats = dataset[0].get("stats", None)
            if stats is not None:
                stats = stats.unsqueeze(0).to(device)
            
            x_noise = quick_si_sde_sample(
                vel_model=vel_model,
                denoise_model=denoise_model,
                path=path,
                x0=sample.unsqueeze(0),
                stats=stats,
                device=device,
                num_steps=steps*100,
                direction="forward"
            )

            png_path_noise = os.path.join(save_dir, f"si_sde_{name}_quick_forward_{steps}.png")
            save_channels_subplot(x_noise, png_path_noise)
            print(f"[SI-SDE] ✅ Quick forward sample saved to {png_path_noise}")

            sample = dataset[0]["data"].to(device)
            stats = dataset[0].get("stats", None)
            if stats is not None:
                stats = stats.unsqueeze(0).to(device)
            
            x_noise = quick_si_sde_sample(
                vel_model=vel_model,
                denoise_model=denoise_model,
                path=path,
                x0=sample.unsqueeze(0),
                stats=stats,
                device=device,
                num_steps=steps*100,
                direction="backward"
            )

            png_path_noise = os.path.join(save_dir, f"si_sde_{name}_quick_backward_{steps}.png")
            save_channels_subplot(x_noise, png_path_noise)
            print(f"[SI-SDE] ✅ Quick backward sample saved to {png_path_noise}")

    return vel_model, denoise_model

