"""
Inference utilities for SerpentFlow.

This module implements numerical integration of the learned conditional vector field
using adaptive ODE solvers in order to generate samples from the trained generative model.

Main function:
    - integrate_and_store: integrates ODE trajectories conditionally on low-frequency inputs
      and stores generated samples to disk.
"""

from typing import Tuple
import torch
from utils.inference_utils import odeint_torchdiffeq_adaptive, save_channels_subplot


import torch
import os

def integrate_and_store(
    dataloader,
    model_to,
    t_span_to=(0.0, 1.0),
    device="cuda",
    filename=None,
    chunk_size=100,
    temp_dir="temp_results",
    method="dopri5",
    mask=None
):
    """
    Generate samples using ODE integration and store results to disk in chunks.
    All chunks are concatenated into one file at the end.

    Args:
        dataloader (DataLoader): yields dicts with key "noisy"
        model_to (EMA): EMA-wrapped model (use model_to.model for inference)
        t_span_to (tuple): integration time interval
        device (str): device to run inference on
        filename (str): final output file path
        chunk_size (int): number of samples per temporary file
        temp_dir (str): directory to store temporary chunks
    """

    # Create directory for temporary files if it doesn't exist
    os.makedirs(temp_dir, exist_ok=True)

    model = model_to.model.to(device)
    model.eval()

    buffer = []
    temp_files = []
    total_samples = 0
    file_idx = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):

            # Input: structured data + noise
            x_state = data["noisy"].to(device)
            stats = data.get('stats', None)
            if stats is not None:
                stats = stats.to(device)

            # Solve ODE backward from noise to data
            sample = odeint_torchdiffeq_adaptive(
                model,
                x_state,
                stats,
                t_span=t_span_to,
                device=device,
                method=method,
                mask=mask
            )

            # Map output from [-1, 1] to [0, 1]
            #sample = (sample + 1.0) / 2.0
            sample = sample.cpu()

            buffer.append(sample)
            total_samples += sample.shape[0]

            # Save a chunk when enough samples are accumulated
            current_count = sum(x.shape[0] for x in buffer)
            if current_count >= chunk_size:
                chunk = torch.cat(buffer, dim=0)

                tmp_path = os.path.join(temp_dir, f"{filename.split('/')[-1].split('.')[0]}_chunk_{file_idx:05d}.pt")
                torch.save(chunk, tmp_path)

                print(f"[SerpentFlow] Chunk saved: {tmp_path} ({chunk.shape[0]} samples) {batch_idx}/{len(dataloader)}")

                temp_files.append(tmp_path)
                file_idx += 1
                buffer = []  # Clear buffer

        # Save remaining samples
        if buffer:
            chunk = torch.cat(buffer, dim=0)
            tmp_path = os.path.join(temp_dir, f"{filename.split('/')[-1].split('.')[0]}_chunk_{file_idx:05d}.pt")
            torch.save(chunk, tmp_path)
            print(f"[SerpentFlow] Final chunk saved: {tmp_path} ({chunk.shape[0]} samples)")
            temp_files.append(tmp_path)

    # ===== FINAL CONCATENATION =====
    print("[SerpentFlow] Concatenating temporary files...")

    all_results = []
    for path in temp_files:
        all_results.append(torch.load(path, map_location="cpu"))

    final_tensor = torch.cat(all_results, dim=0)

    # Save final output
    if filename is not None:
        torch.save(final_tensor, filename)
        print(f"[SerpentFlow] Final result saved to: {filename}")

    # ===== OPTIONAL CLEANUP =====
    for path in temp_files:
        os.remove(path)

    os.rmdir(temp_dir)

    print(f"[SerpentFlow] Total samples generated: {final_tensor.shape[0]}")
    return final_tensor


def dual_integrate_and_store(
    dataloader,
    model_from,
    model_to,
    t_span_from=(1.0, 0.0),
    t_span_to=(0.0, 1.0),
    device="cuda",
    filename=None,
    chunk_size=100,
    temp_dir="temp_results",
    method="dopri5",
    mask=None
):
    """
    Generate samples using ODE integration and store results to disk in chunks.
    All chunks are concatenated into one file at the end.

    Args:
        dataloader (DataLoader): yields dicts with key "noisy"
        model_to (EMA): EMA-wrapped model (use model_to.model for inference)
        t_span_to (tuple): integration time interval
        device (str): device to run inference on
        filename (str): final output file path
        chunk_size (int): number of samples per temporary file
        temp_dir (str): directory to store temporary chunks
    """

    # Create directory for temporary files if it doesn't exist
    os.makedirs(temp_dir, exist_ok=True)

    model_from = model_from.model.to(device)
    model_from.eval()

    model_to = model_to.model.to(device)
    model_to.eval()

    buffer = []
    temp_files = []
    total_samples = 0
    file_idx = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):

            # Input: structured data + noise
            x_state = data["data"].to(device)
            stats = data.get('stats', None)
            if stats is not None:
                stats = stats.to(device)

            # Solve ODE backward from noise to data
            z = odeint_torchdiffeq_adaptive(
                model_from,
                x_state,
                stats,
                t_span=t_span_from,
                device=device,
                method=method,
                mask=mask
            )

            sample = odeint_torchdiffeq_adaptive(
                model_to,
                z,
                stats,
                t_span=t_span_to,
                device=device,
                method=method,
                mask=mask
            )
            sample = sample.cpu()

            buffer.append(sample)
            total_samples += sample.shape[0]

            # Save a chunk when enough samples are accumulated
            current_count = sum(x.shape[0] for x in buffer)
            if current_count >= chunk_size:
                chunk = torch.cat(buffer, dim=0)

                tmp_path = os.path.join(temp_dir, f"{filename.split('/')[-1].split('.')[0]}_chunk_{file_idx:05d}.pt")
                torch.save(chunk, tmp_path)

                print(f"[Dual FM] Chunk saved: {tmp_path} ({chunk.shape[0]} samples) {batch_idx}/{len(dataloader)}")

                temp_files.append(tmp_path)
                file_idx += 1
                buffer = []  # Clear buffer

        # Save remaining samples
        if buffer:
            chunk = torch.cat(buffer, dim=0)
            tmp_path = os.path.join(temp_dir, f"{filename.split('/')[-1].split('.')[0]}_chunk_{file_idx:05d}.pt")
            torch.save(chunk, tmp_path)
            print(f"[Dual FM] Final chunk saved: {tmp_path} ({chunk.shape[0]} samples)")
            temp_files.append(tmp_path)

    # ===== FINAL CONCATENATION =====
    print("[Dual FM] Concatenating temporary files...")

    all_results = []
    for path in temp_files:
        all_results.append(torch.load(path, map_location="cpu"))

    final_tensor = torch.cat(all_results, dim=0)

    # Save final output
    if filename is not None:
        torch.save(final_tensor, filename)
        print(f"[Dual FM] Final result saved to: {filename}")

    # ===== OPTIONAL CLEANUP =====
    for path in temp_files:
        os.remove(path)

    os.rmdir(temp_dir)

    print(f"[Dual FM] Total samples generated: {final_tensor.shape[0]}")
    return final_tensor

def dual_sde_sample_and_store(
    dataloader,
    vel_model_from,
    denoise_model_from,
    vel_model_to,
    denoise_model_to,
    path,
    device="cuda",
    filename=None,
    chunk_size=100,
    temp_dir="temp_results",
    num_steps=50,
):
    import os
    import torch

    os.makedirs(temp_dir, exist_ok=True)

    for m in [
        vel_model_from, denoise_model_from,
        vel_model_to, denoise_model_to
    ]:
        m.to(device).eval()

    buffer, temp_files = [], []
    file_idx = 0

    with torch.no_grad():
        for en, data in enumerate(dataloader):
            x = data["data"].to(device)
            stats = data.get("stats", None)
            if stats is not None:
                stats = stats.to(device)

            batch_size = x.shape[0]

            # ==================================================
            # Forward SDE (1 → 0) : Domaine A → Bruit
            # ==================================================
            t_vals = torch.linspace(1.0 , 0.0, num_steps, device=device)

            x_t = x.clone()

            for i in range(num_steps - 1):
                t = t_vals[i].expand(batch_size)
                dt = t_vals[i + 1] - t_vals[i]

                with torch.amp.autocast("cuda"):
                    b_hat = vel_model_from(x_t, t, stats=stats)
                    s_hat = denoise_model_from(x_t, t, stats=stats)

                gamma = path.gamma_fn(t).view(-1, 1, 1, 1)
                score = -s_hat/gamma
                eps = path.epsilon(t).view(-1, 1, 1, 1)
                sqrt2eps = torch.sqrt(2.0 * eps)

                drift = b_hat - eps * score
                dW = torch.randn_like(x_t) * torch.sqrt(torch.abs(dt))

                x_t = x_t + dt * drift + sqrt2eps * dW

            z = x_t.clone()

            # ==================================================
            # Reverse SDE (0 → 1) : Bruit → Domaine B
            # ==================================================
            
            x_t = z
            t_vals = torch.linspace(0.0, 1.0, num_steps, device=device)

            for i in range(num_steps - 1):
                t = t_vals[i].expand(batch_size)
                dt = t_vals[i + 1] - t_vals[i]

                with torch.amp.autocast("cuda"):
                    b_hat = vel_model_to(x_t, t, stats=stats)
                    s_hat = denoise_model_to(x_t, t, stats=stats)
                gamma = path.gamma_fn(t).view(-1, 1, 1, 1)
                score = -s_hat/gamma
                eps = path.epsilon(t).view(-1, 1, 1, 1)
                sqrt2eps = torch.sqrt(2.0 * eps)
                drift = b_hat - eps * score
                dW = torch.randn_like(x_t) * torch.sqrt(dt)

                x_t = x_t + dt * drift + sqrt2eps * dW

            buffer.append(x_t.cpu())

            if sum(b.shape[0] for b in buffer) >= chunk_size:
                chunk = torch.cat(buffer, dim=0)
                path_out = os.path.join(
                    temp_dir, f"chunk_{file_idx:05d}.pt"
                )
                torch.save(chunk, path_out)
                temp_files.append(path_out)
                buffer, file_idx = [], file_idx + 1

        if buffer:
            chunk = torch.cat(buffer, dim=0)
            path_out = os.path.join(
                temp_dir, f"chunk_{file_idx:05d}.pt"
            )
            torch.save(chunk, path_out)
            temp_files.append(path_out)

    final = torch.cat([torch.load(p) for p in temp_files])
    if filename is not None:
        torch.save(final, filename)

    for p in temp_files:
        os.remove(p)
    os.rmdir(temp_dir)

    print(f"[Dual SDE] Total samples generated: {final.shape[0]}")
    return final

def sde_sample_and_store(
    dataloader,
    vel_model,
    denoise_model,
    path,
    device="cuda",
    filename=None,
    chunk_size=100,
    temp_dir="temp_results",
    num_steps=50,
):
    import os
    import torch

    os.makedirs(temp_dir, exist_ok=True)

    for m in [vel_model, denoise_model]:
        m.to(device).eval()

    buffer, temp_files = [], []
    file_idx = 0

    with torch.no_grad():
        for en, data in enumerate(dataloader):
            x_t = data["noisy"].to(device)
            stats = data.get("stats", None)
            if stats is not None:
                stats = stats.to(device)

            batch_size = x_t.shape[0]

            # =========================
            # Reverse SDE (0 → 1)
            # =========================
            t_vals = torch.linspace(0.0, 1.0, num_steps, device=device)

            for i in range(num_steps - 1):
                t = t_vals[i].expand(batch_size)
                dt = t_vals[i + 1] - t_vals[i]

                with torch.amp.autocast("cuda"):
                    b_hat = vel_model(x_t, t, stats=stats)
                    s_hat = denoise_model(x_t, t, stats=stats)

                gamma = path.gamma_fn(t).view(-1, 1, 1, 1)
                score = -s_hat/gamma
                eps = path.epsilon(t).view(-1, 1, 1, 1)
                sqrt2eps = torch.sqrt(2.0 * eps)

                drift = b_hat + eps * score
                dW = torch.randn_like(x_t) * torch.sqrt(dt)

                x_t = x_t + dt * drift + sqrt2eps * dW

            buffer.append(x_t.cpu())

            if sum(b.shape[0] for b in buffer) >= chunk_size:
                chunk = torch.cat(buffer, dim=0)
                path_out = os.path.join(
                    temp_dir, f"chunk_{file_idx:05d}.pt"
                )
                torch.save(chunk, path_out)
                temp_files.append(path_out)
                buffer, file_idx = [], file_idx + 1

        if buffer:
            chunk = torch.cat(buffer, dim=0)
            path_out = os.path.join(
                temp_dir, f"chunk_{file_idx:05d}.pt"
            )
            torch.save(chunk, path_out)
            temp_files.append(path_out)

    final = torch.cat([torch.load(p) for p in temp_files])
    if filename is not None:
        torch.save(final, filename)

    for p in temp_files:
        os.remove(p)
    os.rmdir(temp_dir)

    print(f"[SDE] Total samples generated: {final.shape[0]}")
    return final

def sde_sample_single(
    x0,
    stats,
    vel_model,
    denoise_model,
    path,
    device="cuda",
    num_steps=50,
):
    import torch

    vel_model.to(device).eval()
    denoise_model.to(device).eval()

    x_t = x0.to(device)
    batch_size = x_t.shape[0]

    t_vals = torch.linspace(0.0, 1.0, num_steps, device=device)

    with torch.no_grad():
        for i in range(num_steps - 1):
            t = t_vals[i].expand(batch_size)
            dt = t_vals[i + 1] - t_vals[i]

            b_hat = vel_model(x_t, t, stats=stats)
            s_hat = denoise_model(x_t, t, stats=stats)

            gamma = path.gamma_fn(t).view(-1, 1, 1, 1)
            score = -s_hat / gamma
            eps = path.epsilon(t).view(-1, 1, 1, 1)
            sqrt2eps = torch.sqrt(2.0 * eps)

            drift = b_hat + eps * score
            dW = torch.randn_like(x_t) * torch.sqrt(dt)

            x_t = x_t + dt * drift + sqrt2eps * dW

    return x_t.cpu()

def save_climate_subplot(x0, sample, save_path="comparison.png", var_names=None):
    import torch
    import matplotlib.pyplot as plt

    # On prend la première image si batch > 1
    x0 = x0[0].detach().cpu()
    sample = sample[0].detach().cpu()

    C, H, W = x0.shape

    fig, axs = plt.subplots(C, 2, figsize=(8, 3 * C))

    # Si une seule variable → axs devient 1D
    if C == 1:
        axs = axs.reshape(1, 2)

    for c in range(C):
        x0_c = x0[c]
        sample_c = sample[c]

        # Optionnel : normalisation par variable (meilleure lecture visuelle)
        vmin = min(x0_c.min(), sample_c.min())
        vmax = max(x0_c.max(), sample_c.max())

        axs[c, 0].imshow(x0_c, cmap="viridis", vmin=vmin, vmax=vmax)
        axs[c, 0].axis("off")

        axs[c, 1].imshow(sample_c, cmap="viridis", vmin=vmin, vmax=vmax)
        axs[c, 1].axis("off")

        if var_names is not None:
            axs[c, 0].set_ylabel(var_names[c], rotation=90, fontsize=10)

        if c == 0:
            axs[c, 0].set_title("x0")
            axs[c, 1].set_title("sample")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)

def dual_sde_sample_single(
    x0,
    stats,
    vel_model_from,
    denoise_model_from,
    vel_model_to,
    denoise_model_to,
    path,
    device="cuda",
    num_steps=50,
):
    import torch

    # mise en eval
    for m in [
        vel_model_from, denoise_model_from,
        vel_model_to, denoise_model_to
    ]:
        m.to(device).eval()

    x = x0.to(device)

    if stats is not None:
        stats = stats.to(device)

    batch_size = x.shape[0]

    with torch.no_grad():

        # =====================================
        # 1️⃣ Forward SDE (A → bruit)
        # =====================================
        t_vals = torch.linspace(1.0, 0.0, num_steps, device=device)
        x_t = x.clone()

        for i in range(num_steps - 1):
            t = t_vals[i].expand(batch_size)
            dt = t_vals[i + 1] - t_vals[i]

            b_hat = vel_model_from(x_t, t, stats=stats)
            s_hat = denoise_model_from(x_t, t, stats=stats)

            gamma = path.gamma_fn(t).view(-1, 1, 1, 1)
            score = -s_hat / gamma
            eps = path.epsilon(t).view(-1, 1, 1, 1)
            sqrt2eps = torch.sqrt(2.0 * eps)

            drift = b_hat - eps * score
            dW = torch.randn_like(x_t) * torch.sqrt(torch.abs(dt))

            x_t = x_t + dt * drift + sqrt2eps * dW

        z = x_t.clone()

        # =====================================
        # 2️⃣ Reverse SDE (bruit → B)
        # =====================================
        t_vals = torch.linspace(0.0, 1.0, num_steps, device=device)
        x_t = z

        for i in range(num_steps - 1):
            t = t_vals[i].expand(batch_size)
            dt = t_vals[i + 1] - t_vals[i]

            b_hat = vel_model_to(x_t, t, stats=stats)
            s_hat = denoise_model_to(x_t, t, stats=stats)

            gamma = path.gamma_fn(t).view(-1, 1, 1, 1)
            score = -s_hat / gamma
            eps = path.epsilon(t).view(-1, 1, 1, 1)
            sqrt2eps = torch.sqrt(2.0 * eps)

            drift = b_hat - eps * score
            dW = torch.randn_like(x_t) * torch.sqrt(dt)

            x_t = x_t + dt * drift + sqrt2eps * dW

    return x_t.cpu()
