"""
Inference utilities for SerpentFlow.

This module implements numerical integration of the learned conditional vector field
using adaptive ODE solvers in order to generate samples from the trained generative model.

Main function:
    - integrate_and_store: integrates ODE trajectories conditionally on low-frequency inputs
      and stores generated samples to disk.
"""

import torch
from utils.inference_utils import odeint_torchdiffeq_adaptive


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
    method="dopri5"
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

            # Solve ODE backward from noise to data
            sample = odeint_torchdiffeq_adaptive(
                model,
                x_state,
                t_span=t_span_to,
                device=device,
                method=method
            )

            # Map output from [-1, 1] to [0, 1]
            sample = (sample + 1.0) / 2.0
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
