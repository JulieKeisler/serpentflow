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


def integrate_and_store(
    dataloader,
    model_to,
    t_span_to=(0.0, 1.0),
    device="cuda",
    filename=None
):
    """
    Generate samples using ODE integration and store results to disk.

    The model is assumed to be trained using Flow Matching / score-based dynamics.

    Args:
        dataloader (DataLoader): yields dicts with key "noisy"
        model_to (EMA): EMA-wrapped model (use model_to.model for inference)
        t_span_to (tuple): integration time interval
        device (str): device to run inference on
        filename (str or None): output path for storing results

    Returns:
        torch.Tensor: generated samples (N x C x H x W)
    """

    model = model_to.model.to(device)
    model.eval()

    results = []

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):

            # Input: shared structure + noise
            x_state = data["noisy"].to(device)

            # Solve ODE backwards from noise to data
            sample = odeint_torchdiffeq_adaptive(
                model,
                x_state,
                t_span=t_span_to,
                device=device
            )

            # Map output from [-1, 1] to [0, 1]
            sample = (sample + 1.0) / 2.0

            results.append(sample.cpu())

    results = torch.cat(results, dim=0)

    # Save to disk if requested
    if filename is not None:
        torch.save(results, filename)
        print(f"[SerpentFlow] Results saved to: {filename}")

    return results
