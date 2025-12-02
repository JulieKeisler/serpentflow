"""
Inference utilities for SerpentFlow.

This module provides:
    - adaptive ODE integration via torchdiffeq
    - sampling and visualization helpers

Functions:
    - odeint_torchdiffeq_adaptive: integrate learned vector field
    - generate_grid: generate and store image samples
"""

import torch
import os
from torchdiffeq import odeint
from utils.plots import to_grid_and_save


@torch.no_grad()
def odeint_torchdiffeq_adaptive(
    model,
    x,
    t_span=(0.0, 1.0),
    device="cuda",
    args={"rtol": 1e-3, "atol": 1e-4},
    method="dopri5"
):
    """
    Integrate the learned continuous flow using an adaptive ODE solver.

    Args:
        model (torch.nn.Module): trained vector field model
        x (torch.Tensor): initial condition (N, C, H, W)
        t_span (tuple): integration time interval
        device (str): computation device
        args (dict): ODE solver precision parameters
        method (str): integration method

    Returns:
        torch.Tensor: integrated sample (final state)
    """

    model = model.to(device).eval()
    x = x.to(device)
    t0, t1 = t_span

    time_steps = torch.tensor([t0, t1], device=device)

    def f(t, x_state):
        # Time conditioning expanded to batch size
        t = t.expand(x_state.size(0))
        return model(x_state, t)

    x_out = odeint(f, x, time_steps, method=method, **args)[-1]
    return x_out


@torch.no_grad()
def generate_grid(
    model,
    shape,
    n_samples=16,
    device="cuda",
    save_dir="samples",
    prefix="sample",
    x=None,
    ds=None
):
    """
    Generate samples and save them as image grids.

    Args:
        model (torch.nn.Module): trained flow model
        shape (tuple): sample shape (C, H, W)
        n_samples (int): number of samples
        device (str): computation device
        save_dir (str): directory to store images
        prefix (str): filename prefix
        x (torch.Tensor or None): optional conditioning input
        ds (Dataset or None): optional dataset (unused here but kept for compatibility)

    Returns:
        torch.Tensor: generated samples
    """

    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device).eval()

    # Initial condition
    if x is None:
        x = torch.randn((n_samples, *shape), device=device)
    else:
        x = x.to(device)

    # Integrate the ODE
    x_out = odeint_torchdiffeq_adaptive(model, x, device=device)

    # Map from [-1, 1] to [0, 1]
    x_vis = (x_out + 1) / 2

    # Save results
    if x_vis.shape[1] == 2:
        to_grid_and_save(x_vis[:, :1], os.path.join(save_dir, f"{prefix}_grid_0.png"))
        to_grid_and_save(x_vis[:, 1:], os.path.join(save_dir, f"{prefix}_grid_1.png"))
    else:
        to_grid_and_save(x_vis, os.path.join(save_dir, f"{prefix}_grid.png"))

    print(f"[SerpentFlow] Saved grid to {save_dir}/{prefix}_grid.png")

    return x_out
