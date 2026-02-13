"""
Inference utilities for SerpentFlow.

This module provides:
    - adaptive ODE integration via torchdiffeq
    - sampling and visualization helpers

Functions:
    - odeint_torchdiffeq_adaptive: integrate learned vector field
    - generate_grid: generate and store image samples
"""

from matplotlib import pyplot as plt
import torch
import os
from torchdiffeq import odeint

from utils.training_utils import SDEPath


@torch.no_grad()
def odeint_torchdiffeq_adaptive(
    model,
    x,
    stats=None,
    t_span=(0.0, 1.0),
    device="cuda",
    args={"rtol": 1e-3, "atol": 1e-4},
    method="dopri5",
    mask=None
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
    if mask is not None:
        x = torch.nan_to_num(x) * mask
    if stats is not None:
        stats = stats.to(device)
    t0, t1 = t_span

    time_steps = torch.tensor([t0, t1], device=device)

    def f(t, x_state):
        # Time conditioning expanded to batch size
        t = t.expand(x_state.size(0))
        if mask is not None:
            x_state = torch.nan_to_num(x_state) * mask
            x_state = torch.cat([x_state, mask.repeat(x_state.shape[0], 1, 1, 1)], dim=1)
        v = model(x_state, t, stats=stats)
        if mask is not None:
            v = v * mask
        return v


    x_out = odeint(f, x, time_steps, method=method, **args)[-1]
    return x_out

def save_channels_subplot(x, path, channel_names=None, cmap="viridis"):
    """
    x : Tensor [B, C, H, W] ou [C, H, W]
    Sauvegarde la première image du batch avec 1 subplot par channel
    """
    if x.dim() == 4:
        x = x[0]  # première carte météo du batch

    x = x.cpu()
    C = x.shape[0]

    fig, axes = plt.subplots(1, C, figsize=(4 * C, 4))
    if C == 1:
        axes = [axes]

    for c in range(C):
        im = axes[c].imshow(x[c], cmap=cmap)
        title = channel_names[c] if channel_names is not None else f"Channel {c}"
        axes[c].set_title(title)
        axes[c].axis("off")
        plt.colorbar(im, ax=axes[c], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def quick_si_sde_sample(
    vel_model,
    denoise_model,
    path: SDEPath,
    x0: torch.Tensor = None,
    stats: torch.Tensor = None,
    mask: torch.Tensor = None,
    device: str = "cuda",
    num_steps: int = 50,
    direction: str = "forward"
):
    """
    Sample from a stochastic interpolant SDE.
    x_t = I(t,x0,x1) + γ(t) z
    dX_t = (b_hat ± eps(t) * s_hat) dt + sqrt(2 eps(t)) dW_t
    """

    vel_model.eval()
    denoise_model.eval()
    x = x0
    if direction == "forward":
        t_vals = torch.linspace(0.0, 1.0, num_steps, device=device)
    else:
        t_vals = torch.linspace(1.0, 0.0, num_steps, device=device)
    for i in range(num_steps-1):
        t = t_vals[i].expand(x.shape[0])
        dt = t_vals[i + 1] - t_vals[i]

        x_input = x
        if mask is not None:
            x_input = torch.cat(
                [x, mask.repeat(x.shape[0], 1, 1, 1)],
                dim=1
            )

        with torch.amp.autocast("cuda"):
            b_hat = vel_model(x_input, t, stats=stats)
            s_hat = denoise_model(x_input, t, stats=stats)
        gamma = path.gamma_fn(t).view(-1, 1, 1, 1)
        score = -s_hat/gamma
        eps = path.epsilon(t).view(-1, 1, 1, 1)
        sqrt2eps = torch.sqrt(2.0 * eps)

        if direction == "forward":
            drift = b_hat + eps * score
        elif direction == "backward":
            drift = b_hat - eps * score
        else:
            raise ValueError(f"direction invalide : {direction}")

        dW = torch.randn_like(x) * torch.sqrt(torch.abs(dt))
        print(f'Step = {i}, t = {t.mean():.6f}, gamma = {gamma.mean():.6f}, s_hat = {s_hat.mean():.6f}, score = {score.mean():.6f}, drift: {drift.mean():.6f}, 2eps: {sqrt2eps .mean():.6f}, dW: {dW.mean():.6f}')

        x = x + dt * drift + sqrt2eps * dW

        if i < 3 or i % 10 == 0:
            print(
                f"step={t[0].item():.4f} | "
                f"b_hat={b_hat.mean():.6f} | "
                f"s_hat={s_hat.mean():.6f} | "
                f"eps={eps.mean():.6f} | "
                f"x_std={x.std():.6f}"
            )

    return x
