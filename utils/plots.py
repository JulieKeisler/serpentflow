"""
Plotting utilities for SerpentFlow.

Provides functions to save image batches as grids.
"""

import torch
from torchvision.utils import save_image, make_grid


@torch.no_grad()
def to_grid_and_save(x: torch.Tensor, path: str, nrow: int = None, pad_value: float = 0.05):
    """
    Convert a batch of images to a grid and save to disk.

    Args:
        x (torch.Tensor): batch of images (N, C, H, W)
        path (str): file path to save the grid
        nrow (int, optional): number of images per row. Defaults to sqrt(N)
        pad_value (float, optional): padding value between images. Default 0.05
    """
    n = x.shape[0]
    nrow = int(n ** 0.5) if nrow is None else nrow
    grid = make_grid(x, nrow=nrow, pad_value=pad_value)
    save_image(grid, path)
