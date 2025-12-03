"""
Frequency decomposition utilities for SerpentFlow.

This module provides tools for splitting images or 2D fields into:
    - low-frequency shared structure
    - high-frequency stochastic noise

Main functions:
    - low_pass_tensor_batch: circular low-pass filtering of a batch of images
    - upsample_and_lowpass: upsample low-resolution data and apply a Gaussian low-pass filter
"""

import torch
import torch.nn.functional as F


def low_pass_tensor_batch(imgs: torch.Tensor, r_cut: int, apply_noise: bool = False, method="fourier"):
    if method == "fourier":
        return fourier_tensor_batch(imgs, r_cut, apply_noise)
    else:
        return low_pass_blur_ignore_nans(imgs, sigma=20-r_cut, apply_noise=apply_noise)

def fourier_tensor_batch(imgs: torch.Tensor, r_cut: int, apply_noise: bool = False):
    """
    Apply a circular low-pass filter in Fourier space to a batch of images.

    Optionally, also extract the complementary high-frequency noise.

    Args:
        imgs (torch.Tensor): shape (N, C, H, W) or (C, H, W)
        r_cut (int): cutoff frequency (radius in Fourier domain)
        apply_noise (bool): whether to return high-frequency noise

    Returns:
        If apply_noise = False:
            torch.Tensor: low-frequency component (same shape as input)

        If apply_noise = True:
            Tuple[torch.Tensor, torch.Tensor]: (low-frequency, high-frequency noise)
    """

    # Ensure batch dimension exists
    if imgs.ndim == 3:
        imgs = imgs.unsqueeze(0)  # (1, C, H, W)

    N, C, H, W = imgs.shape

    # Create frequency grid
    Y, X = torch.meshgrid(
        torch.arange(H, device=imgs.device),
        torch.arange(W, device=imgs.device),
        indexing='ij'
    )
    cy, cx = H // 2, W // 2
    radius = torch.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    # Circular low-pass filter mask
    mask = (radius <= r_cut).float()
    mask = mask[None, None, :, :]  # broadcast to (1, 1, H, W)

    # Fourier transform
    fft = torch.fft.fftshift(
        torch.fft.fft2(imgs, dim=(-2, -1)),
        dim=(-2, -1)
    )

    # Apply low-pass filter
    f_low = fft * mask
    low_freq = torch.fft.ifft2(
        torch.fft.ifftshift(f_low, dim=(-2, -1)),
        dim=(-2, -1)
    ).real

    if not apply_noise:
        return low_freq.squeeze(0) if low_freq.shape[0] == 1 else low_freq

    # ---------------------------
    # Compute high-frequency noise
    # ---------------------------
    noise = torch.randn_like(imgs)

    fft_noise = torch.fft.fftshift(
        torch.fft.fft2(noise, dim=(-2, -1)),
        dim=(-2, -1)
    )

    # Keep only high-frequency components
    f_high = fft_noise * (1 - mask)

    high_freq = torch.fft.ifft2(
        torch.fft.ifftshift(f_high, dim=(-2, -1)),
        dim=(-2, -1)
    ).real

    return (
        low_freq.squeeze(0) if low_freq.shape[0] == 1 else low_freq,
        high_freq.squeeze(0) if high_freq.shape[0] == 1 else high_freq
    )

def gaussian_kernel_1d(sigma, max_size=101, device='cpu'):
    size = int(2 * torch.ceil(torch.tensor(3.0 * sigma)) + 1)
    size = min(size, max_size)
    x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    kernel = torch.exp(-(x**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def low_pass_blur_ignore_nans(imgs: torch.Tensor, sigma: float, apply_noise=False):
    if imgs.ndim == 3:
        imgs = imgs.unsqueeze(0)
    N, C, H, W = imgs.shape
    device = imgs.device

    # Gaussian kernels
    k_lat = gaussian_kernel_1d(sigma, device=device).view(1, 1, -1, 1)
    k_lon = gaussian_kernel_1d(sigma, device=device).view(1, 1, 1, -1)
    pad_lat = k_lat.shape[2] // 2
    pad_lon = k_lon.shape[3] // 2

    # Mask to ignore NaNs
    mask = (~torch.isnan(imgs)).float()
    imgs_zeroed = torch.nan_to_num(imgs, nan=0.0)

    # Convolution
    imgs_reshape = imgs_zeroed.reshape(N*C, 1, H, W)
    mask_reshape = mask.reshape(N*C, 1, H, W)
    blurred = F.conv2d(F.pad(imgs_reshape, (0,0,pad_lat,pad_lat)), k_lat)
    blurred = F.conv2d(F.pad(blurred, (pad_lon,pad_lon,0,0)), k_lon)

    mask_blur = F.conv2d(F.pad(mask_reshape, (0,0,pad_lat,pad_lat)), k_lat)
    mask_blur = F.conv2d(F.pad(mask_blur, (pad_lon,pad_lon,0,0)), k_lon)
    mask_blur = torch.clamp(mask_blur, min=1e-6)

    blurred /= mask_blur
    blurred = blurred.reshape(N, C, H, W)
    blurred[mask == 0] = float('nan')

    if not apply_noise:
        return blurred.squeeze(0) if blurred.shape[0] == 1 else blurred

    # High-frequency noise
    noise = torch.randn_like(imgs)
    noise_reshape = noise.reshape(N*C, 1, H, W)
    noise_blurred = F.conv2d(F.pad(noise_reshape, (0,0,pad_lat,pad_lat)), k_lat)
    noise_blurred = F.conv2d(F.pad(noise_blurred, (pad_lon,pad_lon,0,0)), k_lon)
    noise_blurred = noise_blurred.reshape(N, C, H, W)

    high_freq = noise - noise_blurred

    return (
        blurred.squeeze(0) if blurred.shape[0] == 1 else blurred,
        high_freq.squeeze(0) if high_freq.shape[0] == 1 else high_freq
    )

def upsample_and_lowpass(x: torch.Tensor, target_size: tuple, cutoff: float = 0.2) -> torch.Tensor:
    """
    Upsample a low-resolution tensor and apply a Gaussian low-pass filter.

    Args:
        x (torch.Tensor): shape (B, h, w) or (h, w)
        target_size (tuple): (H, W) target spatial dimensions
        cutoff (float): relative cutoff frequency (0 < cutoff <= 0.5)
                        0.5 corresponds to Nyquist frequency

    Returns:
        torch.Tensor: upsampled and low-pass filtered tensor, shape (B, H, W)
    """

    # Ensure batch dimension
    if x.ndim == 2:
        x = x.unsqueeze(0)  # (1, h, w)

    B, h, w = x.shape
    H, W = target_size

    # Upsample in spatial domain
    x = x.unsqueeze(1)  # (B, 1, h, w)
    x_up = F.interpolate(x, size=(H, W), mode='nearest')
    x_up = x_up.to(torch.complex64)

    # Centered 2D FFT
    x_fft = torch.fft.fftshift(torch.fft.fft2(x_up, dim=(-2, -1)), dim=(-2, -1))

    # Create a Gaussian low-pass filter mask
    Y, X = torch.meshgrid(
        torch.linspace(-0.5, 0.5, H, device=x_fft.device),
        torch.linspace(-0.5, 0.5, W, device=x_fft.device),
        indexing='ij'
    )
    radius = torch.sqrt(X**2 + Y**2)
    gaussian_mask = torch.exp(-0.5 * (radius / cutoff) ** 2)
    gaussian_mask = gaussian_mask.unsqueeze(0).unsqueeze(0)  # broadcast to (B, 1, H, W)

    # Apply low-pass filter
    x_fft = x_fft * gaussian_mask

    # Inverse FFT and return real part
    x_ifft = torch.fft.ifft2(torch.fft.ifftshift(x_fft, dim=(-2, -1)), dim=(-2, -1))
    x_real = x_ifft.real

    return x_real.squeeze(1) if x_real.shape[1] == 1 else x_real
