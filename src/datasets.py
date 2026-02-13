"""
Datasets used in SerpentFlow.

This module defines:
    - SerpentFlowDataset: dataset used for training and inference of the generative model.
    - TwoClassImageDataset: simple binary classification dataset (used for cutoff estimation).

SerpentFlowDataset exposes:
    - "data": high-resolution target samples (domain B during training, domain A during inference)
    - "noisy": low-frequency representation + optional injected noise

Both datasets are PyTorch-compatible.
"""

import random
import torch
from torch.utils.data import Dataset
from utils.data_utils import low_pass_tensor_batch

def normalize_column(x):
    mask = torch.isfinite(x)
    if not mask.any():
        return x  # leave as-is
    mean_val = x[mask].mean()
    std_val  = x[mask].std()
    x[mask] = (x[mask] - mean_val) / (std_val + 1e-8)
    return x


class SerpentFlowDataset(Dataset):
    def __init__(self, ds_path, r_cut, apply_noise=True, method="fourier", dual=False):
        super().__init__()

        self.r_cut = r_cut
        self.method = method
        self.apply_noise = apply_noise

        self.data = torch.load(ds_path, map_location="cpu")
        lp_data = low_pass_tensor_batch(self.data, self.r_cut, apply_noise=False, method=self.method)
        self.stats_data = torch.stack([
            lp_data.mean(dim=(2,3)),
            lp_data.std(dim=(2,3))
        ], dim=2)
        # Channel-wise normalization
        for c in range(lp_data.shape[1]):
            lp_data[:, c] = normalize_column(lp_data[:, c])
        for c in range(self.data.shape[1]):
            self.data[:, c] = normalize_column(self.data[:, c])    
        self.lp_data = lp_data
        self.dual = dual

    def __getitem__(self, index):
        data = self.data[index]
        noisy_data = self.lp_data[index]

        if self.apply_noise:
            _, noise_lr = low_pass_tensor_batch(data, self.r_cut, apply_noise=True, method=self.method)
            noisy_data = noisy_data + noise_lr

        stats = self.stats_data[index]  # shape [C, 2]
        if self.dual:
            noisy_data = torch.randn_like(noisy_data)
        return {
            "noisy": noisy_data,
            "data": data,
            "stats": stats
        }
    def __len__(self):
        return len(self.data)

class UnpairedDataset(Dataset):
    def __init__(self, ds_A, ds_B=None, margin=30):
        super().__init__()

        self.data_A = torch.load(ds_A, map_location="cpu")
        self.stats_data = torch.stack([
            self.data_A.mean(dim=(2,3)),
            self.data_A.std(dim=(2,3))
        ], dim=2)
        for c in range(self.data_A.shape[1]):
            self.data_A[:, c] = normalize_column(self.data_A[:, c])    
        if ds_B is not None:
            self.data_B = torch.load(ds_B, map_location="cpu")
            for c in range(self.data_B.shape[1]):
                self.data_B[:, c] = normalize_column(self.data_B[:, c])  
        self.margin=margin  

    def __getitem__(self, index):
        data_A = self.data_A[index]

        if hasattr(self, "data_B"):
            rand_idx = random.randint(0, len(self.data_B) - 1)
            data_B = self.data_B[rand_idx]
        else:
            data_B = data_A
        stats = self.stats_data[index]  # shape [C, 2]
        
        return {
            "noisy": data_A,
            "data": data_B,
            "stats": stats
        }
    def __len__(self):
        return len(self.data_A)

class TwoClassImageDataset(Dataset):
    """
    Binary classification dataset used to estimate cutoff frequency.

    Combines domain A and domain B into a single dataset and assigns:
        label = 0 → domain A
        label = 1 → domain B
    """

    def __init__(self, data_A, data_B):
        """
        Args:
            data_A (torch.Tensor): samples from domain A
            data_B (torch.Tensor): samples from domain B
        """
        self.data = torch.cat([data_A, data_B], dim=0)
        self.labels = torch.cat([
            torch.zeros(len(data_A), dtype=torch.long),
            torch.ones(len(data_B), dtype=torch.long)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
