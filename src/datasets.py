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

import torch
from torch.utils.data import Dataset
from utils.data_utils import low_pass_tensor_batch


class SerpentFlowDataset(Dataset):
    """
    Dataset for training and inference in SerpentFlow.

    Given a dataset of high-resolution images, this class:
      1) extracts a low-frequency component using a frequency cutoff r_cut
      2) normalizes it channel-wise to [-1, 1]
      3) optionally injects stochastic high-frequency noise
      4) returns (noisy low-frequency input, original data) pairs

    Output format (per sample):
        {
            "noisy": tensor (low-frequency + noise),
            "data":  tensor (original high-resolution data)
        }
    """

    def __init__(self, ds_path, r_cut, apply_noise=True):
        """
        Args:
            ds_path (str): path to torch tensor (.pt file)
            r_cut (int): cutoff frequency for low-pass filtering
            apply_noise (bool): whether to inject stochastic HF noise
        """
        super().__init__()

        self.r_cut = r_cut
        self.apply_noise = apply_noise

        # Load full-resolution dataset
        self.data = torch.load(ds_path, map_location="cpu")

        # Precompute low-frequency component
        lp_data = low_pass_tensor_batch(self.data, self.r_cut, apply_noise=False)

        # Channel-wise normalization (using dataset statistics)
        for c in range(lp_data.shape[1]):
            min_val = lp_data[:, c].min()
            max_val = lp_data[:, c].max()
            lp_data[:, c] = (lp_data[:, c] - min_val) / (max_val - min_val + 1e-8)

        # Scale to [-1, 1] for neural network compatibility
        self.lp_data = lp_data * 2 - 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # Ground truth (high-resolution)
        data = self.data[index]

        # Low-frequency shared component
        noisy_data = self.lp_data[index]

        # Inject stochastic high-frequency noise if enabled
        if self.apply_noise:
            _, noise_lr = low_pass_tensor_batch(
                data,
                self.r_cut,
                apply_noise=self.apply_noise
            )
            noisy_data = noisy_data + noise_lr

        return {
            "noisy": noisy_data,
            "data": data
        }


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
