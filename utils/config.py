"""
Configuration utilities for SerpentFlow.

This module defines dataset-specific configurations for:
    - model architecture
    - training hyperparameters

Each dataset name must be registered in both configuration functions.
"""


def make_model_cfg(data_type: str):
    """
    Return model configuration depending on the dataset type.

    Args:
        data_type (str): Dataset identifier (e.g. 'MNIST', 'ERA5', 'CIFAR10')

    Returns:
        dict: keyword arguments for UNetFlow initialization
    """

    if data_type == 'MNIST':
        return dict(
            base_ch=64,
            ch_mult=(1, 2, 2)
        )

    if data_type == 'ERA5':
        return dict(
            base_ch=96,
            ch_mult=(1, 2, 2)
        )

    if data_type == 'CIFAR10':
        return dict(
            base_ch=128,
            ch_mult=(1, 2, 2, 4, 4),
            num_res_blocks=3,
            attention_resolutions=(14, 28, 56)
        )

    raise ValueError(f"Unknown dataset type: {data_type}")


def make_train_cfg(data_type: str):
    """
    Return training configuration for a given dataset.

    Args:
        data_type (str): Dataset identifier

    Returns:
        dict: training hyperparameters
    """

    configs = {
        'MNIST': dict(
            batch_size=256,
            epochs=200,
            lr=2e-4,
            accum_iter=1
        ),
        'ERA5': dict(
            batch_size=64,
            epochs=20,
            lr=2e-4,
            accum_iter=1
        ),
        'CIFAR10': dict(
            batch_size=32,
            epochs=100,
            lr=2e-4,
            accum_iter=4
        )
    }

    if data_type not in configs:
        raise ValueError(f"Unknown dataset type: {data_type}")

    return configs[data_type]
