"""
Training script for SerpentFlow using Flow Matching.

This script trains a conditional generative model to reconstruct domain B given
low-frequency shared structure extracted using a classifier-based cutoff frequency
(r_cut).

The dataset provides:
    - "noisy": low-frequency input + stochastic high-frequency noise
    - "data": original high-resolution samples from domain B

The model is trained with Flow Matching and saved using an Exponential Moving
Average (EMA) of weights.

Example usage:
--------------
python train_serpentflow.py \
    --path_B data/domain_B_train.pt \
    --r_cut 7 \
    --name_config ERA5 \
    --save_name serpentflow_era5 \
"""


import argparse
import torch

from utils.inference_utils import generate_grid
from src.train import train_flow_matching
from utils.training_utils import EMA
from src.datasets import SerpentFlowDataset
from utils.models import UNetFlow
from utils.config import make_model_cfg, make_train_cfg


def main():

    # ---------------------------
    # Argument parsing
    # ---------------------------
    parser = argparse.ArgumentParser(description="Train SerpentFlow with Flow Matching.")
    parser.add_argument("--path_B", type=str, required=True, help="Path to dataset B (.pt)")
    parser.add_argument("--r_cut", type=int, default=5, help="Cutoff frequency for shared structure extraction")
    parser.add_argument("--name_config", type=str, default="ERA5", help="Dataset configuration name")
    parser.add_argument("--save_name", type=str, default="experiment", help="Model save identifier")
    parser.add_argument("--num_generations", type=int, default=4, help="Number of samples generated after training")
    parser.add_argument("--method", type=str, default="fourier", help="Low pass method")
    parser.add_argument("--mask_path", type=str, default="")

    args = parser.parse_args()

    # ---------------------------
    # Device
    # ---------------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # ---------------------------
    # Dataset
    # ---------------------------
    print("Loading SerpentFlow dataset...")
    dataset = SerpentFlowDataset(args.path_B, args.r_cut, method=args.method)

    # ---------------------------
    # Model definition
    # ---------------------------
    print("Building model...")
    model = UNetFlow(
        C_in=dataset[0]["noisy"].shape[0],
        C_out=dataset[0]["data"].shape[0],
        **make_model_cfg(args.name_config)
    )

    # Wrap with EMA weights
    model = EMA(model=model)

    # ---------------------------
    # Load checkpoint (if exists)
    # ---------------------------
    checkpoint_path = f"checkpoints/fm_{args.save_name}_best.pth"
    checkpoint = None

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["ema"])
        print(f"Loaded checkpoint from {checkpoint_path}")
    except FileNotFoundError:
        print(f"No checkpoint found for {args.save_name}. Training from scratch.")

    # ---------------------------
    # Training configuration
    # ---------------------------
    tcfg = make_train_cfg(args.name_config)

    # ---------------------------
    # Training
    # ---------------------------
    print("Starting Flow Matching training...")
    if len(args.mask_path)>0:
        mask = torch.load(args.mask_path)
    else:
        mask=None
    model = train_flow_matching(
        model=model,
        dataset=dataset,
        device=device,
        name=args.save_name,
        ckpt=checkpoint,
        mask=mask,
        **tcfg
    )

    # ---------------------------
    # Sampling after training
    # ---------------------------
    print("Generating samples...")

    indices = torch.randperm(len(dataset))[:args.num_generations]
    noisy_batch = torch.stack([dataset[i]["noisy"] for i in indices]).to(device)

    generate_grid(
        model=model.model,
        shape=dataset[0]["data"].shape,
        n_samples=args.num_generations,
        device=device,
        prefix=args.save_name,
        x=noisy_batch,
        ds=dataset
    )


if __name__ == "__main__":
    main()
