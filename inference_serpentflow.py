"""
Inference script for SerpentFlow.

This script loads a trained Flow Matching model and applies it to domain A
to generate synthetic realizations in domain B while preserving shared
low-frequency structure.

The model was trained only on domain B and is now used conditionally on domain A
through the shared representation.

Example usage:
--------------
python inference_serpentflow.py \
    --path_A data/domain_A_test.pt \
    --r_cut 7 \
    --name_config ERA5 \
    --save_name_model serpentflow_era5 \
    --save_name ERA5_predictions
"""

import argparse
import torch
from torch.utils.data import DataLoader

from src.inference import integrate_and_store
from utils.training_utils import EMA
from src.datasets import SerpentFlowDataset
from utils.models import UNetFlow
from utils.config import make_model_cfg, make_train_cfg


def main():

    # ---------------------------
    # Argument parsing
    # ---------------------------
    parser = argparse.ArgumentParser(description="Run SerpentFlow inference on domain A.")
    parser.add_argument("--path_A", type=str, required=True, help="Path to domain A dataset (.pt)")
    parser.add_argument("--r_cut", type=int, default=5, help="Cutoff frequency used during training")
    parser.add_argument("--name_config", type=str, default="ERA5", help="Dataset configuration name")
    parser.add_argument("--save_name_model", type=str, default="experiment", help="Model identifier (checkpoint name)")
    parser.add_argument("--save_name", type=str, default="experiment", help="Output result file name")

    args = parser.parse_args()

    # ---------------------------
    # Device
    # ---------------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # ---------------------------
    # Dataset
    # ---------------------------
    print("Loading SerpentFlow dataset for inference...")
    dataset = SerpentFlowDataset(args.path_A, args.r_cut)

    # ---------------------------
    # Model definition
    # ---------------------------
    print("Building model...")
    model = UNetFlow(
        C_in=dataset[0]["noisy"].shape[0],
        C_out=dataset[0]["data"].shape[0],
        **make_model_cfg(args.name_config)
    )

    # Wrap with EMA model
    model = EMA(model=model)

    # ---------------------------
    # Load checkpoint
    # ---------------------------
    checkpoint_path = f"checkpoints/fm_{args.save_name_model}_best.pth"

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["ema"])
        print(f"Loaded model from {checkpoint_path}")
    except FileNotFoundError:
        print(f"ERROR: No checkpoints found for {args.save_name_model}.")
        return

    # ---------------------------
    # DataLoader
    # ---------------------------
    tcfg = make_train_cfg(args.name_config)

    dataloader = DataLoader(
        dataset,
        batch_size=tcfg['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # ---------------------------
    # Inference
    # ---------------------------
    print("Running inference...")
    integrate_and_store(
        dataloader=dataloader,
        model_to=model,
        device=device,
        filename=f"data/results/{args.save_name}.pt"
    )

    print("Inference complete.")


if __name__ == "__main__":
    main()
