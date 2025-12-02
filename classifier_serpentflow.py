"""
Classifier-based cutoff frequency estimation for SerpentFlow

This script trains a binary classifier to distinguish between domain A and B.
Then, it progressively applies a low-pass filter to both domains and retrains the
classifier until the classification accuracy drops below a threshold.

The cutoff frequency at which the classifier can no longer reliably discriminate
between domains is interpreted as the optimal separation point between:
    - low-frequency shared structure
    - high-frequency domain-specific content

This value is later used in SerpentFlow for shared structure decomposition.

Usage:
    python classifier_serpentflow.py \
        --path_train_A data/train_A.pt \
        --path_train_B data/train_B.pt \
        --path_test_A data/test_A.pt \
        --path_test_B data/test_B.pt
"""

import argparse
import torch
from torch.utils.data import DataLoader

from utils.data_utils import low_pass_tensor_batch
from utils.training_utils import train_classifier, classifier_prediction
from src.datasets import TwoClassImageDataset
from utils.models import BinaryImageClassifier
from utils.config import make_train_cfg


def normalize_using_stats(x, min_val, max_val):
    """
    Normalize tensor using pre-computed statistics to avoid data leakage.

    Args:
        x (torch.Tensor): input tensor
        min_val (float): minimum value from training set
        max_val (float): maximum value from training set

    Returns:
        torch.Tensor: normalized tensor
    """
    return (x - min_val) / (max_val - min_val + 1e-8)


def main():

    # ---------------------------
    # Argument parsing
    # ---------------------------
    parser = argparse.ArgumentParser(description="Estimate cutoff frequency using classifier accuracy.")
    parser.add_argument("--path_train_A", type=str, required=True, help="Path to training dataset A (.pt)")
    parser.add_argument("--path_train_B", type=str, required=True, help="Path to training dataset B (.pt)")
    parser.add_argument("--path_test_A", type=str, required=True, help="Path to testing dataset A (.pt)")
    parser.add_argument("--path_test_B", type=str, required=True, help="Path to testing dataset B (.pt)")
    parser.add_argument("--name_config", type=str, default="ERA5", help="Dataset configuration name")
    parser.add_argument("--start_r_cut", type=int, default=10, help="Initial cutoff frequency")
    parser.add_argument("--get_r_cut", action='store_true', help="If set, only compute and print r_cut")

    args = parser.parse_args()

    # ---------------------------
    # Device and config
    # ---------------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tcfg = make_train_cfg(args.name_config)

    # ---------------------------
    # Load datasets
    # ---------------------------
    train_A = torch.load(args.path_train_A, map_location="cpu")
    train_B = torch.load(args.path_train_B, map_location="cpu")
    test_A = torch.load(args.path_test_A, map_location="cpu")
    test_B = torch.load(args.path_test_B, map_location="cpu")

    # ---------------------------
    # Balance datasets
    # ---------------------------
    if train_A.shape[0] > train_B.shape[0]:
        idx = torch.randperm(train_A.shape[0])[:train_B.shape[0]]
        train_A = train_A[idx]

    if test_A.shape[0] > test_B.shape[0]:
        idx = torch.randperm(test_A.shape[0])[:test_B.shape[0]]
        test_A = test_A[idx]

    if not args.get_r_cut:
        print(f"Balanced training: {train_A.shape[0]} samples in A, {train_B.shape[0]} in B")
        print(f"Balanced testing: {test_A.shape[0]} samples in A, {test_B.shape[0]} in B")

    # ---------------------------
    # Initial dataset + model
    # ---------------------------
    ds_train = TwoClassImageDataset(train_A, train_B)
    dl_train = DataLoader(ds_train, batch_size=tcfg['batch_size'], shuffle=True)
    model = BinaryImageClassifier(in_channels=ds_train[0][0].shape[0]).to(device)

    if not args.get_r_cut:
        print("Training initial classifier on raw data...")

    model = train_classifier(model, dl_train, epochs=20, device=device)

    ds_test = TwoClassImageDataset(test_A, test_B)
    dl_test = DataLoader(ds_test, batch_size=tcfg['batch_size'], shuffle=True)
    acc = classifier_prediction(model, dl_test, device)

    if not args.get_r_cut:
        print(f"Initial classifier accuracy = {acc:.4f}")

    # ---------------------------
    # Search cutoff frequency
    # ---------------------------
    r_cut = args.start_r_cut + 1

    while acc > 0.6 and r_cut >= 0:
        r_cut -= 1
        if not args.get_r_cut:
            print(f"Trying cutoff = {r_cut}")

        # Apply low-pass filtering
        lp_train_A = low_pass_tensor_batch(train_A, r_cut, apply_noise=False)
        lp_train_B = low_pass_tensor_batch(train_B, r_cut, apply_noise=False)

        # Normalize
        min_A, max_A = lp_train_A.min(), lp_train_A.max()
        min_B, max_B = lp_train_B.min(), lp_train_B.max()
        lp_train_A = normalize_using_stats(lp_train_A, min_A, max_A)
        lp_train_B = normalize_using_stats(lp_train_B, min_B, max_B)

        # Retrain
        ds_lp_train = TwoClassImageDataset(lp_train_A, lp_train_B)
        dl_train = DataLoader(ds_lp_train, batch_size=tcfg['batch_size'], shuffle=True)
        model = BinaryImageClassifier(in_channels=ds_lp_train[0][0].shape[0]).to(device)
        model = train_classifier(model, dl_train, epochs=20, device=device)

        # Test
        lp_test_A = normalize_using_stats(low_pass_tensor_batch(test_A, r_cut, apply_noise=False), min_A, max_A)
        lp_test_B = normalize_using_stats(low_pass_tensor_batch(test_B, r_cut, apply_noise=False), min_B, max_B)
        ds_lp_test = TwoClassImageDataset(lp_test_A, lp_test_B)
        dl_test = DataLoader(ds_lp_test, batch_size=tcfg['batch_size'], shuffle=False)
        acc = classifier_prediction(model, dl_test, device)

        if args.get_r_cut:
            continue  # do not print anything
        print(f"Accuracy at r_cut={r_cut}: {acc:.4f}")

    # ---------------------------
    # Final r_cut output
    # ---------------------------
    if args.get_r_cut:
        print(r_cut)  # only the number for bash
    else:
        print("====================================")
        print(f"Final selected cutoff frequency: {r_cut}")
        print(f"Final classifier accuracy: {acc:.4f}")
        print("====================================")


if __name__ == "__main__":
    main()
