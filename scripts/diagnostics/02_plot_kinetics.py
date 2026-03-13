import argparse
import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.models.hybrid_pinn import SpatialMechanisticModel

def main():
    parser = argparse.ArgumentParser(description="Plot learned kinetic rates")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt", help="Path to checkpoint")
    parser.add_argument("--output", default="logs/kinetic_distributions.png", help="Output image path")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    device = torch.device("cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]

    # Instantiate model with checkpoint dimensions
    model = SpatialMechanisticModel(
        input_spatial_dim=config.get("input_spatial_dim", 2),
        num_tfs=config["num_tfs"],
        num_target_genes=config["num_target_genes"],
        num_terminal_fates=config["num_terminal_fates"],
        frozen_grn_matrix=torch.zeros((config["num_tfs"], config["num_target_genes"])), # Dummy GRN
        dt=config.get("dt", 0.02)
    )

    # Clean state dict keys (strip DataParallel/Wrapper prefixes)
    clean_state_dict = {}
    for k, v in checkpoint["model_state_dict"].items():
        new_k = k.replace("module.", "").replace("model.", "")
        clean_state_dict[new_k] = v

    model.load_state_dict(clean_state_dict, strict=False)

    # Extract strictly positive kinetic rates via the property methods
    beta_vals = model.beta.detach().numpy()
    gamma_vals = model.gamma.detach().numpy()

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.hist(beta_vals, bins=40, color='#4C72B0', edgecolor='white', linewidth=0.5)
    ax1.set_title(f"Splicing Rates (Beta)\nMean: {beta_vals.mean():.4f} | Min: {beta_vals.min():.4e}", fontweight="bold")
    ax1.set_xlabel("Rate (1/time)")
    ax1.set_ylabel("Number of Genes")

    ax2.hist(gamma_vals, bins=40, color='#55A868', edgecolor='white', linewidth=0.5)
    ax2.set_title(f"Degradation Rates (Gamma)\nMean: {gamma_vals.mean():.4f} | Min: {gamma_vals.min():.4e}", fontweight="bold")
    ax2.set_xlabel("Rate (1/time)")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Successfully saved distributions to {args.output}")

if __name__ == "__main__":
    main()
