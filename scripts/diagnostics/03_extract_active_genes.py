"""
03_extract_active_genes.py — Identify genes that learned valid biological kinetics.

Genes clamped at the training floor (min=1e-4) in both beta and gamma failed to
learn meaningful rates. This script loads a checkpoint, aligns the rate tensors to
expressed_genes.csv, and exports only the genes where at least one of beta/gamma
escaped the clamp by a configurable margin.

Usage:
  python scripts/diagnostics/03_extract_active_genes.py \
      --checkpoint checkpoints/checkpoint_epoch_500.pt \
      --genes data/processed/expressed_genes.csv \
      --output logs/active_genes_epoch500.csv \
      --clamp_threshold 2e-4
"""

import argparse
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.models.hybrid_pinn import SpatialMechanisticModel


CLAMP_FLOOR = 1e-4  # min= value used in training's torch.clamp calls


def main():
    parser = argparse.ArgumentParser(description="Extract active (unclamped) PINN genes")
    parser.add_argument("--checkpoint", default="checkpoints/checkpoint_epoch_500.pt",
                        help="Path to checkpoint file")
    parser.add_argument("--genes", default="data/processed/expressed_genes.csv",
                        help="Path to expressed_genes.csv (one gene name per row, no header)")
    parser.add_argument("--output", default="logs/active_genes_epoch500.csv",
                        help="Output CSV path")
    parser.add_argument("--clamp_threshold", type=float, default=2e-4,
                        help="Genes with both beta AND gamma below this value are considered "
                             "clamped and excluded. Default: 2e-4 (2x the training clamp floor).")
    args = parser.parse_args()

    # --- Load checkpoint ---
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    num_target_genes = config["num_target_genes"]

    # --- Load gene names ---
    genes = pd.read_csv(args.genes, header=None)[0].tolist()
    if len(genes) != num_target_genes:
        print(
            f"WARNING: expressed_genes.csv has {len(genes)} genes but checkpoint expects "
            f"num_target_genes={num_target_genes}.\n"
            f"  The local gene list was likely updated after this checkpoint was saved.\n"
            f"  Falling back to numeric indices (gene_0 … gene_{num_target_genes - 1}).\n"
            f"  To get named output, sync expressed_genes.csv from the cluster."
        )
        genes = [f"gene_{i}" for i in range(num_target_genes)]

    # --- Instantiate model and load weights ---
    model = SpatialMechanisticModel(
        input_spatial_dim=config.get("input_spatial_dim", 2),
        num_tfs=config["num_tfs"],
        num_target_genes=num_target_genes,
        num_terminal_fates=config["num_terminal_fates"],
        frozen_grn_matrix=torch.zeros((config["num_tfs"], num_target_genes)),
        dt=config.get("dt", 0.02),
    )

    clean_state_dict = {
        k.replace("module.", "").replace("model.", ""): v
        for k, v in checkpoint["model_state_dict"].items()
    }
    model.load_state_dict(clean_state_dict, strict=False)
    model.eval()

    # --- Extract rates ---
    with torch.no_grad():
        beta_vals  = model.beta.numpy()   # (num_target_genes,)
        gamma_vals = model.gamma.numpy()  # (num_target_genes,)

    # --- Filter: keep genes where at least one rate escaped the clamp ---
    # A gene is considered clamped only if BOTH rates are at or below the threshold.
    # If either rate learned a meaningful value the gene is retained.
    df = pd.DataFrame({
        "gene":  genes,
        "beta":  beta_vals,
        "gamma": gamma_vals,
    })
    df["beta_clamped"]  = df["beta"]  <= args.clamp_threshold
    df["gamma_clamped"] = df["gamma"] <= args.clamp_threshold
    df["fully_clamped"] = df["beta_clamped"] & df["gamma_clamped"]

    active = df[~df["fully_clamped"]].copy()
    active = active.drop(columns=["beta_clamped", "gamma_clamped", "fully_clamped"])
    active = active.sort_values("beta", ascending=False).reset_index(drop=True)

    # --- Report ---
    n_total   = len(df)
    n_clamped = df["fully_clamped"].sum()
    n_active  = len(active)
    print(f"\nTotal genes:    {n_total}")
    print(f"Fully clamped:  {n_clamped}  (beta <= {args.clamp_threshold:.0e} "
          f"AND gamma <= {args.clamp_threshold:.0e})")
    print(f"Active genes:   {n_active}  ({100 * n_active / n_total:.1f}%)")
    print(f"\nTop 10 by beta:")
    print(active.head(10).to_string(index=False))

    # --- Save ---
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    active.to_csv(args.output, index=False)
    print(f"\nSaved active gene list to: {args.output}")


if __name__ == "__main__":
    main()
