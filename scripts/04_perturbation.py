"""
04_perturbation.py — In silico TF perturbation analysis using a trained PINN.

Loads a trained checkpoint, runs wildtype inference, then systematically
knocks out (or overexpresses) each TF to measure downstream gene expression
changes through the mechanistic model.

Outputs:
  - perturbation_effects.csv     — (num_tfs x num_genes) matrix of log2 fold changes
  - perturbation_fates.csv       — (num_tfs x num_fates) matrix of fate probability shifts
  - perturbation_summary.csv     — ranked TFs by total downstream impact

Usage:
    python scripts/04_perturbation.py \
        --checkpoint results/mousehead_.../checkpoints/best_model.pt \
        --h5ad data/processed/spatial_adata.h5ad \
        --output_dir results/perturbation/ \
        --mode knockout
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import scanpy as sc

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.hybrid_pinn import SpatialMechanisticModel
from src.data.regulatory_networks import build_frozen_grn_matrix


def load_trained_model(checkpoint_path: str, device: torch.device):
    """Load a trained SpatialMechanisticModel from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    # Rebuild GRN matrix
    expressed_genes = pd.read_csv(
        "data/processed/expressed_genes.csv", header=None
    )[0].tolist()
    frozen_grn = build_frozen_grn_matrix(
        tftg_path="src/data/frozen_databases/TFTGDB.csv",
        expressed_tfs=expressed_genes,
        expressed_target_genes=expressed_genes,
    )

    model = SpatialMechanisticModel(
        input_spatial_dim=config["input_spatial_dim"],
        num_tfs=frozen_grn.shape[0],
        num_target_genes=frozen_grn.shape[1],
        num_terminal_fates=config["num_terminal_fates"],
        frozen_grn_matrix=frozen_grn,
        dt=config["dt"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Loaded model from epoch {ckpt['epoch']}")
    print(f"  Architecture: {frozen_grn.shape[0]} TFs -> {frozen_grn.shape[1]} genes")
    print(f"  Training loss: {ckpt['losses']['total']:.4f}")

    return model, config


def run_wildtype(model, u_seq, device):
    """Run wildtype (unperturbed) forward pass and return moments + fates."""
    with torch.no_grad():
        result = model(u_seq)

    h_seq = result["hidden_tfs"]                          # (B, S, num_tfs)
    B, S, _ = h_seq.shape

    # Compute moments at bin midpoints
    dt = model.dt
    bin_t = torch.linspace(dt / 2, 1.0 - dt / 2, S, device=device)
    bin_t = bin_t.unsqueeze(-1).unsqueeze(0).expand(B, S, 1)  # (B, S, 1)

    with torch.no_grad():
        moments = model.moment_mlp(bin_t, h_seq)   # 5 x (B, S, G)

    # Fate probabilities
    fate_probs = F.softmax(result["fate_logits"], dim=-1)   # (B, S, num_fates)

    return {
        "h_seq": h_seq,
        "nascent_mean": moments[0],
        "mature_mean": moments[1],
        "fate_probs": fate_probs,
        "burst_freq": result["burst_freq"],
        "burst_size": result["burst_size"],
    }


def run_perturbation(model, u_seq, tf_idx, device, mode="knockout", scale=0.0):
    """
    Run perturbed forward pass with one TF knocked out or overexpressed.

    Args:
        tf_idx: Index of the TF to perturb in h_seq[..., tf_idx]
        mode: 'knockout' (set to 0), 'overexpress' (multiply by scale), or 'scale'
        scale: Multiplier for the TF activity (0.0 = knockout, 2.0 = 2x overexpression)
    """
    with torch.no_grad():
        # Get RNN hidden states
        h_seq = model.rnn(u_seq)   # (B, S, num_tfs)

        # Apply perturbation to specific TF
        h_perturbed = h_seq.clone()
        if mode == "knockout":
            h_perturbed[:, :, tf_idx] = 0.0
        elif mode == "overexpress":
            h_perturbed[:, :, tf_idx] *= scale
        else:
            h_perturbed[:, :, tf_idx] *= scale

        # Recompute burst parameters from perturbed h_seq
        a_t = F.softplus(torch.matmul(h_perturbed, model.frozen_grn))
        b_t = F.softplus(model.W_size(h_perturbed))

        # Recompute moments
        B, S, _ = h_perturbed.shape
        dt = model.dt
        bin_t = torch.linspace(dt / 2, 1.0 - dt / 2, S, device=device)
        bin_t = bin_t.unsqueeze(-1).unsqueeze(0).expand(B, S, 1)

        moments = model.moment_mlp(bin_t, h_perturbed)

        # Recompute fates
        fate_logits = model.fate_head(h_perturbed)
        fate_probs = F.softmax(fate_logits, dim=-1)

    return {
        "nascent_mean": moments[0],
        "mature_mean": moments[1],
        "fate_probs": fate_probs,
    }


def perturbation_analysis(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load model ---
    model, config = load_trained_model(args.checkpoint, device)
    num_tfs = model.num_tfs
    num_genes = model.num_target_genes

    # --- Load spatial data ---
    print(f"\nLoading spatial data from {args.h5ad} ...")
    adata = sc.read_h5ad(args.h5ad)

    # Build spatial input sequences (same KNN logic as dataloader)
    from sklearn.neighbors import NearestNeighbors
    coords = adata.obsm["spatial"]
    seq_len = int(1.0 / config["dt"])
    nn = NearestNeighbors(n_neighbors=seq_len).fit(coords)
    _, neighbors = nn.kneighbors(coords)

    # Sample a subset for efficiency
    n_sample = min(args.n_cells, adata.n_obs)
    sample_idx = np.random.choice(adata.n_obs, n_sample, replace=False)

    u_seqs = []
    for idx in sample_idx:
        seq = coords[neighbors[idx]]
        u_seqs.append(seq)
    u_seq = torch.tensor(np.array(u_seqs), dtype=torch.float32, device=device)
    print(f"  Sampled {n_sample} cells, seq_len={seq_len}")

    # --- Wildtype inference ---
    print("\nRunning wildtype inference ...")
    wt = run_wildtype(model, u_seq, device)
    # Average over sequence and batch: (G,) and (F,)
    wt_expr = wt["mature_mean"].mean(dim=(0, 1)).cpu().numpy()
    wt_fate = wt["fate_probs"].mean(dim=(0, 1)).cpu().numpy()

    # --- Perturbation loop ---
    print(f"\nRunning {args.mode} perturbations for {num_tfs} TFs ...")
    scale = args.scale if args.mode == "overexpress" else 0.0

    effect_matrix = np.zeros((num_tfs, num_genes))     # log2 fold changes
    fate_matrix = np.zeros((num_tfs, len(wt_fate)))     # fate probability shifts

    for tf_i in range(num_tfs):
        perturbed = run_perturbation(
            model, u_seq, tf_i, device,
            mode=args.mode, scale=scale,
        )
        pert_expr = perturbed["mature_mean"].mean(dim=(0, 1)).cpu().numpy()
        pert_fate = perturbed["fate_probs"].mean(dim=(0, 1)).cpu().numpy()

        # Log2 fold change (with pseudocount to avoid log(0))
        effect_matrix[tf_i] = np.log2((pert_expr + 1e-6) / (wt_expr + 1e-6))
        fate_matrix[tf_i] = pert_fate - wt_fate

        if (tf_i + 1) % 50 == 0 or tf_i == 0:
            total_effect = np.abs(effect_matrix[tf_i]).sum()
            print(f"  TF {tf_i+1:4d}/{num_tfs} | total |log2FC| = {total_effect:.2f}")

    # --- Build DataFrames ---
    # Get TF and gene names from expressed_genes.csv
    expressed_genes = pd.read_csv(
        "data/processed/expressed_genes.csv", header=None
    )[0].tolist()

    # TF names come from GRN matrix rows — need to reconstruct
    from src.data.regulatory_networks import build_frozen_grn_matrix
    grn_info = build_frozen_grn_matrix(
        tftg_path="src/data/frozen_databases/TFTGDB.csv",
        expressed_tfs=expressed_genes,
        expressed_target_genes=expressed_genes,
        return_names=True,
    )
    if isinstance(grn_info, tuple):
        _, tf_names, gene_names = grn_info
    else:
        tf_names = [f"TF_{i}" for i in range(num_tfs)]
        gene_names = expressed_genes[:num_genes]

    fate_names = adata.uns.get("palantir_terminal_states",
                               [f"Fate_{i}" for i in range(len(wt_fate))])

    effects_df = pd.DataFrame(effect_matrix, index=tf_names, columns=gene_names)
    fates_df = pd.DataFrame(fate_matrix, index=tf_names, columns=fate_names)

    # Summary: rank TFs by total downstream impact
    summary_df = pd.DataFrame({
        "tf_name": tf_names,
        "total_abs_log2fc": np.abs(effect_matrix).sum(axis=1),
        "max_abs_log2fc": np.abs(effect_matrix).max(axis=1),
        "n_affected_genes": (np.abs(effect_matrix) > 0.5).sum(axis=1),
        "max_fate_shift": np.abs(fate_matrix).max(axis=1),
    }).sort_values("total_abs_log2fc", ascending=False)

    # --- Save ---
    os.makedirs(args.output_dir, exist_ok=True)
    effects_df.to_csv(os.path.join(args.output_dir, "perturbation_effects.csv"))
    fates_df.to_csv(os.path.join(args.output_dir, "perturbation_fates.csv"))
    summary_df.to_csv(os.path.join(args.output_dir, "perturbation_summary.csv"),
                      index=False)

    print(f"\n--- Perturbation Results ---")
    print(f"  Effects matrix: {effects_df.shape} saved to perturbation_effects.csv")
    print(f"  Fate shifts:    {fates_df.shape} saved to perturbation_fates.csv")
    print(f"\n  Top 10 most impactful TFs ({args.mode}):")
    print(summary_df.head(10).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="In silico TF perturbation analysis"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best_model.pt checkpoint")
    parser.add_argument("--h5ad", default="data/processed/spatial_adata.h5ad",
                        help="Spatial transcriptomics h5ad")
    parser.add_argument("--output_dir", default="results/perturbation/",
                        help="Directory for output CSVs")
    parser.add_argument("--mode", choices=["knockout", "overexpress"],
                        default="knockout",
                        help="Perturbation mode")
    parser.add_argument("--scale", type=float, default=2.0,
                        help="Scale factor for overexpression mode")
    parser.add_argument("--n_cells", type=int, default=500,
                        help="Number of cells to sample for perturbation")
    args = parser.parse_args()

    perturbation_analysis(args)


if __name__ == "__main__":
    main()
