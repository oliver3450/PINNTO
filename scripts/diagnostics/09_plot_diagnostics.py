"""
09_plot_diagnostics.py — Lightweight diagnostic plots from a trained checkpoint.

No KO sweeps. One WT forward pass + direct inspection of model weights + h5ad metadata.

Produces (all saved to <outdir>/diagnostics/):
  spatial_pseudotime.png   — beads colored by Palantir pseudotime
  spatial_entropy.png      — beads colored by Palantir differentiation entropy
  spatial_wt_fate.png      — beads colored by model-predicted dominant fate
  kinetic_rates.png        — beta / gamma distributions + scatter
  fate_probabilities.png   — per-fate probability histograms (WT)
  top_active_tfs.png       — top 20 TFs by beta+gamma kinetic activity

Usage:
    python scripts/diagnostics/09_plot_diagnostics.py \
        --checkpoint results/.../checkpoints/best_model.pt \
        --h5ad data/processed/spatial_adata.h5ad \
        --genes data/processed/expressed_genes.csv \
        --outdir logs/evaluation
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.models.hybrid_pinn import SpatialMechanisticModel
from src.data.dataloader import get_dataloader


def load_model(checkpoint_path, device):
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = SpatialMechanisticModel(
        input_spatial_dim=cfg.get("input_spatial_dim", 2),
        num_tfs=cfg["num_tfs"],
        num_target_genes=cfg["num_target_genes"],
        num_terminal_fates=cfg["num_terminal_fates"],
        frozen_grn_matrix=torch.zeros((cfg["num_tfs"], cfg["num_target_genes"])),
        dt=cfg.get("dt", 0.02),
    )
    clean_sd = {k.replace("module.", "").replace("model.", ""): v
                for k, v in ck["model_state_dict"].items()}
    model.load_state_dict(clean_sd, strict=False)
    model.eval()
    return model, cfg


def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--h5ad", default="data/processed/spatial_adata.h5ad")
    parser.add_argument("--genes", default="data/processed/expressed_genes.csv")
    parser.add_argument("--outdir", default="logs/evaluation")
    args = parser.parse_args()

    diag_dir = os.path.join(args.outdir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    device = torch.device("cpu")

    model, cfg = load_model(args.checkpoint, device)
    num_tfs   = cfg["num_tfs"]
    num_fates = cfg["num_terminal_fates"]
    seq_len   = int(1.0 / cfg.get("dt", 0.02))

    gene_names = pd.read_csv(args.genes, header=None)[0].tolist()
    if len(gene_names) < num_tfs:
        gene_names += [f"gene_{i}" for i in range(len(gene_names), num_tfs)]
    gene_names = gene_names[:num_tfs]

    # ----------------------------------------------------------------
    # Load h5ad metadata (no model needed for these)
    # ----------------------------------------------------------------
    print("Loading h5ad metadata...")
    adata = sc.read_h5ad(args.h5ad)
    SPATIAL_KEYS = ["spatial", "X_xy_loc", "xy_loc", "X_spatial"]
    spatial_key = next((k for k in SPATIAL_KEYS if k in adata.obsm), None)
    coords = adata.obsm[spatial_key]
    s_size = max(1, min(8, 40000 // len(coords)))

    # ----------------------------------------------------------------
    # 1. Spatial pseudotime
    # ----------------------------------------------------------------
    if "palantir_pseudotime" in adata.obs:
        print("Plotting pseudotime map...")
        pt = adata.obs["palantir_pseudotime"].values
        fig, ax = plt.subplots(figsize=(6, 5))
        sc_plot = ax.scatter(coords[:, 0], coords[:, 1], c=pt,
                             cmap="viridis", s=s_size, edgecolors="none", rasterized=True)
        plt.colorbar(sc_plot, ax=ax, label="Pseudotime")
        ax.set_title("Palantir Pseudotime")
        ax.axis("equal"); ax.axis("off")
        plt.tight_layout()
        save(fig, os.path.join(diag_dir, "spatial_pseudotime.png"))

    # ----------------------------------------------------------------
    # 2. Spatial entropy (differentiation potential)
    # ----------------------------------------------------------------
    if "palantir_entropy" in adata.obs:
        print("Plotting entropy map...")
        ent = adata.obs["palantir_entropy"].values
        fig, ax = plt.subplots(figsize=(6, 5))
        sc_plot = ax.scatter(coords[:, 0], coords[:, 1], c=ent,
                             cmap="magma", s=s_size, edgecolors="none", rasterized=True)
        plt.colorbar(sc_plot, ax=ax, label="Entropy")
        ax.set_title("Differentiation Entropy (Palantir)")
        ax.axis("equal"); ax.axis("off")
        plt.tight_layout()
        save(fig, os.path.join(diag_dir, "spatial_entropy.png"))

    # ----------------------------------------------------------------
    # 3. WT predicted fate map (1 forward pass)
    # ----------------------------------------------------------------
    print("Running WT forward pass for fate map...")
    dataloader = get_dataloader(
        h5ad_path=args.h5ad, batch_size=2048, seq_len=seq_len,
        num_genes=cfg["num_target_genes"], num_fates=num_fates,
        shuffle=False, drop_last=False,
    )

    all_coords_model, all_probs = [], []
    with torch.no_grad():
        for batch in dataloader:
            u = batch["u_seq"].to(device)
            all_coords_model.append(u[:, 0, :].cpu().numpy())
            out = model(u, collocation_t=None)
            all_probs.append(F.softmax(out["fate_logits"].mean(dim=1), dim=-1).cpu().numpy())
    model_coords = np.concatenate(all_coords_model)  # (N, 2) — aligned with probs
    wt_probs = np.concatenate(all_probs)             # (N, num_fates)
    dom_fate = wt_probs.argmax(axis=-1)              # (N,)

    cmap_cat = plt.cm.get_cmap("tab10", num_fates)
    fig, ax = plt.subplots(figsize=(6, 5))
    sc_plot = ax.scatter(model_coords[:, 0], model_coords[:, 1], c=dom_fate,
                         cmap=cmap_cat, s=s_size, vmin=-0.5, vmax=num_fates - 0.5,
                         edgecolors="none", rasterized=True)
    cbar = plt.colorbar(sc_plot, ax=ax)
    cbar.set_ticks(range(num_fates))
    cbar.set_ticklabels([f"Fate {f}" for f in range(num_fates)])
    fate_counts = [int((dom_fate == f).sum()) for f in range(num_fates)]
    ax.set_title("Model-Predicted Dominant Fate (WT)\n" +
                 "  ".join(f"F{f}={fate_counts[f]:,}" for f in range(num_fates)), fontsize=10)
    ax.axis("equal"); ax.axis("off")
    plt.tight_layout()
    save(fig, os.path.join(diag_dir, "spatial_wt_fate.png"))

    # Per-fate probability heatmaps (continuous probability, not just dominant)
    fig, axes = plt.subplots(1, num_fates, figsize=(5 * num_fates, 4))
    if num_fates == 1:
        axes = [axes]
    for f, ax in enumerate(axes):
        sc_plot = ax.scatter(model_coords[:, 0], model_coords[:, 1], c=wt_probs[:, f],
                             cmap="viridis", s=s_size, vmin=0, vmax=1,
                             edgecolors="none", rasterized=True)
        ax.set_title(f"Fate {f} Probability")
        ax.axis("equal"); ax.axis("off")
        plt.colorbar(sc_plot, ax=ax, fraction=0.046, pad=0.02)
    plt.suptitle("WT Fate Probability Maps", fontsize=12)
    plt.tight_layout()
    save(fig, os.path.join(diag_dir, "spatial_wt_fate_probs.png"))

    # ----------------------------------------------------------------
    # 4. Fate probability distributions (histograms)
    # ----------------------------------------------------------------
    print("Plotting fate probability distributions...")
    fig, axes = plt.subplots(1, num_fates, figsize=(4 * num_fates, 4), sharey=True)
    if num_fates == 1:
        axes = [axes]
    for f, ax in enumerate(axes):
        ax.hist(wt_probs[:, f], bins=50, color=f"C{f}", edgecolor="none")
        ax.set_title(f"Fate {f}")
        ax.set_xlabel("Probability")
        if f == 0:
            ax.set_ylabel("Cell count")
        ax.grid(True, linestyle="--", alpha=0.4)
    plt.suptitle("WT Fate Probability Distributions", fontsize=12)
    plt.tight_layout()
    save(fig, os.path.join(diag_dir, "fate_probabilities.png"))

    # ----------------------------------------------------------------
    # 5. Kinetic rates: beta, gamma, scatter
    # ----------------------------------------------------------------
    print("Plotting kinetic rates...")
    with torch.no_grad():
        beta  = model.beta.numpy()
        gamma = model.gamma.numpy()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].hist(beta, bins=40, color="#3274A1", edgecolor="none")
    axes[0].set_title(f"Splicing Rates (β)\nmean={beta.mean():.4f}  min={beta.min():.2e}")
    axes[0].set_xlabel("Rate")
    axes[0].set_ylabel("Genes")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].hist(gamma, bins=40, color="#3A923A", edgecolor="none")
    axes[1].set_title(f"Degradation Rates (γ)\nmean={gamma.mean():.4f}  min={gamma.min():.2e}")
    axes[1].set_xlabel("Rate")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    axes[2].scatter(beta, gamma, s=12, alpha=0.6, color="#555555", edgecolors="none")
    axes[2].set_xlabel("β (splicing)")
    axes[2].set_ylabel("γ (degradation)")
    axes[2].set_title("β vs γ per Gene")
    axes[2].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    save(fig, os.path.join(diag_dir, "kinetic_rates.png"))

    # ----------------------------------------------------------------
    # 6. Top 20 TFs by kinetic activity (beta + gamma)
    # ----------------------------------------------------------------
    print("Plotting top active TFs...")
    activity = beta + gamma
    top_idx = np.argsort(activity)[::-1][:20]
    top_names = [gene_names[i] for i in top_idx]
    top_vals  = activity[top_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(20), top_vals[::-1], color="#3274A1", edgecolor="none")
    ax.set_yticks(range(20))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("β + γ (kinetic activity)")
    ax.set_title("Top 20 TFs by Kinetic Activity")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    save(fig, os.path.join(diag_dir, "top_active_tfs.png"))

    print(f"\nAll diagnostics saved to: {diag_dir}")


if __name__ == "__main__":
    main()
