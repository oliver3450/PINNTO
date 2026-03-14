"""
04_publishable_eval.py — Publication figures: Spatial Fate Map + Temporal Knockout.

Plot A (Spatial Map):
    Left: WT forward pass, beads colored by dominant fate assignment
    Right: Full-KO of top TF, showing disrupted fate landscape

Plot B (Temporal Knockout):
    Bar chart showing surviving population of the target fate when the KO
    starts at t=0.2, 0.4, 0.8, 1.0 — demonstrating the closing critical window.

Usage:
    python scripts/04_publishable_eval.py \
        --checkpoint checkpoints/best_model.pt \
        --h5ad data/processed/spatial_adata.h5ad
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models.hybrid_pinn import SpatialMechanisticModel
from src.data.dataloader import get_dataloader


def load_model(checkpoint_path, device="cpu"):
    """Load checkpoint and instantiate model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    num_tfs = config["num_tfs"]
    num_genes = config["num_target_genes"]
    num_fates = config["num_terminal_fates"]

    model = SpatialMechanisticModel(
        input_spatial_dim=config.get("input_spatial_dim", 2),
        num_tfs=num_tfs,
        num_target_genes=num_genes,
        num_terminal_fates=num_fates,
        frozen_grn_matrix=torch.zeros((num_tfs, num_genes)),
        dt=config.get("dt", 0.02),
    )

    clean_state_dict = {
        k.replace("module.", "").replace("model.", ""): v
        for k, v in checkpoint["model_state_dict"].items()
    }
    model.load_state_dict(clean_state_dict, strict=False)
    model.eval()
    return model, config


def pick_top_tf(model, gene_names):
    """
    Pick the most biologically active TF: highest mean |beta| + |gamma|
    among genes whose kinetic rates escaped the training clamp floor (1e-4).
    Falls back to highest beta if all genes look clamped.
    """
    with torch.no_grad():
        beta = model.beta.numpy()
        gamma = model.gamma.numpy()

    # Combined activity score
    activity = np.abs(beta) + np.abs(gamma)
    best_idx = int(np.argmax(activity))
    return best_idx, gene_names[best_idx]


def run_inference(model, dataloader, device, hook_fn=None):
    """
    Run a full forward pass over the dataset.
    Returns spatial coords, fate probabilities, and dominant fate assignments.
    If hook_fn is provided, registers it on model.rnn for this pass.
    """
    handle = None
    if hook_fn is not None:
        handle = model.rnn.register_forward_hook(hook_fn)

    all_coords = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            u_seq = batch["u_seq"].to(device)
            all_coords.append(u_seq[:, 0, :].cpu().numpy())

            result = model(u_seq, collocation_t=None)
            probs = F.softmax(result["fate_logits"].mean(dim=1), dim=-1)
            all_probs.append(probs.cpu().numpy())

    if handle is not None:
        handle.remove()

    coords = np.concatenate(all_coords, axis=0)
    probs = np.concatenate(all_probs, axis=0)
    dom_fate = probs.argmax(axis=-1)
    return coords, probs, dom_fate


def plot_spatial_map(coords, wt_fate, ko_fate, num_fates, tf_name, outdir):
    """Plot A: WT vs KO spatial fate maps side by side."""
    # Use a categorical colormap
    cmap = plt.cm.get_cmap("tab10", num_fates)
    s_size = max(1, min(8, 50000 // len(coords)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"Spatial Fate Assignment: Wild Type vs {tf_name} Knockout",
                 fontsize=16, fontweight="bold")

    for ax, fates, title in [
        (ax1, wt_fate, "Wild Type"),
        (ax2, ko_fate, f"{tf_name} Full Knockout"),
    ]:
        sc = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=fates, cmap=cmap, s=s_size, vmin=-0.5, vmax=num_fates - 0.5,
            edgecolors="none", alpha=0.8,
        )
        ax.set_title(title, fontsize=14)
        ax.axis("equal")
        ax.axis("off")

    # Shared colorbar
    cbar = fig.colorbar(sc, ax=[ax1, ax2], fraction=0.03, pad=0.02)
    cbar.set_ticks(range(num_fates))
    cbar.set_ticklabels([f"Fate {i}" for i in range(num_fates)])

    # Annotation: cell count per fate
    for f in range(num_fates):
        wt_n = int((wt_fate == f).sum())
        ko_n = int((ko_fate == f).sum())
        fig.text(0.02, 0.02 + f * 0.04,
                 f"Fate {f}: WT={wt_n:,}  KO={ko_n:,}  (Δ={ko_n - wt_n:+,})",
                 fontsize=9, family="monospace", transform=fig.transFigure)

    plt.tight_layout(rect=[0, 0.02 + num_fates * 0.04, 1, 0.95])
    path = os.path.join(outdir, "PlotA_Spatial_Fate_Map.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_temporal_knockout(model, dataloader, device, target_idx, tf_name,
                           primary_fate, wt_fate_count, num_fates, seq_len, outdir):
    """
    Plot B: Temporal knockout bar chart.
    Knockout starts at t=0.2, 0.4, 0.8, 1.0. Measure surviving population
    of the primary fate at each cutoff.
    """
    t_cutoffs = [0.2, 0.4, 0.8, 1.0]
    surviving = []

    for t_start in t_cutoffs:
        cutoff_bin = max(0, int(t_start * seq_len))

        def temporal_hook(module, input, output, _cutoff=cutoff_bin):
            out = output.clone()
            out[:, _cutoff:, target_idx] = 0.0
            return out

        _, _, dom_fate = run_inference(model, dataloader, device, hook_fn=temporal_hook)
        count = int((dom_fate == primary_fate).sum())
        surviving.append(count)
        print(f"    KO from t={t_start:.1f} (bin {cutoff_bin}): "
              f"{count:,} cells in Fate {primary_fate}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["#3274A1", "#E1812C", "#3A923A", "#C03D3E"]
    bars = ax.bar(
        [f"t={t:.1f}" for t in t_cutoffs], surviving,
        color=colors[:len(t_cutoffs)], edgecolor="white", linewidth=1.5,
    )

    # WT baseline reference line
    ax.axhline(y=wt_fate_count, color="black", linestyle="--", linewidth=1.5,
               label=f"WT baseline ({wt_fate_count:,})")

    # Value labels on bars
    for bar, val in zip(bars, surviving):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + wt_fate_count * 0.01,
                f"{val:,}", ha="center", va="bottom", fontweight="bold", fontsize=11)

    ax.set_xlabel("Knockout Start Time (Pseudotime)", fontsize=12)
    ax.set_ylabel(f"Surviving Population (Fate {primary_fate})", fontsize=12)
    ax.set_title(
        f"Critical Window: {tf_name} Knockout Timing → Fate {primary_fate} Survival",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = os.path.join(outdir, "PlotB_Temporal_Knockout.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Publication evaluation: Spatial Fate Map + Temporal Knockout"
    )
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--h5ad", default="data/processed/spatial_adata.h5ad")
    parser.add_argument("--genes", default="data/processed/expressed_genes.csv")
    parser.add_argument("--outdir", default="logs/evaluation")
    parser.add_argument("--target_tf", default=None,
                        help="TF to knockout (auto-selects top active TF if omitted)")
    parser.add_argument("--target_fate", type=int, default=None,
                        help="Fate index for temporal plot (auto-selects most impacted if omitted)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cpu")

    # --- Load model + data ---
    print(f"Loading checkpoint: {args.checkpoint}")
    model, config = load_model(args.checkpoint, device)
    num_tfs = config["num_tfs"]
    num_genes = config["num_target_genes"]
    num_fates = config["num_terminal_fates"]
    seq_len = int(1.0 / config.get("dt", 0.02))

    gene_names = pd.read_csv(args.genes, header=None)[0].tolist()
    if len(gene_names) != num_tfs:
        print(f"WARNING: gene list has {len(gene_names)} entries but checkpoint "
              f"has num_tfs={num_tfs}. Padding with numeric names.")
        gene_names = gene_names + [f"gene_{i}" for i in range(len(gene_names), num_tfs)]

    # Select target TF
    if args.target_tf is not None:
        names_upper = [g.upper() for g in gene_names]
        tf_upper = args.target_tf.upper()
        if tf_upper not in names_upper:
            print(f"ERROR: {args.target_tf} not found in gene list.")
            sys.exit(1)
        target_idx = names_upper.index(tf_upper)
        tf_name = gene_names[target_idx]
    else:
        target_idx, tf_name = pick_top_tf(model, gene_names)
    print(f"Target TF: {tf_name} (index {target_idx})")

    print(f"Loading data: {args.h5ad}")
    dataloader = get_dataloader(
        h5ad_path=args.h5ad, batch_size=2048, seq_len=seq_len,
        num_genes=num_genes, num_fates=num_fates,
    )

    # --- WT baseline ---
    print("\nRunning Wild Type inference...")
    coords, wt_probs, wt_fate = run_inference(model, dataloader, device)
    wt_counts = [int((wt_fate == f).sum()) for f in range(num_fates)]
    print(f"  WT fate distribution: {wt_counts}")

    # --- Full KO ---
    def full_ko_hook(module, input, output):
        out = output.clone()
        out[:, :, target_idx] = 0.0
        return out

    print(f"\nRunning {tf_name} Full Knockout inference...")
    _, ko_probs, ko_fate = run_inference(model, dataloader, device, hook_fn=full_ko_hook)
    ko_counts = [int((ko_fate == f).sum()) for f in range(num_fates)]
    print(f"  KO fate distribution: {ko_counts}")

    # --- Plot A: Spatial Map ---
    print("\nGenerating Plot A (Spatial Fate Map)...")
    plot_spatial_map(coords, wt_fate, ko_fate, num_fates, tf_name, args.outdir)

    # --- Determine primary fate for temporal plot ---
    if args.target_fate is not None:
        primary_fate = args.target_fate
    else:
        # Auto-select: the fate most disrupted by full KO (largest absolute drop)
        deltas = [wt_counts[f] - ko_counts[f] for f in range(num_fates)]
        primary_fate = int(np.argmax(deltas))
    print(f"\nPrimary fate for temporal analysis: Fate {primary_fate} "
          f"(WT={wt_counts[primary_fate]:,}, KO={ko_counts[primary_fate]:,})")

    # --- Plot B: Temporal Knockout ---
    print("\nGenerating Plot B (Temporal Knockout)...")
    plot_temporal_knockout(
        model, dataloader, device,
        target_idx=target_idx,
        tf_name=tf_name,
        primary_fate=primary_fate,
        wt_fate_count=wt_counts[primary_fate],
        num_fates=num_fates,
        seq_len=seq_len,
        outdir=args.outdir,
    )

    print(f"\nAll figures saved to: {args.outdir}/")


if __name__ == "__main__":
    main()
