"""
08_plot_spatial_ko_atlas.py — Spatial delta-probability maps for every TF knockout.

For each transcription factor: zeroes its RNN hidden state output (full KO),
computes ΔP = P_KO - P_WT per fate, and saves a spatial scatter plot.

Produces:
  <outdir>/ko_atlas/<TF_name>.png    — one per TF, num_fates subplots
  <outdir>/ko_atlas/summary.csv      — mean |ΔP| per TF per fate (for ranking)

Usage:
    python scripts/diagnostics/08_plot_spatial_ko_atlas.py \
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
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def full_pass(model, dataloader, device, hook_fn=None):
    """Single forward pass over full dataset. Returns (coords, probs)."""
    handle = model.rnn.register_forward_hook(hook_fn) if hook_fn else None
    coords_list, probs_list = [], []
    with torch.no_grad():
        for batch in dataloader:
            u = batch["u_seq"].to(device)
            coords_list.append(u[:, 0, :].cpu().numpy())
            out = model(u, collocation_t=None)
            probs_list.append(F.softmax(out["fate_logits"].mean(dim=1), dim=-1).cpu().numpy())
    if handle:
        handle.remove()
    return np.concatenate(coords_list), np.concatenate(probs_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--h5ad", default="data/processed/spatial_adata.h5ad")
    parser.add_argument("--genes", default="data/processed/expressed_genes.csv")
    parser.add_argument("--outdir", default="logs/evaluation")
    args = parser.parse_args()

    atlas_dir = os.path.join(args.outdir, "ko_atlas")
    os.makedirs(atlas_dir, exist_ok=True)
    device = torch.device("cpu")

    model, cfg = load_model(args.checkpoint, device)
    num_tfs   = cfg["num_tfs"]
    num_fates = cfg["num_terminal_fates"]
    seq_len   = int(1.0 / cfg.get("dt", 0.02))

    gene_names = pd.read_csv(args.genes, header=None)[0].tolist()
    if len(gene_names) < num_tfs:
        gene_names += [f"gene_{i}" for i in range(len(gene_names), num_tfs)]
    gene_names = gene_names[:num_tfs]

    dataloader = get_dataloader(
        h5ad_path=args.h5ad, batch_size=2048, seq_len=seq_len,
        num_genes=cfg["num_target_genes"], num_fates=num_fates,
        shuffle=False, drop_last=False,
    )

    print("Running WT baseline...")
    coords, wt_probs = full_pass(model, dataloader, device)
    s_size = max(1, min(8, 40000 // len(coords)))

    summary_rows = []

    print(f"Running KO for {num_tfs} TFs...")
    for idx in range(num_tfs):
        tf_name = gene_names[idx]

        def ko_hook(module, input, output, _idx=idx):
            out = output.clone()
            out[:, :, _idx] = 0.0
            return out

        _, ko_probs = full_pass(model, dataloader, device, hook_fn=ko_hook)
        delta = ko_probs - wt_probs  # (N, num_fates)

        # Summary row
        row = {"TF": tf_name}
        for f in range(num_fates):
            row[f"mean_abs_delta_fate{f}"] = float(np.abs(delta[:, f]).mean())
        row["mean_abs_delta_all"] = float(np.abs(delta).mean())
        summary_rows.append(row)

        # Figure: one subplot per fate showing delta_P
        vmax = max(float(np.abs(delta).max()), 1e-6)
        fig, axes = plt.subplots(1, num_fates, figsize=(4 * num_fates, 4))
        if num_fates == 1:
            axes = [axes]
        fig.suptitle(f"{tf_name} KO  |  mean|ΔP|={row['mean_abs_delta_all']:.4f}",
                     fontsize=11)
        for f, ax in enumerate(axes):
            sc = ax.scatter(coords[:, 0], coords[:, 1],
                            c=delta[:, f], cmap="coolwarm",
                            s=s_size, vmin=-vmax, vmax=vmax,
                            edgecolors="none", rasterized=True)
            ax.set_title(f"Fate {f}  Δ={delta[:, f].mean():+.4f}", fontsize=9)
            ax.axis("equal")
            ax.axis("off")
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)

        plt.tight_layout()
        plt.savefig(os.path.join(atlas_dir, f"{tf_name}.png"), dpi=120, bbox_inches="tight")
        plt.close()

        if (idx + 1) % 10 == 0 or idx == num_tfs - 1:
            print(f"  [{idx+1}/{num_tfs}] {tf_name}")

    # Save summary CSV sorted by overall impact
    summary_df = pd.DataFrame(summary_rows).sort_values("mean_abs_delta_all", ascending=False)
    csv_path = os.path.join(atlas_dir, "summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\nTop 10 most impactful TFs:")
    print(summary_df.head(10)[["TF", "mean_abs_delta_all"]].to_string(index=False))
    print(f"\nSaved {num_tfs} figures + summary to: {atlas_dir}")


if __name__ == "__main__":
    main()
