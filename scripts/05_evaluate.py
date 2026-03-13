"""
05_evaluate.py — Post-training evaluation of the Spatial Mechanistic PINN.

Loads a trained checkpoint, runs inference on the full dataset, and computes:
  - Gene-level R² for all 5 moment types
  - Physics residual statistics at dense collocation points
  - Fate prediction accuracy and calibration
  - Kinetic rate (beta/gamma) summaries

Outputs:
  - evaluation_summary.json   — all metrics in one file
  - gene_r2_scores.csv        — per-gene R² for each moment type
  - kinetic_rates.csv         — learned beta/gamma per gene

Usage:
    python scripts/05_evaluate.py \
        --checkpoint results/.../checkpoints/best_model.pt \
        --h5ad data/processed/spatial_adata.h5ad \
        --output_dir results/.../evaluation/
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import torch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.hybrid_pinn import SpatialMechanisticModel
from src.physics.cme_equations import compute_cme_residuals
from src.physics.autograd import compute_time_derivatives
from src.data.regulatory_networks import build_frozen_grn_matrix
from src.data.dataloader import get_dataloader
from src.utils.metrics import (
    gene_r2, moment_mse, moment_r2_summary,
    physics_residual_stats, kinetic_rate_summary, fate_accuracy,
)


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]

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

    # Restore moment_scales saved at training time (per-dim population std)
    moment_scales = ckpt.get("moment_scales", None)
    if moment_scales is not None:
        moment_scales = moment_scales.to(device)
        print(f"  Moment scales: {moment_scales.tolist()}")
    else:
        print("  WARNING: No moment_scales in checkpoint — physics residuals will be unscaled")

    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")
    print(f"  Training losses: {ckpt['losses']}")
    return model, config, expressed_genes, moment_scales


@torch.no_grad()
def evaluate_data_fit(model, dataloader, config, device):
    """Evaluate moment prediction accuracy across the full dataset."""
    all_pred = [[] for _ in range(5)]
    all_target = [[] for _ in range(5)]
    all_fate_logits = []
    all_fate_targets = []

    seq_len = int(1.0 / config["dt"])
    dt = config["dt"]
    bin_midpoints = torch.linspace(dt / 2, 1.0 - dt / 2, seq_len, device=device)

    for batch in dataloader:
        u_seq = batch["u_seq"].to(device)
        empirical = tuple(m.to(device) for m in batch["empirical_moments"])
        fate_targets = batch["fate_targets"].to(device)

        result = model(u_seq)
        h_seq = result["hidden_tfs"]
        B, S, _ = u_seq.shape

        bin_t = bin_midpoints.unsqueeze(-1).unsqueeze(0).expand(B, S, 1)
        moments = model.moment_mlp(bin_t, h_seq)

        for i in range(5):
            all_pred[i].append(moments[i].cpu().numpy())
            all_target[i].append(empirical[i].cpu().numpy())

        all_fate_logits.append(result["fate_logits"].cpu())
        all_fate_targets.append(fate_targets.cpu())

    # Concatenate and flatten to (N, G)
    pred_flat = tuple(
        np.concatenate(p, axis=0).reshape(-1, p[0].shape[-1]) for p in all_pred
    )
    target_flat = tuple(
        np.concatenate(t, axis=0).reshape(-1, t[0].shape[-1]) for t in all_target
    )
    fate_logits_cat = torch.cat(all_fate_logits, dim=0).reshape(-1, all_fate_logits[0].shape[-1])
    fate_targets_cat = torch.cat(all_fate_targets, dim=0).reshape(-1, all_fate_targets[0].shape[-1])

    return pred_flat, target_flat, fate_logits_cat, fate_targets_cat


def evaluate_physics(model, config, device, n_collocation=500, moment_scales=None):
    """Evaluate physics residuals at dense collocation points using a single dummy batch.

    moment_scales: (5,) tensor from training — if provided, residuals are scaled
                   identically to training so the reported value is directly comparable.
    """
    seq_len = int(1.0 / config["dt"])

    # Create a dummy spatial input to get h_cont
    dummy_u = torch.randn(1, seq_len, config["input_spatial_dim"], device=device)
    collocation_t = torch.linspace(0.01, 0.99, n_collocation, device=device).unsqueeze(-1)
    collocation_t.requires_grad_(True)

    result = model(dummy_u, collocation_t=collocation_t)
    h_cont = result["h_cont"]
    B = h_cont.shape[0]
    C = collocation_t.shape[0]

    t_expanded = collocation_t.unsqueeze(0).expand(B, C, -1).clone()
    t_expanded.requires_grad_(True)

    nascent_mean, mature_mean, nascent_var, mature_var, cov_nm = model.moment_mlp(
        t_expanded, h_cont
    )

    d_nascent_mean_dt = compute_time_derivatives(t_expanded, nascent_mean)
    d_mature_mean_dt = compute_time_derivatives(t_expanded, mature_mean)
    d_nascent_var_dt = compute_time_derivatives(t_expanded, nascent_var)
    d_mature_var_dt = compute_time_derivatives(t_expanded, mature_var)
    d_cov_nm_dt = compute_time_derivatives(t_expanded, cov_nm)

    beta = torch.clamp(model.beta, min=1e-4)
    gamma = torch.clamp(model.gamma, min=1e-4)

    physics_loss = compute_cme_residuals(
        nascent_mean=nascent_mean,
        mature_mean=mature_mean,
        nascent_var=nascent_var,
        mature_var=mature_var,
        cov_nm=cov_nm,
        d_nascent_mean_dt=d_nascent_mean_dt,
        d_mature_mean_dt=d_mature_mean_dt,
        d_nascent_var_dt=d_nascent_var_dt,
        d_mature_var_dt=d_mature_var_dt,
        d_cov_nm_dt=d_cov_nm_dt,
        a_t=result["burst_freq_cont"],
        b_t=result["burst_size_cont"],
        beta=beta,
        gamma=gamma,
        moment_scales=moment_scales,
    )

    return physics_loss


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained PINN model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--h5ad", default="data/processed/spatial_adata.h5ad")
    parser.add_argument("--output_dir", default="results/evaluation/")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load Model ---
    model, config, gene_names, moment_scales = load_model(args.checkpoint, device)
    num_genes = model.num_target_genes

    # --- Data Fit ---
    print("\nEvaluating data fit ...")
    seq_len = int(1.0 / config["dt"])
    dataloader = get_dataloader(
        h5ad_path=args.h5ad,
        batch_size=args.batch_size,
        seq_len=seq_len,
        num_genes=num_genes,
        num_fates=config["num_terminal_fates"],
        shuffle=False,
    )

    pred_moments, target_moments, fate_logits, fate_targets = evaluate_data_fit(
        model, dataloader, config, device
    )

    mse_results = moment_mse(pred_moments, target_moments)
    r2_results = moment_r2_summary(pred_moments, target_moments)
    fate_results = fate_accuracy(fate_logits, fate_targets)

    print(f"  Moment MSE: {mse_results['total_mse']:.6f}")
    print(f"  Mature mean median R²: {r2_results['mature_mean_median_r2']:.4f}")
    print(f"  Fate hard accuracy: {fate_results['fate_hard_accuracy']:.4f}")

    # --- Physics Fit ---
    print("\nEvaluating physics residuals ...")
    model.eval()
    # Need grad for physics evaluation
    for p in model.parameters():
        p.requires_grad_(False)

    physics_loss = evaluate_physics(model, config, device, moment_scales=moment_scales)
    phys_results = physics_residual_stats(physics_loss)
    print(f"  Physics loss: {phys_results.get('physics_loss', phys_results.get('physics_mean', 'N/A'))}")

    # --- Kinetic Rates ---
    kinetic_results = kinetic_rate_summary(model.beta, model.gamma)
    print(f"  Beta mean: {kinetic_results['beta_mean']:.4f}")
    print(f"  Gamma mean: {kinetic_results['gamma_mean']:.4f}")
    print(f"  Beta/Gamma ratio: {kinetic_results['beta_gamma_ratio_mean']:.4f}")

    # --- Per-Gene R² ---
    moment_names = ["nascent_mean", "mature_mean", "nascent_var", "mature_var", "cov_nm"]
    gene_labels = gene_names[:num_genes] if len(gene_names) >= num_genes else \
                  [f"Gene_{i}" for i in range(num_genes)]
    r2_df = pd.DataFrame(index=gene_labels)
    for name, pred, target in zip(moment_names, pred_moments, target_moments):
        r2_df[f"{name}_r2"] = gene_r2(pred, target)

    r2_df.to_csv(os.path.join(args.output_dir, "gene_r2_scores.csv"))
    print(f"\n  Per-gene R² saved ({r2_df.shape})")

    # --- Kinetic Rates per Gene ---
    rates_df = pd.DataFrame({
        "gene": gene_labels,
        "beta": model.beta.detach().cpu().numpy(),
        "gamma": model.gamma.detach().cpu().numpy(),
        "beta_gamma_ratio": (model.beta / (model.gamma + 1e-8)).detach().cpu().numpy(),
    })
    rates_df.to_csv(os.path.join(args.output_dir, "kinetic_rates.csv"), index=False)

    # --- Combined Summary ---
    all_metrics = {}
    all_metrics.update(mse_results)
    all_metrics.update(r2_results)
    all_metrics.update(phys_results)
    all_metrics.update(kinetic_results)
    all_metrics.update(fate_results)

    with open(os.path.join(args.output_dir, "evaluation_summary.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n--- Evaluation Complete ---")
    print(f"  Results saved to: {args.output_dir}")
    print(f"  Files: evaluation_summary.json, gene_r2_scores.csv, kinetic_rates.csv")


if __name__ == "__main__":
    main()
