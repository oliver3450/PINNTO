"""
03_train_model.py — Main training loop for the Spatial Mechanistic PINN.

Orchestrates the three-loss optimization:
  L_total = lambda_data * L_data + lambda_phys * L_phys + lambda_fate * L_fate

Usage:
  python scripts/03_train_model.py \
      --config configs/train_config.yaml \
      --loss_weights configs/loss_weights.yaml
"""

import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.hybrid_pinn import SpatialMechanisticModel
from src.physics.cme_equations import compute_cme_residuals
from src.physics.autograd import compute_time_derivatives
from src.data.regulatory_networks import build_frozen_grn_matrix
from src.data.dataloader import get_dataloader


def load_configs(config_path: str, loss_weights_path: str) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    with open(loss_weights_path) as f:
        loss_weights = yaml.safe_load(f)
    config.update(loss_weights)
    return config


def compute_data_loss(moments, empirical_moments):
    """
    MSE between the MomentMLP predictions at bin midpoints and
    the lineage-weighted empirical RNA statistics.

    Args:
        moments:           tuple of 5 tensors, each (Num_Bins, Num_Genes)
        empirical_moments: tuple of 5 tensors, each (Num_Bins, Num_Genes)
    """
    loss = 0.0
    for pred, target in zip(moments, empirical_moments):
        loss = loss + F.mse_loss(pred, target)
    return loss


def compute_physics_loss(model, collocation_t, result):
    """
    Evaluate the 5-ODE CME residuals at random collocation points.

    The MomentMLP predictions and their time-derivatives are computed here.
    The burst parameters a(t), b(t) come from the RNN via differentiable
    linear interpolation (already computed in the forward pass).
    """
    nascent_mean, mature_mean, nascent_var, mature_var, cov_nm = result["moments"]

    # Compute d/dt of each moment via autograd through the MomentMLP
    d_nascent_mean_dt = compute_time_derivatives(collocation_t, nascent_mean)
    d_mature_mean_dt = compute_time_derivatives(collocation_t, mature_mean)
    d_nascent_var_dt = compute_time_derivatives(collocation_t, nascent_var)
    d_mature_var_dt = compute_time_derivatives(collocation_t, mature_var)
    d_cov_nm_dt = compute_time_derivatives(collocation_t, cov_nm)

    # The interpolated burst parameters are already batch-expanded:
    # shape (Batch, Num_Collocation, Num_Genes) — average over batch dim
    a_cont = result["burst_freq_cont"].mean(dim=0)   # (C, G)
    b_cont = result["burst_size_cont"].mean(dim=0)    # (C, G)

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
        a_t=a_cont,
        b_t=b_cont,
        beta=model.beta,
        gamma=model.gamma,
    )

    return physics_loss


def compute_fate_loss(fate_logits, fate_targets):
    """
    Cross-entropy between predicted fate logits and Palantir target probabilities.

    Args:
        fate_logits:  (Batch, Seq_Len, Num_Fates) — raw logits, no softmax
        fate_targets: (Batch, Seq_Len, Num_Fates) — soft probability targets from Palantir

    Uses soft cross-entropy (KL divergence) since targets are probabilities, not hard labels.
    """
    # Reshape to (Batch*Seq_Len, Num_Fates) for loss computation
    B, S, F_dim = fate_logits.shape
    logits_flat = fate_logits.reshape(B * S, F_dim)
    targets_flat = fate_targets.reshape(B * S, F_dim)

    # Soft cross-entropy: -sum(target * log_softmax(logits))
    log_probs = torch.log_softmax(logits_flat, dim=-1)
    loss = -(targets_flat * log_probs).sum(dim=-1).mean()

    return loss


def train_one_epoch(model, dataloader, optimizer, config, device, epoch):
    model.train()
    total_loss_accum = 0.0
    data_loss_accum = 0.0
    phys_loss_accum = 0.0
    fate_loss_accum = 0.0
    n_batches = 0

    lambda_data = config["lambda_data"]
    lambda_phys = config["lambda_phys"]
    lambda_fate = config["lambda_fate"]
    num_collocation = config["collocation_points"]
    dt = config["dt"]

    for batch in dataloader:
        u_seq = batch["u_seq"].to(device)               # (B, S, D_spatial)
        empirical = tuple(m.to(device) for m in batch["empirical_moments"])  # 5 x (S, G)
        fate_targets = batch["fate_targets"].to(device)  # (B, S, Num_Fates)

        # Sample random collocation points in [0, 1] for physics
        collocation_t = torch.rand(num_collocation, 1, device=device, requires_grad=True)

        # --- Forward pass ---
        result = model(u_seq, collocation_t=collocation_t)

        # --- L_data: MomentMLP at bin midpoints vs empirical weighted moments ---
        seq_len = u_seq.shape[1]
        bin_midpoints = torch.linspace(
            dt / 2, 1.0 - dt / 2, seq_len, device=device
        ).unsqueeze(-1)  # (S, 1)
        moments_at_bins = model.moment_mlp(bin_midpoints)
        l_data = compute_data_loss(moments_at_bins, empirical)

        # --- L_phys: CME residuals at collocation points ---
        l_phys = compute_physics_loss(model, collocation_t, result)

        # --- L_fate: Cross-entropy at every time step ---
        l_fate = compute_fate_loss(result["fate_logits"], fate_targets)

        # --- Total loss ---
        loss = lambda_data * l_data + lambda_phys * l_phys + lambda_fate * l_fate

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss_accum += loss.item()
        data_loss_accum += l_data.item()
        phys_loss_accum += l_phys.item()
        fate_loss_accum += l_fate.item()
        n_batches += 1

    avg = lambda x: x / max(n_batches, 1)
    return {
        "total": avg(total_loss_accum),
        "data": avg(data_loss_accum),
        "phys": avg(phys_loss_accum),
        "fate": avg(fate_loss_accum),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--loss_weights", required=True)
    parser.add_argument("--h5ad", default="data/processed/spatial_adata.h5ad",
                        help="Path to processed .h5ad with spliced/unspliced layers")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    args = parser.parse_args()

    config = load_configs(args.config, args.loss_weights)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # --- Build frozen GRN matrix ---
    # TODO: Replace with actual expressed TF/gene lists from your preprocessed data
    # These would come from 01_run_palantir.py and 02_build_spatial.py outputs
    print("Loading frozen GRN matrix...")
    # frozen_grn = build_frozen_grn_matrix(
    #     tftg_path="src/data/frozen_databases/TFTGDB.csv",
    #     expressed_tfs=expressed_tfs,
    #     expressed_target_genes=expressed_genes,
    # )
    # Placeholder until preprocessing scripts populate real gene lists
    frozen_grn = torch.randn(config["num_tfs"], config["num_target_genes"])

    # --- Initialize model ---
    model = SpatialMechanisticModel(
        input_spatial_dim=config["input_spatial_dim"],
        num_tfs=config["num_tfs"],
        num_target_genes=config["num_target_genes"],
        num_terminal_fates=config["num_terminal_fates"],
        frozen_grn_matrix=frozen_grn,
        dt=config["dt"],
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=1e-6
    )

    # --- DataLoader ---
    seq_len = int(1.0 / config["dt"])  # e.g. 50 steps if dt=0.02
    print(f"Loading dataset from: {args.h5ad}")
    dataloader = get_dataloader(
        h5ad_path=args.h5ad,
        batch_size=config["batch_size"],
        seq_len=seq_len,
        num_genes=config["num_target_genes"],
        num_fates=config["num_terminal_fates"],
    )
    print(f"Dataset loaded: {len(dataloader.dataset):,} beads, seq_len={seq_len}")

    # --- Training loop ---
    print(f"\nStarting training for {config['epochs']} epochs")
    print(f"  lambda_data={config['lambda_data']}, "
          f"lambda_phys={config['lambda_phys']}, "
          f"lambda_fate={config['lambda_fate']}")
    print(f"  collocation_points={config['collocation_points']}, dt={config['dt']}")
    print("-" * 70)

    best_loss = float("inf")

    for epoch in range(1, config["epochs"] + 1):
        losses = train_one_epoch(model, dataloader, optimizer, config, device, epoch)
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:4d} | "
                f"Total: {losses['total']:.4f} | "
                f"Data: {losses['data']:.4f} | "
                f"Phys: {losses['phys']:.4f} | "
                f"Fate: {losses['fate']:.4f} | "
                f"LR: {lr:.2e} | "
                f"beta_mean: {model.beta.mean().item():.4f} | "
                f"gamma_mean: {model.gamma.mean().item():.4f}"
            )

        # Save best checkpoint
        if losses["total"] < best_loss:
            best_loss = losses["total"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "losses": losses,
                "config": config,
            }, os.path.join(args.checkpoint_dir, "best_model.pt"))

        # Periodic checkpoint every 100 epochs
        if epoch % 100 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "losses": losses,
                "config": config,
            }, os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"))

    print("-" * 70)
    print(f"Training complete. Best total loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}/")


if __name__ == "__main__":
    main()
