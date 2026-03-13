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
import pandas as pd
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


def get_model(m):
    """Unwrap DataParallel and DistributedPINNWrapper to reach the base SpatialMechanisticModel."""
    if hasattr(m, "module"): m = m.module   # unwrap DataParallel
    if hasattr(m, "model"):  m = m.model    # unwrap DistributedPINNWrapper
    return m


class DistributedPINNWrapper(nn.Module):
    """
    Thin nn.Module wrapper that moves compute_physics_loss INSIDE the DataParallel
    forward call so the Jacobian computation is distributed across all GPUs.

    DataParallel chunks both u_seq (batch dim) and collocation_t (collocation dim),
    so each GPU evaluates CME residuals at C/num_gpus collocation points independently.
    The scalar l_phys from each GPU is unsqueezed to (1,) so DataParallel can gather
    and concatenate them; .mean() in train_one_epoch gives the distributed physics loss.

    moment_scales is registered as a buffer so DataParallel automatically replicates
    it to each GPU replica — no manual .to(device) required inside forward.
    """
    def __init__(self, model, moment_scales: torch.Tensor):
        super().__init__()
        self.model = model
        # register_buffer ensures DataParallel copies moment_scales to each GPU
        self.register_buffer("moment_scales", moment_scales)

    def forward(self, u_seq, collocation_t):
        result = self.model(u_seq, collocation_t=collocation_t)

        # Clamp kinetic rates to prevent degenerate zero-rate solutions
        base = self.model.module if hasattr(self.model, "module") else self.model
        clamped_beta  = torch.clamp(base.beta,  min=1e-4)
        clamped_gamma = torch.clamp(base.gamma, min=1e-4)

        l_phys_chunk = compute_physics_loss(
            self.model, collocation_t, result,
            beta=clamped_beta, gamma=clamped_gamma,
            moment_scales=self.moment_scales,
        )
        result["l_phys"] = l_phys_chunk.unsqueeze(0)   # (1,) for DataParallel gather
        return result


def load_configs(config_path: str, loss_weights_path: str) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    with open(loss_weights_path) as f:
        loss_weights = yaml.safe_load(f)
    config.update(loss_weights)
    return config


def compute_data_loss(moments, empirical_moments, moment_scales):
    """
    Huber (Smooth L1) loss between MomentMLP predictions and empirical moments,
    designed to handle zero-inflated spatial RNA data with transcriptional burst tails.

    MSE squares large errors, causing the optimizer to hyper-focus on burst outliers
    and ignore baseline spatial dynamics (gradient starvation). Huber loss uses a
    quadratic penalty for small residuals (|error| < delta) and a linear penalty
    for large residuals (|error| >= delta), bounding the gradient contribution of
    extreme burst events.

        L_data = sum_over_dims( Huber(pred, target, delta=1.0) )

    Args:
        moments:           tuple of 5 tensors, each (..., Num_Genes)
        empirical_moments: tuple of 5 tensors, each (..., Num_Genes)
        moment_scales:     (5,) tensor — retained for API compatibility
    """
    loss = 0.0
    for i, (pred, target) in enumerate(zip(moments, empirical_moments)):
        scale = moment_scales[i]
        # Normalize the tensors to keep the gradients strictly bound
        loss = loss + F.huber_loss(pred / scale, target / scale, delta=1.0)
    return loss


def compute_physics_loss(raw_model, collocation_t, result, beta=None, gamma=None, moment_scales=None):
    """
    Evaluate the 5-ODE CME residuals at random collocation points.

    t_expanded and moments are rebuilt here — AFTER DataParallel gather — so the
    computation graph is intact and autograd.grad correctly traces moment → t_expanded.
    h_cont is a regular gathered tensor (B, C, num_tfs) used to condition moment_mlp.

    beta/gamma: pre-clamped kinetic rates. If None, reads directly from raw_model.
    moment_scales: (5,) tensor — per-dim population std for dimensionless residuals.
    """
    beta  = beta  if beta  is not None else raw_model.beta
    gamma = gamma if gamma is not None else raw_model.gamma
    h_cont = result["h_cont"]                                          # (B, C, num_tfs)
    B = h_cont.shape[0]
    C = collocation_t.shape[0]

    # Rebuild leaf node post-gather: independent gradient per (b, c) pair
    t_expanded = collocation_t.unsqueeze(0).expand(B, C, -1).clone()
    t_expanded.requires_grad_(True)                                    # (B, C, 1) fresh leaf

    # Recompute moments with this fresh t_expanded — graph is clean
    nascent_mean, mature_mean, nascent_var, mature_var, cov_nm = raw_model.moment_mlp(
        t_expanded, h_cont
    )  # each (B, C, G)

    d_nascent_mean_dt = compute_time_derivatives(t_expanded, nascent_mean)
    d_mature_mean_dt  = compute_time_derivatives(t_expanded, mature_mean)
    d_nascent_var_dt  = compute_time_derivatives(t_expanded, nascent_var)
    d_mature_var_dt   = compute_time_derivatives(t_expanded, mature_var)
    d_cov_nm_dt       = compute_time_derivatives(t_expanded, cov_nm)

    a_cont = result["burst_freq_cont"]   # (B, C, G)
    b_cont = result["burst_size_cont"]   # (B, C, G)

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
        beta=beta,
        gamma=gamma,
        moment_scales=moment_scales,
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


def train_one_epoch(model, dataloader, optimizer, config, device, epoch, moment_scales):
    model.train()
    total_loss_accum = 0.0
    data_loss_accum = 0.0
    phys_loss_accum = 0.0
    fate_loss_accum = 0.0
    n_batches = 0

    raw_model = get_model(model)

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

        # --- L_data: per-bead MomentMLP predictions vs empirical moments ---
        # MomentMLP is conditioned on h_seq so each bead gets its own trajectory.
        # bin_t: (B, S, 1) — same pseudotime grid for all beads
        # h_seq: (B, S, num_tfs) — spatially unique regulatory context per bead
        B, seq_len, _ = u_seq.shape
        bin_midpoints = torch.linspace(
            dt / 2, 1.0 - dt / 2, seq_len, device=device
        ).unsqueeze(-1)  # (S, 1)
        bin_t = bin_midpoints.unsqueeze(0).expand(B, -1, -1)   # (B, S, 1)
        h_seq = result["hidden_tfs"]                                      # (B, S, num_tfs)
        moments_at_bins = raw_model.moment_mlp(bin_t, h_seq)             # tuple of 5 x (B, S, G)
        l_data = compute_data_loss(moments_at_bins, empirical, moment_scales)

        # --- L_phys: gathered from per-GPU physics loss computed in DistributedPINNWrapper ---
        l_phys = result["l_phys"].mean()

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
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint file to resume training")
    args = parser.parse_args()

    config = load_configs(args.config, args.loss_weights)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # --- Pre-load checkpoint: override config dims before model is instantiated ---
    # Must happen before GRN build and model init so the architecture is consistent
    # with the saved weights. Without this, a config.yaml with num_terminal_fates=2
    # would build a mismatched model before the weights are ever loaded.
    start_epoch = 1
    checkpoint = None
    if args.resume is not None:
        print(f"\nPre-loading checkpoint for architecture alignment: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        for key in ["num_tfs", "num_target_genes", "num_terminal_fates", "input_spatial_dim"]:
            if key in checkpoint["config"]:
                config[key] = checkpoint["config"][key]
        start_epoch = checkpoint["epoch"] + 1
        print(f"  Architecture overrides applied:")
        print(f"    num_tfs={config['num_tfs']}, "
              f"num_target_genes={config['num_target_genes']}, "
              f"num_terminal_fates={config['num_terminal_fates']}, "
              f"input_spatial_dim={config['input_spatial_dim']}")
        print(f"  Will resume from epoch {start_epoch}")

    # --- Build frozen GRN matrix ---
    print("Loading frozen GRN matrix...")
    expressed_genes = pd.read_csv(
        "data/processed/expressed_genes.csv", header=None
    )[0].tolist()
    frozen_grn = build_frozen_grn_matrix(
        tftg_path="src/data/frozen_databases/TFTGDB.csv",
        expressed_tfs=expressed_genes,
        expressed_target_genes=expressed_genes,
    )

    # Override config with true matrix dimensions — skipped when resuming because
    # the checkpoint's architecture overrides (applied above) must be preserved.
    if args.resume is None:
        config["num_tfs"] = frozen_grn.shape[0]
        config["num_target_genes"] = frozen_grn.shape[1]

    # Override num_terminal_fates if Palantir fate probs are present in the h5ad —
    # also skipped when resuming to keep the checkpoint's num_terminal_fates intact.
    import scanpy as _sc
    _adata_peek = _sc.read_h5ad(args.h5ad, backed="r")
    if args.resume is None and "palantir_fate_probs" in _adata_peek.obsm:
        config["num_terminal_fates"] = _adata_peek.obsm["palantir_fate_probs"].shape[1]
        print(f"Detected {config['num_terminal_fates']} terminal fates from Palantir")
    _adata_peek.file.close()

    print(f"Dynamic Architecture: {config['num_tfs']} TFs -> {config['num_target_genes']} Genes, {config['num_terminal_fates']} Fates")

    # --- DataLoader (must come before model wrapping — moment_scales needed for DistributedPINNWrapper) ---
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

    # Extract moment scaling factors computed during dataset init.
    # Move to device once; reused every batch without re-allocation.
    moment_scales = dataloader.dataset.moment_scales.to(device)
    print(f"Moment scales (per-dim std): {moment_scales.tolist()}")

    # --- Initialize model ---
    model = SpatialMechanisticModel(
        input_spatial_dim=config["input_spatial_dim"],
        num_tfs=config["num_tfs"],
        num_target_genes=config["num_target_genes"],
        num_terminal_fates=config["num_terminal_fates"],
        frozen_grn_matrix=frozen_grn,
        dt=config["dt"],
    ).to(device)

    # Always wrap with DistributedPINNWrapper so result["l_phys"] is always present —
    # even on single GPU or CPU. DataParallel is added on top for multi-GPU only.
    model = DistributedPINNWrapper(model, moment_scales)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = nn.DataParallel(model)
    else:
        print(f"Single device mode ({device})")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=1e-6
    )

    # --- Load weights & optimizer state (checkpoint was pre-loaded above) ---
    best_loss = float("inf")

    if checkpoint is not None:
        print(f"\nLoading weights and optimizer state from checkpoint...")
        get_model(model).load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_loss = checkpoint["losses"]["total"]

        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print(f"  Scheduler state restored.")
        else:
            # Old checkpoints lack scheduler_state_dict — fast-forward the cosine
            # annealing to the correct decay level by replaying step() calls.
            for _ in range(start_epoch - 1):
                scheduler.step()
            print(f"  No scheduler_state_dict found — fast-forwarded {start_epoch - 1} steps.")

        print(f"  Resumed at epoch {start_epoch}  |  best_loss so far: {best_loss:.4f}")

    # --- Training loop ---
    print(f"\nStarting training for {config['epochs']} epochs")
    print(f"  lambda_data={config['lambda_data']}, "
          f"lambda_phys={config['lambda_phys']}, "
          f"lambda_fate={config['lambda_fate']}")
    print(f"  collocation_points={config['collocation_points']}, dt={config['dt']}")
    print("-" * 70)

    for epoch in range(start_epoch, config["epochs"] + 1):
        # --- Two-phase parameter freezing ---
        # Phase 1 (epochs 1–100): freeze kinetic rates so the RNN + MomentMLP
        # learn to fit the empirical moment data before the physics branch can
        # exploit the trivial zero-rate solution.
        # Phase 2 (epoch 101+): unfreeze raw_beta/raw_gamma so the splicing and
        # degradation rates can be refined jointly with the rest of the network.
        # NOTE: beta/gamma are @property wrappers around raw_beta/raw_gamma;
        #       requires_grad must be toggled on the raw leaf parameters.
        raw_model = get_model(model)
        if epoch <= 100:
            raw_model.raw_beta.requires_grad_(False)
            raw_model.raw_gamma.requires_grad_(False)
        else:
            raw_model.raw_beta.requires_grad_(True)
            raw_model.raw_gamma.requires_grad_(True)

        losses = train_one_epoch(model, dataloader, optimizer, config, device, epoch, moment_scales)
        scheduler.step()

        if epoch == 101:
            print("-" * 70)
            print("Phase 2: raw_beta / raw_gamma unfrozen — kinetic rates now trainable")
            print("-" * 70)

        if epoch % 10 == 0 or epoch == 1:
            raw_model = get_model(model)
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:4d} | "
                f"Total: {losses['total']:.4f} | "
                f"Data: {losses['data']:.4f} | "
                f"Phys: {losses['phys']:.4f} | "
                f"Fate: {losses['fate']:.4f} | "
                f"LR: {lr:.2e} | "
                f"beta_mean: {raw_model.beta.mean().item():.4f} | "
                f"gamma_mean: {raw_model.gamma.mean().item():.4f}"
            )

        # Save best checkpoint (always save raw_model state for clean loading)
        if losses["total"] < best_loss:
            best_loss = losses["total"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": get_model(model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "losses": losses,
                "config": config,
                "moment_scales": moment_scales.cpu(),
            }, os.path.join(args.checkpoint_dir, "best_model.pt"))

        # Periodic checkpoint every 100 epochs
        if epoch % 100 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": get_model(model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "losses": losses,
                "config": config,
                "moment_scales": moment_scales.cpu(),
            }, os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"))

    print("-" * 70)
    print(f"Training complete. Best total loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}/")


if __name__ == "__main__":
    main()
