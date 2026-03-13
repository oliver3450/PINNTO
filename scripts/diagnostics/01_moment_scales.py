"""
diagnostics/01_moment_scales.py — Profile the raw scale of empirical moments.

Reproduces exactly the moment construction logic from SpatialTranscriptomicsDataset
without building a DataLoader, so you can inspect the full dataset statistics.

Outputs:
  - Console table: min/max/mean/std for each of the 5 moment dimensions
  - logs/moment_distributions.png: multi-panel histogram of all 5 moments

Usage:
    python scripts/diagnostics/01_moment_scales.py \
        --h5ad data/processed/spatial_adata.h5ad \
        --num_genes 2000 \
        --sample_beads 2000
"""

import argparse
import os
import sys
import numpy as np
import scanpy as sc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


def compute_empirical_moments(h5ad_path, num_genes, seq_len, sample_beads=None):
    """
    Reproduce the dataloader's moment construction for the full dataset.
    Returns arrays of shape (N_sampled_beads, seq_len, num_genes) for each moment.
    """
    from sklearn.neighbors import NearestNeighbors

    print(f"Loading {h5ad_path} ...")
    adata = sc.read_h5ad(h5ad_path)
    print(f"  {adata.n_obs:,} beads x {adata.n_vars:,} genes")
    print(f"  Available layers: {list(adata.layers.keys())}")

    # Spatial coordinates — same fallback logic as dataloader
    SPATIAL_KEYS = ["spatial", "X_xy_loc", "xy_loc", "X_spatial", "coordinates"]
    spatial_key = next((k for k in SPATIAL_KEYS if k in adata.obsm), None)
    if spatial_key is None:
        raise KeyError(f"No spatial key found. Available obsm: {list(adata.obsm.keys())}")
    coords = adata.obsm[spatial_key]
    print(f"  Spatial coords from '{spatial_key}': {coords.shape}")

    # RNA layers
    M_raw = adata.layers["spliced"][:, :num_genes]
    N_raw = adata.layers["unspliced"][:, :num_genes]
    if hasattr(M_raw, "toarray"):
        M_raw = M_raw.toarray()
        N_raw = N_raw.toarray()
    M_raw = np.asarray(M_raw, dtype=np.float64)
    N_raw = np.asarray(N_raw, dtype=np.float64)

    actual_genes = M_raw.shape[1]
    print(f"  Using {actual_genes} genes, seq_len={seq_len}")

    # KNN graph
    print("Building KNN graph ...")
    nn_model = NearestNeighbors(n_neighbors=seq_len).fit(coords)
    _, neighbors = nn_model.kneighbors(coords)

    # Sample bead indices
    n_beads = adata.n_obs
    indices = (
        np.random.choice(n_beads, sample_beads, replace=False)
        if sample_beads and sample_beads < n_beads else np.arange(n_beads)
    )
    print(f"  Sampling {len(indices):,} beads ...")

    # Accumulate moment statistics without storing full (N, S, G) arrays (memory-efficient)
    # We compute global stats by running a Welford online pass
    moment_names = ["nascent_mean", "mature_mean", "nascent_var", "mature_var", "cov_nm"]
    accum_mins = {k: np.inf for k in moment_names}
    accum_maxs = {k: -np.inf for k in moment_names}
    # For mean/std we collect flat samples (capped at 1M values each to stay in memory)
    MAX_SAMPLES = 1_000_000
    samples = {k: [] for k in moment_names}
    total_values = {k: 0 for k in moment_names}

    for i, idx in enumerate(indices):
        seq = neighbors[idx]           # (S,)
        N_seq = N_raw[seq]             # (S, G)
        M_seq = M_raw[seq]             # (S, G)

        # Moments — exactly as in SpatialTranscriptomicsDataset.__getitem__
        n_mean = N_seq                                                         # (S, G)
        m_mean = M_seq                                                         # (S, G)
        n_var_val = np.var(N_seq, axis=0, keepdims=True).repeat(seq_len, axis=0)  # (S, G)
        m_var_val = np.var(M_seq, axis=0, keepdims=True).repeat(seq_len, axis=0)  # (S, G)
        cov = np.zeros_like(n_mean)                                            # (S, G) = 0

        moment_arrays = {
            "nascent_mean": n_mean,
            "mature_mean": m_mean,
            "nascent_var": n_var_val,
            "mature_var": m_var_val,
            "cov_nm": cov,
        }

        for k, arr in moment_arrays.items():
            flat = arr.ravel()
            accum_mins[k] = min(accum_mins[k], float(np.nanmin(flat)))
            accum_maxs[k] = max(accum_maxs[k], float(np.nanmax(flat)))
            total_values[k] += flat.size
            if len(samples[k]) < MAX_SAMPLES:
                take = min(MAX_SAMPLES - len(samples[k]), len(flat))
                samples[k].extend(flat[:take].tolist())

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1:,}/{len(indices):,} beads ...")

    return accum_mins, accum_maxs, samples, total_values, actual_genes


def print_summary(accum_mins, accum_maxs, samples):
    """Print formatted stats table."""
    moment_names = ["nascent_mean", "mature_mean", "nascent_var", "mature_var", "cov_nm"]

    header = f"{'Moment':<18} {'Min':>12} {'Max':>14} {'Mean':>12} {'Std':>12} {'NonZero%':>10}"
    print("\n" + "=" * 80)
    print("  EMPIRICAL MOMENT SCALE DIAGNOSTICS")
    print("=" * 80)
    print(header)
    print("-" * 80)

    for k in moment_names:
        arr = np.array(samples[k], dtype=np.float64)
        nonzero_frac = 100.0 * np.mean(arr != 0)
        print(
            f"{k:<18} "
            f"{accum_mins[k]:>12.3e} "
            f"{accum_maxs[k]:>14.3e} "
            f"{np.nanmean(arr):>12.3e} "
            f"{np.nanstd(arr):>12.3e} "
            f"{nonzero_frac:>9.1f}%"
        )
    print("=" * 80)

    # Compute implied L_data scale: MSE of each moment if predictions are near-zero
    print("\n  IMPLIED MSE SCALE (if model predicts ~0 initially):")
    print("-" * 80)
    for k in moment_names:
        arr = np.array(samples[k], dtype=np.float64)
        implied_mse = float(np.nanmean(arr ** 2))
        print(f"  {k:<18}: MSE ~ {implied_mse:.3e}")
    print("=" * 80)


def plot_histograms(samples, output_path):
    """Multi-panel histogram of all 5 moment distributions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    moment_names = ["nascent_mean", "mature_mean", "nascent_var", "mature_var", "cov_nm"]
    labels = {
        "nascent_mean": "Nascent RNA Mean (unspliced counts)",
        "mature_mean": "Mature RNA Mean (spliced counts)",
        "nascent_var": "Nascent RNA Variance",
        "mature_var": "Mature RNA Variance",
        "cov_nm": "Nascent-Mature Covariance",
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()

    for i, k in enumerate(moment_names):
        arr = np.array(samples[k], dtype=np.float64)
        # Clip extreme outliers for visualization
        p1, p99 = np.percentile(arr, [1, 99])
        arr_clip = arr[(arr >= p1) & (arr <= p99)]

        ax = axes[i]
        ax.hist(arr_clip, bins=80, color="#4C72B0", edgecolor="none", alpha=0.85)
        ax.set_title(labels[k], fontsize=11, fontweight="bold")
        ax.set_xlabel("Value", fontsize=9)
        ax.set_ylabel("Frequency", fontsize=9)
        ax.axvline(float(np.nanmean(arr)), color="red", linewidth=1.5,
                   linestyle="--", label=f"mean={np.nanmean(arr):.2e}")
        ax.legend(fontsize=8)

        # Annotate scale in corner
        ax.text(0.97, 0.95, f"std={np.nanstd(arr):.2e}\nmax={arr.max():.2e}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color="darkred",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.8))

    # Final panel: scale comparison bar chart
    ax_bar = axes[5]
    implied_mse = {k: float(np.nanmean(np.array(samples[k], dtype=np.float64) ** 2))
                   for k in moment_names}
    names_short = ["n_mean", "m_mean", "n_var", "m_var", "cov"]
    vals = [implied_mse[k] for k in moment_names]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    bars = ax_bar.bar(names_short, vals, color=colors, edgecolor="white")
    ax_bar.set_yscale("log")
    ax_bar.set_title("Implied MSE Scale per Moment Type", fontsize=11, fontweight="bold")
    ax_bar.set_ylabel("E[moment²]  (log scale)", fontsize=9)
    ax_bar.set_xlabel("Moment", fontsize=9)
    for bar, val in zip(bars, vals):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, val * 1.3,
                    f"{val:.1e}", ha="center", va="bottom", fontsize=8)

    plt.suptitle("Empirical Moment Scale Diagnostics\n"
                 "(Red dashed = mean; scale disparity drives 10¹⁰ data loss)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved histogram to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Profile empirical moment scales")
    parser.add_argument("--h5ad", default="data/processed/spatial_adata.h5ad",
                        help="Training h5ad (with spliced/unspliced layers)")
    parser.add_argument("--num_genes", type=int, default=2000)
    parser.add_argument("--seq_len", type=int, default=50,
                        help="KNN sequence length (same as training dt=0.02 → 50 steps)")
    parser.add_argument("--sample_beads", type=int, default=2000,
                        help="Number of beads to sample (for speed; None = all)")
    parser.add_argument("--output", default="logs/moment_distributions.png")
    args = parser.parse_args()

    mins, maxs, samples, totals, n_genes = compute_empirical_moments(
        args.h5ad, args.num_genes, args.seq_len, args.sample_beads
    )
    print_summary(mins, maxs, samples)
    plot_histograms(samples, args.output)


if __name__ == "__main__":
    main()
