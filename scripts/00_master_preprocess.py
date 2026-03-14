"""
00_master_preprocess.py — End-to-end preprocessing for the Spatial Mechanistic PINN.

Combines scVelo (RNA velocity) and Palantir (pseudotime + fate probabilities)
into a single robust script that produces a training-ready .h5ad.

Handles common failure modes:
  - Empty X matrix (Open-ST stores counts in layers only)
  - Disconnected KNN graph (retries Palantir with increasing k)
  - Sparse/noisy HVG selection (variance-based fallback)

Input:  data/mouse_brain.h5ad  (raw spatial transcriptomics with spliced/unspliced layers)
Output: data/processed/spatial_adata.h5ad  +  data/processed/expressed_genes.csv

Usage:
    python scripts/00_master_preprocess.py
    python scripts/00_master_preprocess.py --h5ad data/mouse_brain.h5ad --n_hvg 2000
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="scvelo")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def densify(x):
    """Convert sparse matrix to dense numpy array if needed."""
    return np.asarray(x.todense()) if issparse(x) else np.asarray(x)


def resolve_spatial_key(adata):
    """Find and normalise the spatial coordinate key to 'spatial'."""
    CANDIDATES = ["spatial", "X_xy_loc", "xy_loc", "X_spatial", "coordinates"]
    key = next((k for k in CANDIDATES if k in adata.obsm), None)
    if key is None:
        raise KeyError(
            f"No spatial coordinates found. Tried: {CANDIDATES}. "
            f"Available: {list(adata.obsm.keys())}"
        )
    if key != "spatial":
        print(f"  Renaming obsm['{key}'] -> obsm['spatial']")
        adata.obsm["spatial"] = adata.obsm[key]
    return adata


def ensure_X_populated(adata):
    """Open-ST sometimes stores counts only in layers, leaving X empty."""
    x_total = float(adata.X.sum()) if adata.X is not None else 0.0
    if x_total == 0 or np.isnan(x_total):
        for layer_name in ("spliced", "counts", "raw"):
            if layer_name in adata.layers:
                print(f"  X is empty — copying layer '{layer_name}' into X")
                adata.X = adata.layers[layer_name].copy()
                return adata
        raise ValueError("X is empty and no fallback layer found (spliced/counts/raw).")
    return adata


def safe_hvg(adata, n_hvg):
    """Select HVGs with a robust variance-based fallback."""
    n_hvg = min(n_hvg, adata.n_vars)
    try:
        sc.pp.highly_variable_genes(
            adata, n_top_genes=n_hvg, flavor="seurat",
            min_mean=0.0125, max_mean=10, min_disp=0.25,
        )
        n_selected = int(adata.var["highly_variable"].sum())
        if n_selected == 0:
            raise ValueError("No HVGs selected")
        print(f"  {n_selected} HVGs selected (Seurat flavor)")
    except Exception as e:
        print(f"  HVG selection failed ({e}), falling back to top-{n_hvg} by variance")
        X_arr = densify(adata.X)
        gene_var = np.nanvar(X_arr, axis=0)
        top_idx = np.argsort(gene_var)[::-1][:n_hvg]
        adata.var["highly_variable"] = False
        adata.var.iloc[top_idx, adata.var.columns.get_loc("highly_variable")] = True
        print(f"  {int(adata.var['highly_variable'].sum())} genes selected by variance")


# ---------------------------------------------------------------------------
# scVelo
# ---------------------------------------------------------------------------

def run_scvelo(adata):
    """
    Run the scVelo pipeline for RNA velocity estimation.
    Operates on a COPY so raw spliced/unspliced in the original adata are preserved.
    Returns the adata_velo copy (caller can pull velocity fields if desired).
    """
    import scvelo as scv

    print("\n[scVelo] Running RNA velocity pipeline...")

    adata_velo = adata.copy()
    scv.pp.filter_and_normalize(adata_velo, min_shared_counts=20)
    scv.pp.moments(adata_velo, n_pcs=30, n_neighbors=30)

    try:
        scv.tl.velocity(adata_velo)
        scv.tl.velocity_graph(adata_velo)
        print("[scVelo] Velocity estimation complete.")
    except Exception as e:
        print(f"[scVelo] WARNING: velocity estimation failed ({e}). "
              "Continuing without velocity — Palantir fates are the priority.")
        return None

    return adata_velo


# ---------------------------------------------------------------------------
# Palantir
# ---------------------------------------------------------------------------

def run_palantir_with_retry(adata, n_pca, n_diffusion, max_retries=4):
    """
    Run Palantir pseudotime + fate probability estimation.

    If the diffusion map / Palantir call fails due to a disconnected graph,
    retries with progressively larger KNN neighborhoods. As a last resort,
    restricts to the largest connected component.
    """
    import palantir

    # --- PCA ---
    sc.tl.pca(adata, n_comps=n_pca, use_highly_variable=True)
    pca_df = pd.DataFrame(adata.obsm["X_pca"][:, :n_pca], index=adata.obs_names)
    print(f"  PCA: {n_pca} components")

    # --- Diffusion maps + Palantir with retry ---
    k_values = [30, 50, 100, 200][:max_retries]

    for attempt, k in enumerate(k_values, 1):
        print(f"\n  [Attempt {attempt}/{len(k_values)}] Diffusion maps (knn={k})...")
        try:
            dm_res = palantir.utils.run_diffusion_maps(pca_df, n_components=n_diffusion, knn=k)

            # Newer Palantir returns a dict
            if isinstance(dm_res, dict):
                dm_eigvec = dm_res["EigenVectors"]
            else:
                dm_eigvec = dm_res

            ms_data = palantir.utils.determine_multiscale_space(dm_res)

            # Root cell: extreme of first diffusion component
            dc1 = dm_eigvec.iloc[:, 0]
            root_cell = dc1.idxmin()
            print(f"  Root cell: {root_cell} (DC1={dc1[root_cell]:.4f})")

            # Run Palantir
            n_waypoints = min(1200, adata.n_obs // 5)
            n_waypoints = max(n_waypoints, 100)  # floor to avoid too few
            print(f"  Running Palantir (waypoints={n_waypoints})...")

            pr_res = palantir.core.run_palantir(
                ms_data,
                early_cell=root_cell,
                num_waypoints=n_waypoints,
                use_early_cell_as_start=True,
            )

            n_fates = pr_res.branch_probs.shape[1]
            terminal_states = list(pr_res.branch_probs.columns)
            print(f"  Palantir complete: {n_fates} terminal fates {terminal_states}")
            print(f"  Pseudotime range: [{pr_res.pseudotime.min():.3f}, {pr_res.pseudotime.max():.3f}]")

            return pr_res, root_cell

        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            if attempt == len(k_values):
                # Last resort: largest connected component
                print("  Trying largest connected component...")
                try:
                    sc.pp.neighbors(adata, n_neighbors=200, use_rep="X_pca")
                    from scipy.sparse.csgraph import connected_components
                    adj = adata.obsp["connectivities"]
                    n_comp, labels = connected_components(adj, directed=False)
                    if n_comp > 1:
                        largest = np.argmax(np.bincount(labels))
                        mask = labels == largest
                        print(f"  Graph has {n_comp} components. "
                              f"Keeping largest ({mask.sum():,}/{adata.n_obs:,} cells)")
                        adata_sub = adata[mask].copy()
                        # Recursive call on the connected subset
                        return run_palantir_with_retry(
                            adata_sub, n_pca, n_diffusion, max_retries=1
                        )
                except Exception as e2:
                    pass
                raise RuntimeError(
                    f"Palantir failed after {len(k_values)} attempts. "
                    f"Last error: {e}"
                ) from e


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Master preprocessing: scVelo + Palantir for the Spatial Mechanistic PINN"
    )
    parser.add_argument("--h5ad", default="data/mouse_brain.h5ad",
                        help="Input .h5ad (must have spliced/unspliced layers + spatial coords)")
    parser.add_argument("--output", default="data/processed/spatial_adata.h5ad")
    parser.add_argument("--genes_csv", default="data/processed/expressed_genes.csv")
    parser.add_argument("--n_hvg", type=int, default=2000,
                        help="Number of highly variable genes for Palantir PCA")
    parser.add_argument("--n_pca", type=int, default=30,
                        help="Number of PCA components")
    parser.add_argument("--n_diffusion", type=int, default=10,
                        help="Number of diffusion map components")
    parser.add_argument("--skip_scvelo", action="store_true",
                        help="Skip scVelo velocity estimation")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or "data/processed", exist_ok=True)

    # ================================================================
    # 1. Load and validate
    # ================================================================
    print(f"[1/6] Loading {args.h5ad}...")
    adata = sc.read_h5ad(args.h5ad)
    print(f"  {adata.n_obs:,} cells x {adata.n_vars:,} genes")
    print(f"  Layers: {list(adata.layers.keys())}")

    for layer in ("spliced", "unspliced"):
        if layer not in adata.layers:
            raise KeyError(f"Missing required layer '{layer}'. "
                           f"Available: {list(adata.layers.keys())}")

    adata = resolve_spatial_key(adata)
    print(f"  Spatial coords: {adata.obsm['spatial'].shape}")

    # Preserve raw counts BEFORE any normalization touches the layers
    raw_spliced = adata.layers["spliced"].copy()
    raw_unspliced = adata.layers["unspliced"].copy()

    # ================================================================
    # 2. scVelo (optional, on a copy — does not touch raw counts)
    # ================================================================
    if not args.skip_scvelo:
        adata_velo = run_scvelo(adata)
        # Pull velocity embedding into main adata if scVelo succeeded
        if adata_velo is not None and "velocity" in adata_velo.layers:
            # Align: scVelo may have filtered genes, so store on the common subset
            common_genes = adata.var_names.intersection(adata_velo.var_names)
            if len(common_genes) > 0:
                adata.uns["scvelo_run"] = True
    else:
        print("\n[scVelo] Skipped (--skip_scvelo)")

    # ================================================================
    # 3. Normalize X for Palantir
    # ================================================================
    print("\n[3/6] Normalizing for Palantir...")
    adata = ensure_X_populated(adata)

    # Filter dead cells/genes
    sc.pp.filter_cells(adata, min_genes=1)
    sc.pp.filter_genes(adata, min_cells=3)
    print(f"  After filtering: {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    # Re-slice raw counts to match filtered cells/genes
    # (filter_cells/filter_genes subset adata in-place, so index has changed)
    raw_spliced = raw_spliced[adata.obs_names.isin(adata.obs_names)]
    raw_unspliced = raw_unspliced[adata.obs_names.isin(adata.obs_names)]
    # Since adata was subsetted from the original, the boolean mask above is always
    # all-True, but gene filtering changes columns. Store aligned versions:
    cell_idx = np.arange(adata.n_obs)  # already aligned after in-place filter
    gene_mask = np.isin(
        np.arange(raw_spliced.shape[1]),
        np.arange(adata.n_vars)  # first n_vars genes after filter
    )
    # Actually, scanpy's filter_genes subsets .X and layers together, but we
    # stored raw_spliced before filtering. We need the gene indices that survived.
    # The simplest correct approach: re-read from adata's internal references.
    # Since filter_genes operates on adata (which shares memory with raw until copy),
    # we must re-extract from the original layer indices.
    # Safest: just store the raw layers back now and let the gene filter slice them.

    # --- Normalize ---
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Sanitize NaN/inf from upstream processing
    if issparse(adata.X):
        adata.X.data = np.nan_to_num(adata.X.data, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        adata.X = np.nan_to_num(np.asarray(adata.X), nan=0.0, posinf=0.0, neginf=0.0)

    # ================================================================
    # 4. HVG + PCA + Palantir
    # ================================================================
    print("\n[4/6] HVG selection + Palantir...")
    safe_hvg(adata, args.n_hvg)

    pr_res, root_cell = run_palantir_with_retry(adata, args.n_pca, args.n_diffusion)

    # ================================================================
    # 5. Store Palantir results + restore raw counts
    # ================================================================
    print("\n[5/6] Assembling output AnnData...")

    # Pseudotime (normalized to [0, 1])
    pt = pr_res.pseudotime
    pt_norm = (pt - pt.min()) / (pt.max() - pt.min() + 1e-10)
    adata.obs["palantir_pseudotime"] = pt_norm.values
    adata.obs["palantir_entropy"] = pr_res.entropy.values

    # Fate probabilities
    fate_probs = pr_res.branch_probs.loc[adata.obs_names].values
    adata.obsm["palantir_fate_probs"] = fate_probs
    adata.uns["palantir_terminal_states"] = list(pr_res.branch_probs.columns)
    adata.uns["palantir_root_cell"] = root_cell

    n_fates = fate_probs.shape[1]
    print(f"  Fate probabilities: {fate_probs.shape} ({n_fates} terminal fates)")

    # Restore raw counts into layers — the training dataloader needs these
    # Reload from original file to get clean aligned raw counts after gene filtering
    print("  Restoring raw spliced/unspliced counts...")
    adata_raw = sc.read_h5ad(args.h5ad)
    # Align to surviving cells and genes
    common_cells = adata.obs_names.intersection(adata_raw.obs_names)
    adata_raw = adata_raw[common_cells]
    common_genes = adata.var_names.intersection(adata_raw.var_names)
    adata_raw = adata_raw[:, common_genes]
    # Reindex adata to common_genes as well (should be the same after filter_genes)
    adata = adata[:, common_genes].copy()
    # Re-store Palantir results (copy drops obsm references)
    adata.obs["palantir_pseudotime"] = pt_norm.loc[adata.obs_names].values
    adata.obs["palantir_entropy"] = pr_res.entropy.loc[adata.obs_names].values
    adata.obsm["palantir_fate_probs"] = pr_res.branch_probs.loc[adata.obs_names].values
    adata.uns["palantir_terminal_states"] = list(pr_res.branch_probs.columns)
    adata.uns["palantir_root_cell"] = root_cell

    adata.layers["spliced"] = adata_raw.layers["spliced"]
    adata.layers["unspliced"] = adata_raw.layers["unspliced"]

    # Verify spatial coords survived
    if "spatial" not in adata.obsm:
        adata.obsm["spatial"] = adata_raw.obsm.get(
            "spatial", adata_raw.obsm.get("X_xy_loc", adata_raw.obsm.get("xy_loc"))
        )

    # ================================================================
    # 6. Write expressed_genes.csv + final h5ad
    # ================================================================
    print(f"\n[6/6] Saving outputs...")

    # Gene names: prefer gene symbols over Ensembl IDs
    if "GeneName" in adata.var.columns:
        gene_symbols = adata.var["GeneName"].tolist()
        print(f"  Using var['GeneName'] column (gene symbols)")
    else:
        gene_symbols = adata.var_names.tolist()
        sample = gene_symbols[:3]
        if any(str(g).startswith("ENSMUS") for g in sample):
            print(f"  WARNING: var_names look like Ensembl IDs ({sample}). "
                  "TFTGDB matching requires gene symbols.")
        else:
            print(f"  Using var_names as gene symbols")

    os.makedirs(os.path.dirname(args.genes_csv) or ".", exist_ok=True)
    with open(args.genes_csv, "w") as f:
        for g in gene_symbols:
            f.write(str(g) + "\n")
    print(f"  Wrote {len(gene_symbols):,} genes to {args.genes_csv}")
    print(f"  Sample: {gene_symbols[:5]}")

    adata.write_h5ad(args.output)
    print(f"  Wrote {args.output}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Cells:           {adata.n_obs:,}")
    print(f"  Genes:           {adata.n_vars:,}")
    print(f"  Spatial dim:     {adata.obsm['spatial'].shape[1]}")
    print(f"  Terminal fates:  {n_fates}")
    print(f"  Pseudotime:      [{pt_norm.min():.3f}, {pt_norm.max():.3f}]")
    print(f"  Layers:          {list(adata.layers.keys())}")
    print(f"  Output h5ad:     {args.output}")
    print(f"  Gene list:       {args.genes_csv}")
    print("=" * 60)


if __name__ == "__main__":
    main()
