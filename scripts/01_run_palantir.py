"""
01_run_palantir.py — Compute pseudotime and fate probabilities with Palantir.

Reads a raw or lightly-processed .h5ad (must have adata.obsm['spatial']),
runs the full Palantir pipeline (PCA -> diffusion maps -> pseudotime + fates),
and writes an augmented .h5ad with:
    adata.obs['palantir_pseudotime']          — continuous pseudotime per cell
    adata.obs['palantir_entropy']             — differentiation potential
    adata.obsm['palantir_fate_probs']         — (n_cells, n_fates) soft fate probs

Usage:
    python scripts/01_run_palantir.py --h5ad data/mouse_brain.h5ad \
                                      --output data/processed/palantir_adata.h5ad \
                                      --n_components 30 \
                                      --n_diffusion 10
"""

import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import palantir


def detect_root_cell(adata, dm_res):
    """
    Heuristic root selection: pick the cell at the extreme of the first
    diffusion component (DC1).  For E13 mouse head, this typically
    corresponds to the least-differentiated progenitor population.
    """
    dc1 = dm_res.iloc[:, 0]
    root_cell = dc1.idxmin()
    print(f"Auto-detected root cell: {root_cell} (DC1 = {dc1[root_cell]:.4f})")
    return root_cell


def run_palantir_pipeline(args):
    print(f"Reading {args.h5ad} ...")
    adata = sc.read_h5ad(args.h5ad)
    print(f"  {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    # --- Preprocessing ---
    from scipy.sparse import issparse

    # Open-ST h5ad sometimes stores counts only in layers['spliced'], leaving X empty.
    # Detect this and fall back to a real count layer so normalization doesn't produce NaN.
    x_total = float(adata.X.sum())
    if x_total == 0 or np.isnan(x_total):
        for layer_name in ("spliced", "counts", "raw"):
            if layer_name in adata.layers:
                print(f"  X is empty — using layer '{layer_name}' for Palantir preprocessing")
                adata.X = adata.layers[layer_name].copy()
                break

    # Filter empty cells/genes before normalizing.
    # Cells with 0 total counts → NaN after normalize_total → NaN gene means → HVG crash.
    sc.pp.filter_cells(adata, min_genes=1)
    sc.pp.filter_genes(adata, min_cells=3)
    print(f"  After filtering: {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Sanitize: individual NaN/inf entries can survive from Spacemake/Velocyto processing.
    # Replace in-place so downstream HVG and PCA never see non-finite values.
    if issparse(adata.X):
        adata.X.data = np.nan_to_num(adata.X.data, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        adata.X = np.nan_to_num(np.asarray(adata.X), nan=0.0, posinf=0.0, neginf=0.0)

    # HVG selection with a variance-based fallback in case binning still fails
    # (can happen with very sparse or unusual expression distributions).
    n_hvg = min(2000, adata.n_vars)
    try:
        sc.pp.highly_variable_genes(
            adata, n_top_genes=n_hvg, flavor="seurat",
            min_mean=0.0125, max_mean=10, min_disp=0.25,
        )
        if adata.var["highly_variable"].sum() == 0:
            raise ValueError("No HVGs found with current thresholds")
        print(f"  {adata.var['highly_variable'].sum()} HVGs selected for Palantir")
    except Exception as e:
        print(f"  HVG failed ({type(e).__name__}: {e}), using top-{n_hvg} by variance")
        X_arr = adata.X.toarray() if issparse(adata.X) else np.asarray(adata.X)
        gene_var = np.nanvar(X_arr, axis=0)
        top_idx = np.argsort(gene_var)[::-1][:n_hvg]
        adata.var["highly_variable"] = False
        adata.var["highly_variable"].iloc[top_idx] = True
        print(f"  {adata.var['highly_variable'].sum()} genes selected by variance")

    # PCA on selected genes
    sc.tl.pca(adata, n_comps=args.n_components, use_highly_variable=True)
    print(f"  PCA: {args.n_components} components")

    # --- Diffusion Maps ---
    print("Computing diffusion maps ...")
    pca_df = pd.DataFrame(
        adata.obsm["X_pca"][:, :args.n_components],
        index=adata.obs_names,
    )
    dm_res = palantir.utils.run_diffusion_maps(pca_df, n_components=args.n_diffusion)

    # Newer Palantir returns a dict {'EigenVectors': df, 'EigenValues': series, ...}
    # rather than a DataFrame directly.
    if isinstance(dm_res, dict):
        dm_eigvec = dm_res["EigenVectors"]
    else:
        dm_eigvec = dm_res  # older API: dm_res is already a DataFrame
    print(f"  Diffusion maps: {dm_eigvec.shape[1]} components")

    # Multi-scale space for Palantir (accepts either dict or DataFrame depending on version)
    ms_data = palantir.utils.determine_multiscale_space(dm_res)
    if hasattr(ms_data, "shape"):
        print(f"  Multi-scale space: {ms_data.shape}")
    else:
        print(f"  Multi-scale space computed")

    # --- Root Cell Detection ---
    if args.root_cell:
        root_cell = args.root_cell
        print(f"Using user-specified root cell: {root_cell}")
    else:
        root_cell = detect_root_cell(adata, dm_eigvec)

    # --- Run Palantir ---
    print("Running Palantir (pseudotime + fate probabilities) ...")
    pr_res = palantir.core.run_palantir(
        ms_data,
        early_cell=root_cell,
        num_waypoints=min(1200, adata.n_obs // 5),
        use_early_cell_as_start=True,
    )

    n_fates = pr_res.branch_probs.shape[1]
    terminal_states = list(pr_res.branch_probs.columns)
    print(f"  Pseudotime range: [{pr_res.pseudotime.min():.3f}, {pr_res.pseudotime.max():.3f}]")
    print(f"  Terminal states ({n_fates}): {terminal_states}")

    # --- Store Results in AnnData ---
    # Pseudotime (normalized to [0, 1])
    pt = pr_res.pseudotime
    pt_norm = (pt - pt.min()) / (pt.max() - pt.min() + 1e-10)
    adata.obs["palantir_pseudotime"] = pt_norm.values

    # Differentiation potential (entropy)
    adata.obs["palantir_entropy"] = pr_res.entropy.values

    # Fate probabilities: (n_cells, n_fates) — aligned to adata.obs_names
    fate_probs = pr_res.branch_probs.loc[adata.obs_names].values
    adata.obsm["palantir_fate_probs"] = fate_probs

    # Store terminal state names for downstream reference
    adata.uns["palantir_terminal_states"] = terminal_states
    adata.uns["palantir_root_cell"] = root_cell

    # --- Save ---
    print(f"\nSaving augmented AnnData to {args.output} ...")
    adata.write_h5ad(args.output)
    print(f"  Done. {adata.n_obs:,} cells, {n_fates} terminal fates.")

    # Print summary statistics
    print("\n--- Palantir Summary ---")
    print(f"  Root cell:        {root_cell}")
    print(f"  Terminal states:  {terminal_states}")
    print(f"  Pseudotime mean:  {pt_norm.mean():.3f}")
    print(f"  Entropy mean:     {pr_res.entropy.mean():.3f}")
    print(f"  Fate prob shape:  {fate_probs.shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute pseudotime and fate probabilities with Palantir"
    )
    parser.add_argument("--h5ad", required=True,
                        help="Input .h5ad file (raw or lightly processed)")
    parser.add_argument("--output", default="data/processed/palantir_adata.h5ad",
                        help="Output .h5ad with Palantir results embedded")
    parser.add_argument("--n_components", type=int, default=30,
                        help="Number of PCA components")
    parser.add_argument("--n_diffusion", type=int, default=10,
                        help="Number of diffusion map components")
    parser.add_argument("--root_cell", default=None,
                        help="Barcode of root cell (auto-detected if omitted)")
    args = parser.parse_args()

    run_palantir_pipeline(args)


if __name__ == "__main__":
    main()
