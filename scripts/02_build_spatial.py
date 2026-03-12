"""
02_build_spatial.py — Build training-ready h5ad for the Spatial Mechanistic PINN.

Merges three data sources into a single training-ready AnnData:
  1. Velocyto outputs  — spliced/unspliced RNA layers
  2. Palantir outputs  — pseudotime + fate probabilities (from 01_run_palantir.py)
  3. Spatial coords    — bead positions from Open-ST / Spacemake

Also generates:
  - data/processed/expressed_genes.csv   — gene list for GRN matrix construction
  - data/processed/spatial_adata.h5ad    — final training-ready file

Usage:
    python scripts/02_build_spatial.py \
        --velocyto data/raw/velocyto.loom \
        --palantir data/processed/palantir_adata.h5ad \
        --output data/processed/spatial_adata.h5ad

    # Or if spliced/unspliced are already in the main h5ad:
    python scripts/02_build_spatial.py \
        --h5ad data/mouse_brain.h5ad \
        --palantir data/processed/palantir_adata.h5ad \
        --output data/processed/spatial_adata.h5ad
"""

import argparse
import os
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse


def load_velocyto_loom(loom_path: str) -> sc.AnnData:
    """Load Velocyto loom output into AnnData with spliced/unspliced layers."""
    import loompy
    ds = loompy.connect(loom_path, mode="r")
    adata = sc.AnnData(
        X=ds.layers["spliced"][:, :].T,
        obs=pd.DataFrame(index=ds.ca["CellID"]),
        var=pd.DataFrame(index=ds.ra["Gene"]),
    )
    adata.layers["spliced"] = adata.X.copy()
    adata.layers["unspliced"] = ds.layers["unspliced"][:, :].T
    ds.close()
    return adata


def densify(x):
    """Convert sparse matrix to dense numpy array if needed."""
    if issparse(x):
        return np.asarray(x.todense())
    return np.asarray(x)


def build_training_adata(args):
    # --- Load base expression data ---
    if args.h5ad:
        print(f"Loading base h5ad: {args.h5ad}")
        adata = sc.read_h5ad(args.h5ad)
    elif args.velocyto:
        print(f"Loading Velocyto loom: {args.velocyto}")
        adata = load_velocyto_loom(args.velocyto)
    else:
        raise ValueError("Must provide either --h5ad or --velocyto")

    print(f"  Base: {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    # --- Validate required layers ---
    required_layers = ["spliced", "unspliced"]
    for layer in required_layers:
        if layer not in adata.layers:
            raise KeyError(f"Missing required layer '{layer}'. "
                           f"Available: {list(adata.layers.keys())}")
    print(f"  Layers: {list(adata.layers.keys())}")

    # --- Resolve spatial coordinates ---
    # Open-ST stores coords as 'X_xy_loc' or 'xy_loc'; Scanpy convention is 'spatial'.
    # Normalise to 'spatial' so all downstream code uses one key.
    SPATIAL_KEY_CANDIDATES = ["spatial", "X_xy_loc", "xy_loc", "X_spatial", "coordinates"]
    spatial_key = next((k for k in SPATIAL_KEY_CANDIDATES if k in adata.obsm), None)
    if spatial_key is None:
        raise KeyError(
            f"No spatial coordinates found. Tried: {SPATIAL_KEY_CANDIDATES}. "
            f"Available obsm keys: {list(adata.obsm.keys())}"
        )
    if spatial_key != "spatial":
        print(f"  Renaming obsm['{spatial_key}'] -> obsm['spatial']")
        adata.obsm["spatial"] = adata.obsm[spatial_key]
    print(f"  Spatial coords: {adata.obsm['spatial'].shape} (from '{spatial_key}')")

    # --- Merge Palantir outputs ---
    if args.palantir:
        print(f"\nMerging Palantir results from: {args.palantir}")
        pdata = sc.read_h5ad(args.palantir)

        # Align cell barcodes
        common_cells = adata.obs_names.intersection(pdata.obs_names)
        if len(common_cells) == 0:
            # Try stripping suffixes (Velocyto adds -1, etc.)
            adata_clean = adata.obs_names.str.replace(r"-\d+$", "", regex=True)
            pdata_clean = pdata.obs_names.str.replace(r"-\d+$", "", regex=True)
            # If all barcodes match after stripping, just copy by position
            if adata.n_obs == pdata.n_obs:
                print("  Barcodes differ but cell counts match — merging by position")
                common_cells = adata.obs_names
                adata.obs["palantir_pseudotime"] = pdata.obs["palantir_pseudotime"].values
                adata.obs["palantir_entropy"] = pdata.obs["palantir_entropy"].values
                adata.obsm["palantir_fate_probs"] = pdata.obsm["palantir_fate_probs"]
            else:
                raise ValueError(
                    f"No common barcodes found. "
                    f"Base has {adata.n_obs} cells, Palantir has {pdata.n_obs} cells."
                )
        else:
            # Subset to common cells
            print(f"  Common cells: {len(common_cells):,} / {adata.n_obs:,}")
            adata = adata[common_cells].copy()
            pdata = pdata[common_cells]
            adata.obs["palantir_pseudotime"] = pdata.obs["palantir_pseudotime"].values
            adata.obs["palantir_entropy"] = pdata.obs["palantir_entropy"].values
            adata.obsm["palantir_fate_probs"] = pdata.obsm["palantir_fate_probs"]

        if "palantir_terminal_states" in pdata.uns:
            adata.uns["palantir_terminal_states"] = pdata.uns["palantir_terminal_states"]

        n_fates = adata.obsm["palantir_fate_probs"].shape[1]
        print(f"  Palantir merged: pseudotime + {n_fates} fate probabilities")
    else:
        print("\nNo Palantir file provided — fate targets will be placeholders.")

    # --- Filter genes: keep only those with nonzero expression ---
    spliced_sum = np.asarray(densify(adata.layers["spliced"]).sum(axis=0)).flatten()
    unspliced_sum = np.asarray(densify(adata.layers["unspliced"]).sum(axis=0)).flatten()
    expressed_mask = (spliced_sum > 0) | (unspliced_sum > 0)
    n_before = adata.n_vars
    adata = adata[:, expressed_mask].copy()
    print(f"\n  Gene filtering: {n_before:,} -> {adata.n_vars:,} expressed genes")

    # --- Write expressed_genes.csv ---
    # Must contain gene *symbols* (e.g. "Aatf") not Ensembl IDs (e.g. "ENSMUSG...").
    # TFTGDB uses gene symbols; Ensembl IDs produce 100% sparsity in the GRN matrix.
    os.makedirs("data/processed", exist_ok=True)
    genes_path = "data/processed/expressed_genes.csv"
    if "GeneName" in adata.var.columns:
        gene_symbols = adata.var["GeneName"].tolist()
        print("  Using var['GeneName'] column for expressed_genes.csv (gene symbols)")
    else:
        gene_symbols = adata.var_names.tolist()
        print("  WARNING: No 'GeneName' column found; using var_names (may be Ensembl IDs)")
    with open(genes_path, "w") as f:
        for g in gene_symbols:
            f.write(str(g) + "\n")
    print(f"  Wrote {len(gene_symbols):,} genes to {genes_path}  (sample: {gene_symbols[:3]})")

    # --- Save ---
    print(f"\nSaving training-ready h5ad to {args.output}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    adata.write_h5ad(args.output)

    # --- Summary ---
    print("\n--- Training Data Summary ---")
    print(f"  Cells:          {adata.n_obs:,}")
    print(f"  Genes:          {adata.n_vars:,}")
    print(f"  Spatial dim:    {adata.obsm['spatial'].shape[1]}")
    print(f"  Layers:         {list(adata.layers.keys())}")
    if "palantir_pseudotime" in adata.obs:
        pt = adata.obs["palantir_pseudotime"]
        print(f"  Pseudotime:     [{pt.min():.3f}, {pt.max():.3f}] (mean={pt.mean():.3f})")
        print(f"  Fate probs:     {adata.obsm['palantir_fate_probs'].shape}")
    print(f"  Output:         {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="Build training-ready h5ad for Spatial Mechanistic PINN"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--h5ad", help="Base h5ad with spliced/unspliced layers")
    group.add_argument("--velocyto", help="Velocyto .loom file")
    parser.add_argument("--palantir", default=None,
                        help="Palantir output h5ad (from 01_run_palantir.py)")
    parser.add_argument("--output", default="data/processed/spatial_adata.h5ad",
                        help="Output path for training-ready h5ad")
    args = parser.parse_args()

    build_training_adata(args)


if __name__ == "__main__":
    main()
