"""
generate_fake_data.py — Create a synthetic h5ad for end-to-end pipeline testing.

Produces a small dataset with:
  - Gene names that overlap with TFTGDB (so GRN construction works)
  - Spliced/unspliced layers
  - 2D spatial coordinates
  - Optional fake Palantir pseudotime + fate probabilities

Usage:
    python scripts/generate_fake_data.py [--with-palantir] [--n-beads 400] [--n-genes 200]
"""

import argparse
import os
import numpy as np
import pandas as pd
import anndata as ad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-beads", type=int, default=400)
    parser.add_argument("--n-genes", type=int, default=200)
    parser.add_argument("--n-fates", type=int, default=2)
    parser.add_argument("--with-palantir", action="store_true",
                        help="Include fake Palantir pseudotime + fate probs")
    parser.add_argument("--output", default="data/processed/fake_spatial.h5ad")
    args = parser.parse_args()

    n_beads = args.n_beads
    n_genes = args.n_genes

    # Use real gene names from TFTGDB so GRN matrix construction works
    tftg = pd.read_csv("src/data/frozen_databases/TFTGDB.csv")
    all_genes = sorted(set(tftg["source"].tolist() + tftg["target"].tolist()))
    if len(all_genes) < n_genes:
        # Pad with synthetic names if TFTGDB doesn't have enough
        extra = [f"SynGene_{i}" for i in range(n_genes - len(all_genes))]
        gene_names = all_genes + extra
    else:
        gene_names = all_genes[:n_genes]

    # Synthetic expression data
    spatial_coords = np.random.rand(n_beads, 2) * 100
    spliced = np.random.poisson(lam=5.0, size=(n_beads, n_genes)).astype(np.float32)
    unspliced = np.random.poisson(lam=2.0, size=(n_beads, n_genes)).astype(np.float32)

    adata = ad.AnnData(
        X=spliced,
        var=pd.DataFrame(index=gene_names),
        obs=pd.DataFrame(index=[f"bead_{i}" for i in range(n_beads)]),
    )
    adata.obsm["spatial"] = spatial_coords
    adata.layers["spliced"] = spliced
    adata.layers["unspliced"] = unspliced

    if args.with_palantir:
        # Fake pseudotime: gradient across spatial x-coordinate
        pt = (spatial_coords[:, 0] - spatial_coords[:, 0].min())
        pt = pt / (pt.max() + 1e-10)
        adata.obs["palantir_pseudotime"] = pt
        adata.obs["palantir_entropy"] = np.random.rand(n_beads) * 0.5

        # Fake fate probabilities: soft partition based on pseudotime
        fate_probs = np.zeros((n_beads, args.n_fates))
        for i in range(args.n_fates):
            fate_probs[:, i] = np.exp(-((pt - i / (args.n_fates - 1)) ** 2) / 0.2)
        fate_probs = fate_probs / fate_probs.sum(axis=1, keepdims=True)
        adata.obsm["palantir_fate_probs"] = fate_probs
        adata.uns["palantir_terminal_states"] = [f"Fate_{i}" for i in range(args.n_fates)]
        print(f"Added Palantir: pseudotime + {args.n_fates} fates")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    adata.write_h5ad(args.output)
    print(f"Wrote {args.output}: {n_beads} beads x {n_genes} genes")
    print(f"  Gene names from TFTGDB: {gene_names[:5]} ...")
    print(f"  Layers: {list(adata.layers.keys())}")


if __name__ == "__main__":
    main()
