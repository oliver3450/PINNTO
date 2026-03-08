import os
import scvelo as scv
import scanpy as sc
import pandas as pd

def main():
    print("Step 1: Loading Spacemake's native spliced/unspliced dataset...")
    # Using the exact path defined in your pipeline docs
    PROJ_DIR = "data/raw/openst_data/spacemake/projects/openst_demo"
    SPACEMAKE_H5AD_PATH = f"{PROJ_DIR}/processed_data/openst_demo_e13_mouse_head/h5ad/spatial.h5ad"
    OUT_PATH = "data/processed/spatial_adata.h5ad"

    sdata = sc.read_h5ad(SPACEMAKE_H5AD_PATH)

    print("Step 2: Verifying Spacemake Splicing Layers...")
    if 'spliced' not in sdata.layers or 'unspliced' not in sdata.layers:
        raise ValueError("Spacemake h5ad is missing spliced/unspliced layers! Check Spacemake logs.")

    print("Step 3: Filtering to 2,000 highly variable genes for 8-A100 GPU limits...")
    # scvelo will automatically detect and use the 'spliced' and 'unspliced' layers
    scv.pp.filter_and_normalize(sdata, min_shared_counts=20, n_top_genes=2000)
    scv.pp.moments(sdata, n_pcs=30, n_neighbors=30)

    print("Step 4: Computing empirical steady-state kinetics...")
    scv.tl.velocity(sdata, mode='deterministic')

    print("Step 5: Saving finalized AnnData and Gene List...")
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    sdata.write(OUT_PATH)

    # Export the gene list so the PINN's GRN matrix perfectly aligns with the data
    expressed_genes = sdata.var_names.tolist()
    pd.Series(expressed_genes).to_csv("data/processed/expressed_genes.csv", index=False, header=False)
    print(f"Successfully prepared {OUT_PATH}")

if __name__ == "__main__":
    main()
