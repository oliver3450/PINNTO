import os
import scvelo as scv
import scanpy as sc
import anndata as ad
import pandas as pd

def main():
    print("Step 1: Loading raw datasets...")
    PROJ_DIR = "data/raw/openst_data/spacemake/projects/openst_demo"
    LOOM_PATH = f"{PROJ_DIR}/velocyto_output/final_converted.loom"
    SPACEMAKE_H5AD_PATH = f"{PROJ_DIR}/processed_data/openst_demo_e13_mouse_head/h5ad/spatial.h5ad"
    OUT_PATH = "data/processed/spatial_adata.h5ad"

    if not os.path.exists(LOOM_PATH): raise FileNotFoundError(f"Missing: {LOOM_PATH}")
    if not os.path.exists(SPACEMAKE_H5AD_PATH): raise FileNotFoundError(f"Missing: {SPACEMAKE_H5AD_PATH}")

    vdata = scv.read_loom(LOOM_PATH)
    sdata = sc.read_h5ad(SPACEMAKE_H5AD_PATH)

    print("Step 2: Cleaning barcodes and performing inner join...")
    vdata.obs.index = vdata.obs.index.str.replace('final_converted:', '').str.replace('x', '')
    common_barcodes = vdata.obs.index.intersection(sdata.obs.index)
    vdata, sdata = vdata[common_barcodes].copy(), sdata[common_barcodes].copy()

    print("Step 3: Transferring spatial geometry...")
    vdata.obsm['spatial'] = sdata.obsm['spatial'].copy()

    print("Step 4: Filtering to 2,000 highly variable genes...")
    scv.pp.filter_and_normalize(vdata, min_shared_counts=20, n_top_genes=2000)
    scv.pp.moments(vdata, n_pcs=30, n_neighbors=30)

    print("Step 5: Computing empirical steady-state kinetics...")
    scv.tl.velocity(vdata, mode='deterministic')

    print("Step 6: Saving finalized AnnData and Gene List...")
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    vdata.write(OUT_PATH)
    
    expressed_genes = vdata.var_names.tolist()
    pd.Series(expressed_genes).to_csv("data/processed/expressed_genes.csv", index=False, header=False)
    print(f"Successfully prepared {OUT_PATH}")

if __name__ == "__main__":
    main()
