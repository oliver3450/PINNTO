import argparse
import os
import scanpy as sc
import scvelo as scv

def main():
    parser = argparse.ArgumentParser(description="Merge Spacemake coordinates with Velocyto matrices.")
    parser.add_argument("--spatial_h5ad", required=True, help="Path to Spacemake .h5ad")
    parser.add_argument("--velocyto_loom", required=True, help="Path to Velocyto .loom")
    parser.add_argument("--output", default="data/processed/spatial_adata.h5ad")
    args = parser.parse_args()

    print(f"Loading Spatial data: {args.spatial_h5ad}")
    adata_spatial = sc.read_h5ad(args.spatial_h5ad)
    
    print(f"Loading Velocyto data: {args.velocyto_loom}")
    adata_loom = scv.read(args.velocyto_loom, cache=True)

    print("\n--- Barcode Diagnostics ---")
    print(f"Spatial Barcode Example: {adata_spatial.obs_names[0]}")
    print(f"Loom Barcode Example:    {adata_loom.obs_names[0]}")
    
    # Normalizing Velocyto barcodes (adjust this if the print statements show a different mismatch)
    # Example: converting 'sample_id:ACGTCGATx' to 'ACGTCGAT'
    clean_barcodes = []
    for bc in adata_loom.obs_names:
        clean_bc = bc.split(':')[-1]  # Strip prefix
        clean_bc = clean_bc.replace('x', '').replace('-1', '') # Strip suffixes
        clean_barcodes.append(clean_bc)
    
    adata_loom.obs_names = clean_barcodes
    print(f"Cleaned Loom Example:    {adata_loom.obs_names[0]}\n")

    print("Merging datasets...")
    adata_merged = scv.utils.merge(adata_spatial, adata_loom)
    
    if adata_merged.n_obs == 0:
        raise ValueError("CRITICAL FAILURE: 0 cells remained after merge. Barcode strings do not match.")

    print(f"Success. Final shape: {adata_merged.shape}")
    print(f"Layers available: {list(adata_merged.layers.keys())}")
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    adata_merged.write_h5ad(args.output)
    print(f"Saved ready-to-train matrix to {args.output}")

if __name__ == "__main__":
    main()
