import scanpy as sc
import scvelo as scv

loom_path = "/gpfs/home/qukungroup/odorn/spatial_mechanistic_model/data/raw/openst_data/spacemake/projects/openst_demo/velocyto_output/final_converted_sorted_U7NIG.loom"

print("Loading massive loom file via scanpy. This will take a moment...")
# Scanpy's read_loom is the most robust parser for velocyto outputs
adata = sc.read_loom(loom_path)

print("\nMatrix successfully loaded!")
print(f"Total Beads (Observations): {adata.n_obs}")
print(f"Total Genes (Variables): {adata.n_vars}")

print("\nCalculating Spliced vs. Unspliced proportions:")
# Now we hand the AnnData object over to scvelo
scv.utils.show_proportions(adata)
