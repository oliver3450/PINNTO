import scanpy as sc
import pandas as pd
import gc

velo_path = "/gpfs/home/qukungroup/odorn/spatial_mechanistic_model/data/raw/openst_data/spacemake/projects/openst_demo/velocyto_output/final_converted_sorted_U7NIG.loom"
spatial_csv = "/gpfs/home/qukungroup/odorn/spatial_mechanistic_model/data/raw/openst_data/spacemake/projects/openst_demo/processed_data/openst_demo_e13_mouse_head/illumina/complete_data/dge/puck_collection.obs.csv"
out_path = "/gpfs/home/qukungroup/odorn/spatial_mechanistic_model/data/processed/e13_mouse_head_VELOCITY_SPATIAL_MASTER.h5ad"

print("1/5: Loading Velocyto loom...", flush=True)
adata = sc.read_loom(velo_path)

print("2/5: Formatting Barcodes to match Spacemake...", flush=True)
# Strip the 'final_converted_sorted:' prefix and 'x' suffix
adata.obs_names = [bc.split(':')[-1].replace('x', '') for bc in adata.obs_names]
adata.obs_names_make_unique()

print("3/5: Loading Spatial Coordinates...", flush=True)
spatial_obs = pd.read_csv(spatial_csv, index_col=0)

print("4/5: Merging Spatial Coordinates with Velocity Matrix...", flush=True)
# Perform an inner join to keep only beads that exist in BOTH files
adata.obs = adata.obs.join(spatial_obs, how='inner')

# Explicitly extract the X and Y columns into the spatial obsm array scanpy requires for plotting
if 'X' in adata.obs.columns and 'Y' in adata.obs.columns:
    adata.obsm['spatial'] = adata.obs[['X', 'Y']].to_numpy()
else:
    print("WARNING: 'X' and 'Y' columns not found in spatial CSV. Check column names.", flush=True)

# Free up memory before writing
del spatial_obs
gc.collect()

print(f"5/5: Saving final H5AD object ({adata.n_obs} beads) to disk...", flush=True)
adata.write(out_path)
print("SUCCESS! Master matrix generated.", flush=True)
