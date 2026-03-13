import scanpy as sc
import scvelo as scv

velo_path = "/gpfs/home/qukungroup/odorn/spatial_mechanistic_model/data/raw/openst_data/spacemake/projects/openst_demo/velocyto_output/final_converted_sorted_U7NIG.loom"

print("1/2: Loading Velocyto loom (Expect 15-30 mins for GPFS I/O)...", flush=True)
adata_velo = sc.read_loom(velo_path)

print(f"Pre-filter Matrix: {adata_velo.n_obs} beads, {adata_velo.n_vars} genes", flush=True)

print("2/2: Applying relaxed velocity filter (min_shared_counts=5)...", flush=True)
scv.pp.filter_and_normalize(adata_velo, min_shared_counts=5, n_top_genes=2000)

print(f"\n=========================================", flush=True)
print(f"THRESHOLD 5 RESULTS:", flush=True)
print(f"Beads remaining: {adata_velo.n_obs}", flush=True)
print(f"Velocity Genes remaining: {adata_velo.n_vars}", flush=True)
print(f"=========================================\n", flush=True)

print("--- SURVIVING GENES ---", flush=True)
genes = adata_velo.var_names.tolist()
print(", ".join(genes), flush=True)
print("-----------------------", flush=True)
