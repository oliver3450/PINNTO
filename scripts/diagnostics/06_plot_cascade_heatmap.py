import argparse
import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.models.hybrid_pinn import SpatialMechanisticModel
from src.data.dataloader import get_dataloader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results/pipeline_20260312_124239_job850060/checkpoints/checkpoint_epoch_500.pt")
    parser.add_argument("--h5ad", default="data/mouse_brain.h5ad")
    parser.add_argument("--genes", default="data/processed/expressed_genes.csv")
    parser.add_argument("--outdir", default="logs/cascade_heatmaps")
    parser.add_argument("--activity_threshold", type=float, default=1e-3, 
                        help="Minimum maximum-activity required to avoid the silent gene clamp")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cpu")

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    
    gene_names = pd.read_csv(args.genes, header=None)[0].tolist()
    num_tfs = config["num_tfs"]
    num_genes = config["num_target_genes"]
    num_fates = config["num_terminal_fates"]
    
    # Instantiate Model
    model = SpatialMechanisticModel(
        input_spatial_dim=config.get("input_spatial_dim", 2),
        num_tfs=num_tfs,
        num_target_genes=num_genes,
        num_terminal_fates=num_fates,
        frozen_grn_matrix=torch.zeros((num_tfs, num_genes)),
        dt=config.get("dt", 0.02)
    )
    
    clean_state_dict = {k.replace("module.", "").replace("model.", ""): v 
                        for k, v in checkpoint["model_state_dict"].items()}
    model.load_state_dict(clean_state_dict, strict=False)
    model.eval()

    seq_len = int(1.0 / config.get("dt", 0.02))
    dataloader = get_dataloader(
        h5ad_path=args.h5ad,
        batch_size=2048,
        seq_len=seq_len,
        num_genes=num_genes,
        num_fates=num_fates
    )

    all_h_seq = []
    all_fates = []

    print("Running CPU inference to extract regulatory trajectories...")
    with torch.no_grad():
        for batch in dataloader:
            u_seq = batch["u_seq"].to(device)
            result = model(u_seq, collocation_t=None)
            
            # Cluster strictly by the PINN's learned logic
            dom_fate = result["fate_logits"].mean(dim=1).argmax(dim=-1) 
            all_h_seq.append(result["hidden_tfs"])
            all_fates.append(dom_fate)

    all_h_seq = torch.cat(all_h_seq, dim=0).numpy() # (N_Beads, Seq_Len, N_TFs)
    all_fates = torch.cat(all_fates, dim=0).numpy() # (N_Beads,)

    print("Generating Chronological Cascade Heatmaps...")
    for f in range(num_fates):
        mask = (all_fates == f)
        if mask.sum() == 0:
            continue
            
        # 1. Average trajectories for this specific fate
        mean_h = all_h_seq[mask].mean(axis=0) # (Seq_Len, N_TFs)
        
        # 2. Filter out biologically dead TFs using the mathematical clamp threshold
        max_act = mean_h.max(axis=0)
        active_idx = np.where(max_act > args.activity_threshold)[0]
        
        if len(active_idx) == 0:
            print(f"Fate {f}: No active TFs found above threshold.")
            continue
            
        active_h = mean_h[:, active_idx]
        active_names = [gene_names[i] for i in active_idx]
        
        # 3. Min-Max Normalize each TF's trajectory so colors scale cleanly from 0 to 1
        min_h = active_h.min(axis=0, keepdims=True)
        max_h = active_h.max(axis=0, keepdims=True)
        norm_h = (active_h - min_h) / (max_h - min_h + 1e-9)
        
        # 4. Calculate activation time (the exact bin where it crosses 50% of its max potential)
        activation_times = np.argmax(norm_h > 0.5, axis=0)
        
        # 5. Sort TFs by activation time to reveal the cascade
        sort_order = np.argsort(activation_times)
        sorted_h = norm_h[:, sort_order].T # Transpose for heatmap: (TFs, Seq)
        sorted_names = [active_names[i] for i in sort_order]
        
        # 6. Render Plot
        fig, ax = plt.subplots(figsize=(10, max(6, len(sorted_names) * 0.25)))
        im = ax.imshow(sorted_h, aspect='auto', cmap='magma', interpolation='nearest',
                       extent=[0, 1, len(sorted_names), 0])
        
        ax.set_yticks(np.arange(len(sorted_names)) + 0.5)
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel("Pseudotime")
        ax.set_title(f"Regulatory Cascade: Terminal Fate {f}\n({len(sorted_names)} Active Master Regulators)", fontweight='bold')
        
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Normalized Activity State")
        
        plt.tight_layout()
        out_path = os.path.join(args.outdir, f"Cascade_Fate_{f}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved {out_path} with {len(sorted_names)} genes.")

    print(f"Success! Generated {num_fates} heatmaps in {args.outdir}")

if __name__ == "__main__":
    main()
