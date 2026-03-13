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
    parser.add_argument("--outdir", default="logs/tf_activity")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    device = torch.device("cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    
    gene_names = pd.read_csv(args.genes, header=None)[0].tolist()
    num_tfs = config["num_tfs"]
    
    # Model instantiation
    model = SpatialMechanisticModel(
        input_spatial_dim=config.get("input_spatial_dim", 2),
        num_tfs=num_tfs,
        num_target_genes=config["num_target_genes"],
        num_terminal_fates=config["num_terminal_fates"],
        frozen_grn_matrix=torch.zeros((num_tfs, config["num_target_genes"])),
        dt=config.get("dt", 0.02)
    )
    
    clean_state_dict = {k.replace("module.", "").replace("model.", ""): v 
                        for k, v in checkpoint["model_state_dict"].items()}
    model.load_state_dict(clean_state_dict, strict=False)
    model.eval()

    print(f"Loading data from {args.h5ad}...")
    seq_len = int(1.0 / config.get("dt", 0.02))
    dataloader = get_dataloader(
        h5ad_path=args.h5ad,
        batch_size=1024,
        seq_len=seq_len,
        num_genes=config["num_target_genes"],
        num_fates=config["num_terminal_fates"]
    )

    all_h_seq = []
    all_fates = []

    print("Running CPU inference...")
    with torch.no_grad():
        for batch in dataloader:
            u_seq = batch["u_seq"].to(device)
            fate_targets = batch["fate_targets"].to(device)
            
            result = model(u_seq, collocation_t=None)
            h_seq = result["hidden_tfs"] 
            
            # Determine dominant fate based on the PINN's optimized predictions
            fate_logits = result["fate_logits"]  # (Batch, Seq_Len, Num_Fates)
            # Average the logits over the sequence length to get a stable whole-trajectory prediction
            dom_fate = fate_logits.mean(dim=1).argmax(dim=-1)
            
            all_h_seq.append(h_seq)
            all_fates.append(dom_fate)

    all_h_seq = torch.cat(all_h_seq, dim=0).numpy() # (Total_Beads, Seq_Len, Num_TFs)
    all_fates = torch.cat(all_fates, dim=0).numpy() # (Total_Beads,)
    
    num_fates = config["num_terminal_fates"]
    
    # Calculate mean trajectory per fate
    mean_h_seq = np.zeros((num_fates, seq_len, num_tfs))
    for f in range(num_fates):
        mask = (all_fates == f)
        if mask.sum() > 0:
            mean_h_seq[f] = all_h_seq[mask].mean(axis=0)

    print("Plotting TF Activity grids...")
    tfs_per_page = 16
    num_pages = int(np.ceil(num_tfs / tfs_per_page))
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860']
    time_axis = np.linspace(0, 1, seq_len)

    for page in range(num_pages):
        fig, axes = plt.subplots(4, 4, figsize=(16, 12))
        fig.suptitle(f"TF Activity Over Pseudotime (Page {page+1}/{num_pages})", fontsize=16)
        axes = axes.flatten()
        
        start_idx = page * tfs_per_page
        end_idx = min(start_idx + tfs_per_page, num_tfs)
        
        for i, tf_idx in enumerate(range(start_idx, end_idx)):
            ax = axes[i]
            tf_name = gene_names[tf_idx]
            
            for f in range(num_fates):
                ax.plot(time_axis, mean_h_seq[f, :, tf_idx], label=f"Fate {f}", 
                        color=colors[f % len(colors)], linewidth=2.5)
            
            ax.set_title(tf_name, fontweight='bold')
            if i >= 12: ax.set_xlabel("Pseudotime")
            if i % 4 == 0: ax.set_ylabel("Activity Rate")
            ax.grid(True, linestyle='--', alpha=0.5)
        
        # Hide any unused subplots on the final page
        for j in range(end_idx - start_idx, 16):
            axes[j].axis('off')
            
        axes[0].legend(loc="upper left", fontsize=10)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        out_path = os.path.join(args.outdir, f"tf_activity_page_{page+1}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        
    print(f"Successfully generated {num_pages} grids in {args.outdir}")

if __name__ == "__main__":
    main()
