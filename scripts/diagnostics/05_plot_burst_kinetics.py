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
    parser.add_argument("--outdir", default="logs/burst_kinetics")
    args = parser.parse_args()

    outdir_freq = os.path.join(args.outdir, "frequency_a")
    outdir_size = os.path.join(args.outdir, "size_b")
    os.makedirs(outdir_freq, exist_ok=True)
    os.makedirs(outdir_size, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    device = torch.device("cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    
    gene_names = pd.read_csv(args.genes, header=None)[0].tolist()
    num_tfs = config["num_tfs"]
    num_genes = config["num_target_genes"]
    
    # Model instantiation
    model = SpatialMechanisticModel(
        input_spatial_dim=config.get("input_spatial_dim", 2),
        num_tfs=num_tfs,
        num_target_genes=num_genes,
        num_terminal_fates=config["num_terminal_fates"],
        frozen_grn_matrix=torch.zeros((num_tfs, num_genes)),
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
        num_genes=num_genes,
        num_fates=config["num_terminal_fates"]
    )

    all_a = []
    all_b = []
    all_fates = []

    print("Running CPU inference...")
    with torch.no_grad():
        for batch in dataloader:
            u_seq = batch["u_seq"].to(device)
            result = model(u_seq, collocation_t=None)
            
            # Determine dominant fate based strictly on PINN logits
            dom_fate = result["fate_logits"].mean(dim=1).argmax(dim=-1) 
            
            all_a.append(result["burst_freq"])
            all_b.append(result["burst_size"])
            all_fates.append(dom_fate)

    all_a = torch.cat(all_a, dim=0).numpy() # (Total_Beads, Seq_Len, Num_Genes)
    all_b = torch.cat(all_b, dim=0).numpy() # (Total_Beads, Seq_Len, Num_Genes)
    all_fates = torch.cat(all_fates, dim=0).numpy() # (Total_Beads,)
    
    num_fates = config["num_terminal_fates"]
    
    mean_a = np.zeros((num_fates, seq_len, num_genes))
    mean_b = np.zeros((num_fates, seq_len, num_genes))
    
    for f in range(num_fates):
        mask = (all_fates == f)
        if mask.sum() > 0:
            mean_a[f] = all_a[mask].mean(axis=0)
            mean_b[f] = all_b[mask].mean(axis=0)

    print("Plotting Burst Kinetics grids...")
    genes_per_page = 16
    num_pages = int(np.ceil(num_genes / genes_per_page))
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860']
    time_axis = np.linspace(0, 1, seq_len)

    def plot_metric(mean_data, out_folder, title_prefix, ylabel):
        for page in range(num_pages):
            fig, axes = plt.subplots(4, 4, figsize=(16, 12))
            fig.suptitle(f"{title_prefix} Over Pseudotime (Page {page+1}/{num_pages})", fontsize=16)
            axes = axes.flatten()
            
            start_idx = page * genes_per_page
            end_idx = min(start_idx + genes_per_page, num_genes)
            
            for i, g_idx in enumerate(range(start_idx, end_idx)):
                ax = axes[i]
                g_name = gene_names[g_idx]
                
                for f in range(num_fates):
                    ax.plot(time_axis, mean_data[f, :, g_idx], label=f"Fate {f}", 
                            color=colors[f % len(colors)], linewidth=2.5)
                
                ax.set_title(g_name, fontweight='bold')
                if i >= 12: ax.set_xlabel("Pseudotime")
                if i % 4 == 0: ax.set_ylabel(ylabel)
                ax.grid(True, linestyle='--', alpha=0.5)
            
            for j in range(end_idx - start_idx, 16):
                axes[j].axis('off')
                
            axes[0].legend(loc="upper left", fontsize=10)
            plt.tight_layout(rect=[0, 0.03, 1, 0.96])
            
            out_path = os.path.join(out_folder, f"page_{page+1}.png")
            plt.savefig(out_path, dpi=150)
            plt.close()

    plot_metric(mean_a, outdir_freq, "Burst Frequency (a)", "Frequency Rate")
    plot_metric(mean_b, outdir_size, "Burst Size (b)", "Transcripts per Burst")
        
    print(f"Successfully generated grids in {args.outdir}")

if __name__ == "__main__":
    main()
