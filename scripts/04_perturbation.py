import argparse
import os
import sys
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models.hybrid_pinn import SpatialMechanisticModel
from src.data.dataloader import get_dataloader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results/pipeline_20260312_124239_job850060/checkpoints/checkpoint_epoch_500.pt")
    parser.add_argument("--h5ad", default="data/mouse_brain.h5ad")
    parser.add_argument("--genes", default="data/processed/expressed_genes.csv")
    parser.add_argument("--outdir", default="logs/perturbation")
    parser.add_argument("--target_tf", default="Msx3", help="TF to knockout")
    parser.add_argument("--target_fate", type=int, default=0, help="Fate index to plot spatial probabilities for")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cpu")

    # Load Genes and find Target Index
    gene_names = pd.read_csv(args.genes, header=None)[0].tolist()
    gene_names_upper = [g.upper() for g in gene_names]
    target_tf_upper = args.target_tf.upper()
    
    if target_tf_upper not in gene_names_upper:
        print(f"Error: {args.target_tf} not found in expressed genes.")
        sys.exit(1)
    target_idx = gene_names_upper.index(target_tf_upper)
    print(f"Target TF: {args.target_tf} (Index {target_idx})")

    # Load Checkpoint & Model
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    num_tfs, num_genes = config["num_tfs"], config["num_target_genes"]
    num_fates = config["num_terminal_fates"]
    
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

    # Load Data
    seq_len = int(1.0 / config.get("dt", 0.02))
    dataloader = get_dataloader(
        h5ad_path=args.h5ad, batch_size=2048, seq_len=seq_len,
        num_genes=num_genes, num_fates=num_fates
    )

    spatial_coords = []
    wt_probs = []
    ko_probs = []

    print("Running Wild Type (WT) baseline inference...")
    with torch.no_grad():
        for batch in dataloader:
            u_seq = batch["u_seq"].to(device)
            spatial_coords.append(u_seq[:, 0, :].cpu().numpy()) # Extract X, Y
            
            result_wt = model(u_seq, collocation_t=None)
            # Average logits over pseudotime, then softmax to get probability distribution
            probs = F.softmax(result_wt["fate_logits"].mean(dim=1), dim=-1)
            wt_probs.append(probs.cpu().numpy())

    # Register Knockout Hook
    print(f"Applying in silico knockout hook to {args.target_tf}...")
    def knockout_hook(module, input, output):
        output_clone = output.clone()
        output_clone[:, :, target_idx] = 0.0 # Clamp TF activity to strictly 0
        return output_clone
    
    hook_handle = model.rnn.register_forward_hook(knockout_hook)

    print("Running Knockout (KO) inference...")
    with torch.no_grad():
        for batch in dataloader:
            u_seq = batch["u_seq"].to(device)
            result_ko = model(u_seq, collocation_t=None)
            probs = F.softmax(result_ko["fate_logits"].mean(dim=1), dim=-1)
            ko_probs.append(probs.cpu().numpy())

    hook_handle.remove()

    # Aggregate
    coords = np.concatenate(spatial_coords, axis=0)
    wt_p = np.concatenate(wt_probs, axis=0)
    ko_p = np.concatenate(ko_probs, axis=0)

    wt_target_p = wt_p[:, args.target_fate]
    ko_target_p = ko_p[:, args.target_fate]
    delta_p = ko_target_p - wt_target_p

    # --- FIGURE 2: Spatial Causal Map ---
    print("Generating Figure 2 (Spatial Map)...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Spatial Causal Map: {args.target_tf} Knockout Effect on Fate {args.target_fate}", fontsize=16, fontweight='bold')
    
    s_size = 5
    # Panel A
    sc0 = axes[0].scatter(coords[:, 0], coords[:, 1], c=wt_target_p, cmap='viridis', s=s_size, vmin=0, vmax=1)
    axes[0].set_title("Wild Type (Baseline Probability)")
    axes[0].axis('equal'); axes[0].axis('off')
    plt.colorbar(sc0, ax=axes[0], fraction=0.046, pad=0.04)

    # Panel B
    sc1 = axes[1].scatter(coords[:, 0], coords[:, 1], c=ko_target_p, cmap='viridis', s=s_size, vmin=0, vmax=1)
    axes[1].set_title(f"{args.target_tf} Knockout Probability")
    axes[1].axis('equal'); axes[1].axis('off')
    plt.colorbar(sc1, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel C
    vmax_delta = max(abs(delta_p.min()), abs(delta_p.max()))
    sc2 = axes[2].scatter(coords[:, 0], coords[:, 1], c=delta_p, cmap='coolwarm', s=s_size, vmin=-vmax_delta, vmax=vmax_delta)
    axes[2].set_title("Δ Probability (KO - WT)")
    axes[2].axis('equal'); axes[2].axis('off')
    plt.colorbar(sc2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig2_path = os.path.join(args.outdir, f"Figure_2_{args.target_tf}_KO_Spatial.png")
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close()

    # --- SUPPLEMENTARY: Fate Redistribution Bar Chart ---
    print("Generating Supplementary Fate Redistribution Chart...")
    wt_dom_fate = wt_p.argmax(axis=-1)
    ko_dom_fate = ko_p.argmax(axis=-1)
    
    wt_counts = [np.sum(wt_dom_fate == f) for f in range(num_fates)]
    ko_counts = [np.sum(ko_dom_fate == f) for f in range(num_fates)]

    x = np.arange(num_fates)
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width/2, wt_counts, width, label='Wild Type', color='#4C72B0')
    ax.bar(x + width/2, ko_counts, width, label=f'{args.target_tf} Knockout', color='#C44E52')
    
    ax.set_ylabel('Number of Cells')
    ax.set_xlabel('Terminal Fate Assignment')
    ax.set_title(f'Global Fate Redistribution post-{args.target_tf} Knockout', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fate {f}" for f in range(num_fates)])
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    supp_path = os.path.join(args.outdir, f"Supp_{args.target_tf}_KO_Redistribution.png")
    plt.savefig(supp_path, dpi=300)
    plt.close()

    print(f"Success! Output saved to: {args.outdir}")

if __name__ == "__main__":
    main()
