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
    parser.add_argument("--outdir", default="logs/critical_window")
    parser.add_argument("--target_tf", default="Foxa2", help="TF to transiently knockout")
    parser.add_argument("--target_fate", type=int, default=1, help="The terminal fate dependent on this TF")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cpu")

    # Load Genes
    gene_names = pd.read_csv(args.genes, header=None)[0].tolist()
    gene_names_upper = [g.upper() for g in gene_names]
    target_tf_upper = args.target_tf.upper()
    
    if target_tf_upper not in gene_names_upper:
        print(f"Error: {args.target_tf} not found in expressed genes.")
        sys.exit(1)
    target_idx = gene_names_upper.index(target_tf_upper)

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    num_tfs, num_genes = config["num_tfs"], config["num_target_genes"]
    num_fates = config["num_terminal_fates"]
    seq_len = int(1.0 / config.get("dt", 0.02))
    
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

    # We use a larger batch size since this is a fast CPU forward pass
    dataloader = get_dataloader(
        h5ad_path=args.h5ad, batch_size=2048, seq_len=seq_len,
        num_genes=num_genes, num_fates=num_fates
    )

    print("Running Wild Type (WT) baseline inference...")
    wt_fate_counts = np.zeros(num_fates)
    with torch.no_grad():
        for batch in dataloader:
            u_seq = batch["u_seq"].to(device)
            result_wt = model(u_seq, collocation_t=None)
            # Use model logits, completely ignoring raw Palantir dummy targets
            dom_fate = result_wt["fate_logits"].mean(dim=1).argmax(dim=-1).cpu().numpy()
            for f in range(num_fates):
                wt_fate_counts[f] += np.sum(dom_fate == f)
                
    baseline_survival = wt_fate_counts[args.target_fate]
    print(f"Baseline WT cells assigned to Fate {args.target_fate}: {int(baseline_survival)}")

    survival_trajectory = []
    
    print(f"Sweeping Knockout Trigger across {seq_len} pseudotime steps...")
    for step in range(seq_len):
        # Define a fresh hook for the current step
        def sweep_hook(module, input, output, current_step=step):
            output_clone = output.clone()
            output_clone[:, current_step:, target_idx] = 0.0 
            return output_clone
            
        hook_handle = model.rnn.register_forward_hook(sweep_hook)
        
        ko_target_count = 0
        with torch.no_grad():
            for batch in dataloader:
                u_seq = batch["u_seq"].to(device)
                result_ko = model(u_seq, collocation_t=None)
                dom_fate = result_ko["fate_logits"].mean(dim=1).argmax(dim=-1).cpu().numpy()
                ko_target_count += np.sum(dom_fate == args.target_fate)
                
        survival_trajectory.append(ko_target_count)
        hook_handle.remove()
        
        # Print progress every 10 steps to keep the terminal clean
        if step % 10 == 0 or step == seq_len - 1:
            print(f"  Trigger at t={step/(seq_len-1):.2f} -> {ko_target_count} cells survived")

    # Plotting the Critical Window Curve
    print("Generating Spatiotemporal Survival Curve...")
    time_axis = np.linspace(0, 1, seq_len)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_axis, survival_trajectory, color='#C44E52', linewidth=3, marker='o', markersize=4)
    # ax.axhline(y=baseline_survival, color='#4C72B0', linestyle='--', linewidth=2, label=f'WT Baseline ({int(baseline_survival)} cells)')
    
    # Calculate the bifurcation point (where survival hits 50% of WT)
    half_max = baseline_survival / 2.0
    # Find the first index where survival exceeds 50%
    bifurcation_idx = np.where(np.array(survival_trajectory) >= half_max)[0]
    if len(bifurcation_idx) > 0:
        bifurcation_t = time_axis[bifurcation_idx[0]]
        # ax.axvline(x=bifurcation_t, color='black', linestyle=':', linewidth=2, label=f'Point of No Return ($t \\approx {bifurcation_t:.2f}$)')

    ax.set_title(f"Critical Window Analysis: {args.target_tf} Dependency for Fate {args.target_fate}", fontweight='bold', fontsize=14)
    ax.set_xlabel("Pseudotime of Knockout Trigger", fontsize=12)
    ax.set_ylabel(f"Surviving Population Size (Fate {args.target_fate})", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    out_path = os.path.join(args.outdir, f"Critical_Window_{args.target_tf}_Fate_{args.target_fate}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    
    print(f"Success! Critical Window plot saved to: {out_path}")

if __name__ == "__main__":
    main()
