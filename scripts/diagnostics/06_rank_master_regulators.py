import argparse
import os
import sys
import torch
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.models.hybrid_pinn import SpatialMechanisticModel
from src.data.dataloader import get_dataloader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results/pipeline_20260312_124239_job850060/checkpoints/checkpoint_epoch_500.pt")
    parser.add_argument("--h5ad", default="data/processed/spatial_adata.h5ad")
    parser.add_argument("--genes", default="data/processed/expressed_genes.csv")
    parser.add_argument("--outdir", default="logs/critical_window")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cpu")

    gene_names = pd.read_csv(args.genes, header=None)[0].tolist()

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

    dataloader = get_dataloader(
        h5ad_path=args.h5ad, batch_size=4096, seq_len=seq_len,
        num_genes=num_genes, num_fates=num_fates
    )

    print("Calculating Wild Type baseline...")
    wt_fate_counts = np.zeros(num_fates)
    with torch.no_grad():
        for batch in dataloader:
            u_seq = batch["u_seq"].to(device)
            result_wt = model(u_seq, collocation_t=None)
            dom_fate = result_wt["fate_logits"].mean(dim=1).argmax(dim=-1).cpu().numpy()
            for f in range(num_fates):
                wt_fate_counts[f] += np.sum(dom_fate == f)

    results = []

    print(f"Evaluating causal impact for all {num_tfs} transcription factors...")
    for target_idx in range(num_tfs):
        tf_name = gene_names[target_idx]

        def full_ko_hook(module, input, output):
            output_clone = output.clone()
            output_clone[:, :, target_idx] = 0.0
            return output_clone

        hook_handle = model.rnn.register_forward_hook(full_ko_hook)

        ko_fate_counts = np.zeros(num_fates)
        with torch.no_grad():
            for batch in dataloader:
                u_seq = batch["u_seq"].to(device)
                result_ko = model(u_seq, collocation_t=None)
                dom_fate = result_ko["fate_logits"].mean(dim=1).argmax(dim=-1).cpu().numpy()
                for f in range(num_fates):
                    ko_fate_counts[f] += np.sum(dom_fate == f)

        hook_handle.remove()

        for f in range(num_fates):
            wt_count = wt_fate_counts[f]
            ko_count = ko_fate_counts[f]
            amplitude = wt_count - ko_count # Positive = Driver, Negative = Repressor

            results.append({
                "Transcription_Factor": tf_name,
                "Terminal_Fate": f,
                "WT_Cells": int(wt_count),
                "KO_Cells": int(ko_count),
                "Impact_Amplitude": int(amplitude)
            })

    df = pd.DataFrame(results)
    out_csv = os.path.join(args.outdir, "master_regulator_rankings.csv")
    df.to_csv(out_csv, index=False)

    print(f"\nSuccess! Full ranking exported to: {out_csv}")

    for f in range(num_fates):
        fate_df = df[df["Terminal_Fate"] == f]
        top_driver = fate_df.loc[fate_df["Impact_Amplitude"].idxmax()]
        top_repressor = fate_df.loc[fate_df["Impact_Amplitude"].idxmin()]
        print(f"\n--- Fate {f} Top Regulators ---")
        print(f"Driver: {top_driver['Transcription_Factor']} (Lost {top_driver['Impact_Amplitude']} cells upon KO)")
        print(f"Repressor: {top_repressor['Transcription_Factor']} (Gained {abs(top_repressor['Impact_Amplitude'])} cells upon KO)")

if __name__ == "__main__":
    main()
