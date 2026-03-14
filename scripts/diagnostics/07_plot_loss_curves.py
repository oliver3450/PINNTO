"""
07_plot_loss_curves.py — Plot training loss curves from checkpoint files.

Scans a checkpoint directory for all *.pt files, extracts epoch + losses,
and produces a 4-panel figure: total, data, physics, fate losses.

Usage:
    python scripts/diagnostics/07_plot_loss_curves.py \
        --checkpoint_dir results/pipeline_TIMESTAMP_JOBID/checkpoints \
        --outdir logs/evaluation
"""

import argparse
import glob
import os
import sys

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Directory containing *.pt checkpoint files")
    parser.add_argument("--outdir", default="logs/evaluation")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    pts = sorted(glob.glob(os.path.join(args.checkpoint_dir, "*.pt")))
    if not pts:
        print(f"No .pt files found in {args.checkpoint_dir}")
        sys.exit(1)

    records = []
    for path in pts:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        records.append({
            "epoch": ck["epoch"],
            "total": ck["losses"]["total"],
            "data":  ck["losses"]["data"],
            "phys":  ck["losses"]["phys"],
            "fate":  ck["losses"]["fate"],
            "is_best": "best_model" in os.path.basename(path),
        })

    records.sort(key=lambda r: r["epoch"])
    print(f"Found {len(records)} checkpoints, epochs: {[r['epoch'] for r in records]}")

    epochs = [r["epoch"] for r in records]
    best_epochs = [r["epoch"] for r in records if r["is_best"]]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle("Training Loss Curves", fontsize=14)

    panels = [
        ("total", "Total Loss",          "#222222"),
        ("data",  "Data Loss (L_data)",  "#3274A1"),
        ("phys",  "Physics Loss (L_phys)","#E1812C"),
        ("fate",  "Fate Loss (L_fate)",  "#3A923A"),
    ]

    for ax, (key, title, color) in zip(axes.flat, panels):
        vals = [r[key] for r in records]
        ax.plot(epochs, vals, color=color, linewidth=2, marker="o", markersize=4)
        # Mark best checkpoint(s)
        for be in best_epochs:
            idx = epochs.index(be)
            ax.axvline(x=be, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.scatter([be], [vals[idx]], color="red", zorder=5, s=40)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle="--", alpha=0.4)

    # Legend for best marker
    from matplotlib.lines import Line2D
    legend_el = [Line2D([0], [0], color="red", linestyle="--", linewidth=0.8, label="best_model.pt")]
    axes.flat[0].legend(handles=legend_el, fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(args.outdir, "loss_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
