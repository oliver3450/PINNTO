#!/bin/bash
#SBATCH --job-name=posthoc_plots
#SBATCH -o logs/posthoc_%j.out
#SBATCH -e logs/posthoc_%j.err
#SBATCH --qos=qos_cpu_64c256gb
#SBATCH -N 1 -n 1 -p CPU-64C256GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --exclude=cnode[66,63,64,60,59,58,12]

# ============================================================================
#  POST-HOC ANALYSIS SUITE — edit RESULTS_DIR before submitting
#
#  Usage: sbatch scripts/diagnostics/submit_posthoc_plots.sh
# ============================================================================

set -e

# >>> EDIT THIS to your timestamped results directory <<<
RESULTS_DIR="results/pipeline_20260315_025229_job852211"

# Derived paths (no need to edit)
PROJECT_DIR="/home/qukungroup/odorn/spatial_mechanistic_model"
CHECKPOINT="${RESULTS_DIR}/checkpoints/best_model.pt"
CHECKPOINT_DIR="${RESULTS_DIR}/checkpoints"
H5AD="data/processed/spatial_adata.h5ad"
GENES="data/processed/expressed_genes.csv"
OUTDIR="${RESULTS_DIR}/evaluation"

cd "${PROJECT_DIR}"
mkdir -p logs "${OUTDIR}"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate gdl_env
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/gdl_env/lib:${LD_LIBRARY_PATH}"

echo "========================================"
echo "  Post-hoc Analysis Suite"
echo "  Results: ${RESULTS_DIR}"
echo "  Start:   $(date)"
echo "========================================"

# --- 1. Loss Curves ---
echo ""
echo "[1/3] Loss curves..."
python scripts/diagnostics/07_plot_loss_curves.py \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --outdir "${OUTDIR}"

# --- 2. Diagnostic Plots ---
echo ""
echo "[2/3] Diagnostic plots (pseudotime, entropy, fate maps, kinetics)..."
python scripts/diagnostics/09_plot_diagnostics.py \
    --checkpoint "${CHECKPOINT}" \
    --h5ad "${H5AD}" \
    --genes "${GENES}" \
    --outdir "${OUTDIR}"

# --- 3. Spatial KO Atlas (all TFs x all fates) ---
echo ""
echo "[3/3] Spatial KO atlas (all TFs)..."
python scripts/diagnostics/08_plot_spatial_ko_atlas.py \
    --checkpoint "${CHECKPOINT}" \
    --h5ad "${H5AD}" \
    --genes "${GENES}" \
    --outdir "${OUTDIR}"

echo ""
echo "========================================"
echo "  Done: $(date)"
echo "  Figures: ${OUTDIR}/"
echo "========================================"
