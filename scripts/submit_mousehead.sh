#!/bin/bash
#SBATCH --job-name=pinn_mousehead
#SBATCH --partition=GPU-8A100
#SBATCH --qos=gpu_8a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/mousehead_%j.out
#SBATCH --error=logs/mousehead_%j.err

# ============================================================================
#  SPATIAL MECHANISTIC PINN — E13 Mouse Head Training Run
#
#  Usage: sbatch scripts/submit_mousehead.sh
#
#  Creates a timestamped results folder with:
#    checkpoints/   — best_model.pt + periodic saves
#    config/        — frozen copy of train_config.yaml + loss_weights.yaml
#    logs/          — stdout/stderr copied after training
#    summary.txt    — run metadata (job ID, GPU count, timing, etc.)
# ============================================================================

set -eo pipefail

# --- Configuration ---
H5AD="data/processed/mouse_brain.h5ad"
TRAIN_CONFIG="configs/train_config.yaml"
LOSS_WEIGHTS="configs/loss_weights.yaml"
PROJECT_DIR="/home/qukungroup/odorn/spatial_mechanistic_model"

cd "${PROJECT_DIR}"
mkdir -p logs

# --- Results Directory ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/mousehead_${TIMESTAMP}_job${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}/checkpoints"
mkdir -p "${RESULTS_DIR}/config"

echo "============================================"
echo "  PINN Mouse Head Training"
echo "  Job ID:      ${SLURM_JOB_ID}"
echo "  Node:        ${HOSTNAME}"
echo "  GPUs:        $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "  Results:     ${RESULTS_DIR}"
echo "  Dataset:     ${H5AD}"
echo "  Start:       $(date)"
echo "============================================"

# --- WeChat Notification ---
send_wechat_notification() {
    local status=$1
    SENDKEY="YOUR_SERVERCHAN_SENDKEY"
    curl -s -X POST "https://sctapi.ftqq.com/${SENDKEY}.send" \
        -d "title=MouseHead ${status}: ${SLURM_JOB_ID}" \
        -d "desp=Results: ${RESULTS_DIR}%0ANode: ${HOSTNAME}%0ATime: $(date)" > /dev/null 2>&1 || true
}

trap 'send_wechat_notification "FAILED"; echo "FAILED at $(date)"' ERR

# --- Environment ---
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate gdl_env
export PYTHONUNBUFFERED=1

# --- Step 1: Extract Gene List from h5ad ---
echo ""
echo "[Step 1/3] Extracting expressed gene list from ${H5AD}..."

python -c "
import scanpy as sc
adata = sc.read_h5ad('${H5AD}')
genes = adata.var_names.tolist()
with open('data/processed/expressed_genes.csv', 'w') as f:
    for g in genes:
        f.write(g + '\n')
print(f'Wrote {len(genes)} genes to data/processed/expressed_genes.csv')
print(f'Dataset: {adata.n_obs} cells x {adata.n_vars} genes')
print(f'Layers: {list(adata.layers.keys())}')
print(f'Spatial coords shape: {adata.obsm[\"spatial\"].shape}')
"

# --- Step 2: Snapshot Config ---
echo ""
echo "[Step 2/3] Saving config snapshot..."
cp "${TRAIN_CONFIG}" "${RESULTS_DIR}/config/"
cp "${LOSS_WEIGHTS}" "${RESULTS_DIR}/config/"

# Record run metadata
cat > "${RESULTS_DIR}/summary.txt" <<SUMMARY
PINN Mouse Head Training Run
=============================
Job ID:           ${SLURM_JOB_ID}
Node:             ${HOSTNAME}
Partition:        GPU-8A100
GPUs:             8x A100
Start Time:       $(date)
Dataset:          ${H5AD}
Train Config:     ${TRAIN_CONFIG}
Loss Weights:     ${LOSS_WEIGHTS}
Results Dir:      ${RESULTS_DIR}
Python:           $(python --version 2>&1)
PyTorch:          $(python -c "import torch; print(torch.__version__)")
CUDA:             $(python -c "import torch; print(torch.version.cuda)")
SUMMARY

# --- Step 3: Train ---
echo ""
echo "[Step 3/3] Starting training..."
echo "============================================"

python scripts/03_train_model.py \
    --config "${TRAIN_CONFIG}" \
    --loss_weights "${LOSS_WEIGHTS}" \
    --h5ad "${H5AD}" \
    --checkpoint_dir "${RESULTS_DIR}/checkpoints"

# --- Post-Training ---
echo ""
echo "============================================"
echo "  Training Complete"
echo "  End:     $(date)"
echo "  Results: ${RESULTS_DIR}"
echo "============================================"

# Append end time to summary
echo "" >> "${RESULTS_DIR}/summary.txt"
echo "End Time:         $(date)" >> "${RESULTS_DIR}/summary.txt"
echo "Exit Status:      SUCCESS" >> "${RESULTS_DIR}/summary.txt"

# Copy logs into results folder
cp "logs/mousehead_${SLURM_JOB_ID}.out" "${RESULTS_DIR}/logs/" 2>/dev/null || true
cp "logs/mousehead_${SLURM_JOB_ID}.err" "${RESULTS_DIR}/logs/" 2>/dev/null || true
mkdir -p "${RESULTS_DIR}/logs"
cp "logs/mousehead_${SLURM_JOB_ID}.out" "${RESULTS_DIR}/logs/" 2>/dev/null || true
cp "logs/mousehead_${SLURM_JOB_ID}.err" "${RESULTS_DIR}/logs/" 2>/dev/null || true

send_wechat_notification "COMPLETE"
echo "Done."
