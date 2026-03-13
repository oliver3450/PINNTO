#!/bin/bash
#SBATCH --job-name=pinn_full_pipeline
#SBATCH --partition=GPU-8A100
#SBATCH --qos=gpu_8a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err

# ============================================================================
#  FULL PIPELINE: Palantir -> Build Spatial -> Train PINN
#
#  Runs the entire preprocessing + training pipeline in one SLURM job.
#  Step 1 & 2 use CPU only (Palantir + data assembly).
#  Step 3 uses all 8 GPUs for PINN training.
#
#  Usage: sbatch scripts/submit_preprocess_and_train.sh
# ============================================================================

set -eo pipefail

# --- Configuration ---
RAW_H5AD="data/mouse_brain.h5ad"
PALANTIR_H5AD="data/processed/palantir_adata.h5ad"
TRAINING_H5AD="data/processed/spatial_adata.h5ad"
TRAIN_CONFIG="configs/train_config.yaml"
LOSS_WEIGHTS="configs/loss_weights.yaml"
PROJECT_DIR="/home/qukungroup/odorn/spatial_mechanistic_model"

cd "${PROJECT_DIR}"
mkdir -p logs data/processed

# --- Results Directory ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/pipeline_${TIMESTAMP}_job${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}/checkpoints"
mkdir -p "${RESULTS_DIR}/config"
mkdir -p "${RESULTS_DIR}/logs"

echo "============================================"
echo "  PINN Full Pipeline"
echo "  Job ID:      ${SLURM_JOB_ID}"
echo "  Node:        ${HOSTNAME}"
echo "  GPUs:        $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "  Results:     ${RESULTS_DIR}"
echo "  Start:       $(date)"
echo "============================================"

# --- WeChat Notification ---
send_wechat_notification() {
    local status=$1
    SENDKEY="YOUR_SERVERCHAN_SENDKEY"
    curl -s -X POST "https://sctapi.ftqq.com/${SENDKEY}.send" \
        -d "title=Pipeline ${status}: ${SLURM_JOB_ID}" \
        -d "desp=Results: ${RESULTS_DIR}%0ANode: ${HOSTNAME}%0ATime: $(date)" > /dev/null 2>&1 || true
}

trap 'send_wechat_notification "FAILED"; echo "FAILED at $(date)"' ERR

# --- Environment ---
source $HOME/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1

# ===== STEP 1: Palantir (CPU, palantir env) =====
echo ""
echo "[Step 1/4] Running Palantir for pseudotime + fate probabilities..."
echo "============================================"

conda activate palantir

python scripts/01_run_palantir.py \
    --h5ad "${RAW_H5AD}" \
    --output "${PALANTIR_H5AD}" \
    --n_components 30 \
    --n_diffusion 10

echo "Palantir complete at $(date)"

# ===== STEP 2: Build Training-Ready h5ad (CPU, palantir env) =====
echo ""
echo "[Step 2/4] Building training-ready h5ad..."
echo "============================================"

python scripts/02_build_spatial.py \
    --h5ad "${RAW_H5AD}" \
    --palantir "${PALANTIR_H5AD}" \
    --output "${TRAINING_H5AD}"

echo "Spatial data built at $(date)"

# ===== STEP 3: Extract Gene List + Snapshot Config =====
echo ""
echo "[Step 3/4] Saving config snapshot..."

conda activate gdl_env

cp "${TRAIN_CONFIG}" "${RESULTS_DIR}/config/"
cp "${LOSS_WEIGHTS}" "${RESULTS_DIR}/config/"

cat > "${RESULTS_DIR}/summary.txt" <<SUMMARY
PINN Full Pipeline Run
=============================
Job ID:           ${SLURM_JOB_ID}
Node:             ${HOSTNAME}
Partition:        GPU-8A100
GPUs:             8x A100
Start Time:       $(date)
Raw Dataset:      ${RAW_H5AD}
Training Dataset: ${TRAINING_H5AD}
Train Config:     ${TRAIN_CONFIG}
Loss Weights:     ${LOSS_WEIGHTS}
Results Dir:      ${RESULTS_DIR}
Python:           $(python --version 2>&1)
PyTorch:          $(python -c "import torch; print(torch.__version__)")
CUDA:             $(python -c "import torch; print(torch.version.cuda)")
SUMMARY

# ===== STEP 4: Train PINN =====
echo ""
echo "[Step 4/4] Starting PINN training..."
echo "============================================"

python scripts/03_train_model.py \
    --config "${TRAIN_CONFIG}" \
    --loss_weights "${LOSS_WEIGHTS}" \
    --h5ad "${TRAINING_H5AD}" \
    --checkpoint_dir "${RESULTS_DIR}/checkpoints"

# --- Post-Training ---
echo ""
echo "============================================"
echo "  Pipeline Complete"
echo "  End:     $(date)"
echo "  Results: ${RESULTS_DIR}"
echo "============================================"

echo "" >> "${RESULTS_DIR}/summary.txt"
echo "End Time:         $(date)" >> "${RESULTS_DIR}/summary.txt"
echo "Exit Status:      SUCCESS" >> "${RESULTS_DIR}/summary.txt"

cp "logs/pipeline_${SLURM_JOB_ID}.out" "${RESULTS_DIR}/logs/" 2>/dev/null || true
cp "logs/pipeline_${SLURM_JOB_ID}.err" "${RESULTS_DIR}/logs/" 2>/dev/null || true

send_wechat_notification "COMPLETE"
echo "Done."
