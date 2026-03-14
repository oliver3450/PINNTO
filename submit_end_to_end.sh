#!/bin/bash
#SBATCH --job-name=pipeline_e2e
#SBATCH --partition=GPU-8A100
#SBATCH --qos=gpu_8a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err

# ============================================================================
#  END-TO-END SPATIAL MECHANISTIC PINN PIPELINE
#
#  Single-command execution:  sbatch submit_end_to_end.sh
#
#  Stages:
#    1. Preprocess  — scVelo + Palantir (CPU, ~20 min)
#    2. Train PINN  — 500 epochs on 8x A100 (~2-8 hrs depending on dataset)
#    3. Evaluate    — Publication spatial map + temporal knockout plots (CPU)
#
#  All outputs land in a timestamped results/ directory.
# ============================================================================

set -e

# --- Configuration ---
PROJECT_DIR="/home/qukungroup/odorn/spatial_mechanistic_model"
H5AD="data/mouse_brain.h5ad"
TRAIN_CONFIG="configs/train_config.yaml"
LOSS_WEIGHTS="configs/loss_weights.yaml"
PROCESSED_H5AD="data/processed/spatial_adata.h5ad"
GENES_CSV="data/processed/expressed_genes.csv"

cd "${PROJECT_DIR}"
mkdir -p logs data/processed

# --- Results Directory ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/pipeline_${TIMESTAMP}_job${SLURM_JOB_ID}"
CHECKPOINT_DIR="${RESULTS_DIR}/checkpoints"
EVAL_DIR="${RESULTS_DIR}/evaluation"
mkdir -p "${CHECKPOINT_DIR}" "${RESULTS_DIR}/config" "${EVAL_DIR}"

echo "============================================"
echo "  END-TO-END PINN PIPELINE"
echo "  Job ID:      ${SLURM_JOB_ID}"
echo "  Node:        ${HOSTNAME}"
echo "  GPUs:        $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "  Results:     ${RESULTS_DIR}"
echo "  Start:       $(date)"
echo "============================================"

# --- Environment ---
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate gdl_env
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/gdl_env/lib:${LD_LIBRARY_PATH}"

# Snapshot config
cp "${TRAIN_CONFIG}" "${RESULTS_DIR}/config/"
cp "${LOSS_WEIGHTS}" "${RESULTS_DIR}/config/"

# ====================================================================
# STAGE 1: PREPROCESSING (scVelo + Palantir)
# ====================================================================
echo ""
echo "============================================"
echo "  STAGE 1/3: Preprocessing"
echo "  $(date)"
echo "============================================"

python scripts/00_master_preprocess.py \
    --h5ad "${H5AD}" \
    --output "${PROCESSED_H5AD}" \
    --genes_csv "${GENES_CSV}" \
    --n_hvg 2000 \
    --n_pca 30 \
    --n_diffusion 10

# Verify outputs exist
if [ ! -f "${PROCESSED_H5AD}" ]; then
    echo "FATAL: Preprocessing failed — ${PROCESSED_H5AD} not found."
    exit 1
fi
if [ ! -f "${GENES_CSV}" ]; then
    echo "FATAL: Preprocessing failed — ${GENES_CSV} not found."
    exit 1
fi

echo "Preprocessing complete. Output: ${PROCESSED_H5AD}"

# ====================================================================
# STAGE 2: PINN TRAINING
# ====================================================================
echo ""
echo "============================================"
echo "  STAGE 2/3: PINN Training"
echo "  $(date)"
echo "============================================"

python scripts/03_train_model.py \
    --config "${TRAIN_CONFIG}" \
    --loss_weights "${LOSS_WEIGHTS}" \
    --h5ad "${PROCESSED_H5AD}" \
    --checkpoint_dir "${CHECKPOINT_DIR}"

# Verify checkpoint exists
if [ ! -f "${CHECKPOINT_DIR}/best_model.pt" ]; then
    echo "FATAL: Training failed — best_model.pt not found."
    exit 1
fi

echo "Training complete. Checkpoint: ${CHECKPOINT_DIR}/best_model.pt"

# ====================================================================
# STAGE 3: EVALUATION (Publication Figures)
# ====================================================================
echo ""
echo "============================================"
echo "  STAGE 3/3: Evaluation"
echo "  $(date)"
echo "============================================"

python scripts/04_publishable_eval.py \
    --checkpoint "${CHECKPOINT_DIR}/best_model.pt" \
    --h5ad "${PROCESSED_H5AD}" \
    --genes "${GENES_CSV}" \
    --outdir "${EVAL_DIR}"

# ====================================================================
# DONE
# ====================================================================
echo ""
echo "============================================"
echo "  PIPELINE COMPLETE"
echo "  End:         $(date)"
echo "  Results:     ${RESULTS_DIR}"
echo "  Checkpoint:  ${CHECKPOINT_DIR}/best_model.pt"
echo "  Figures:     ${EVAL_DIR}/"
echo "============================================"

# Record summary
cat > "${RESULTS_DIR}/summary.txt" <<SUMMARY
End-to-End PINN Pipeline
========================
Job ID:           ${SLURM_JOB_ID}
Node:             ${HOSTNAME}
Start:            ${TIMESTAMP}
End:              $(date +%Y%m%d_%H%M%S)
Dataset:          ${H5AD}
Processed:        ${PROCESSED_H5AD}
Gene List:        ${GENES_CSV}
Train Config:     ${TRAIN_CONFIG}
Loss Weights:     ${LOSS_WEIGHTS}
Checkpoint:       ${CHECKPOINT_DIR}/best_model.pt
Evaluation:       ${EVAL_DIR}/
Python:           $(python --version 2>&1)
PyTorch:          $(python -c "import torch; print(torch.__version__)")
CUDA:             $(python -c "import torch; print(torch.version.cuda)")
SUMMARY

# Copy logs into results
mkdir -p "${RESULTS_DIR}/logs"
cp "logs/pipeline_${SLURM_JOB_ID}.out" "${RESULTS_DIR}/logs/" 2>/dev/null || true
cp "logs/pipeline_${SLURM_JOB_ID}.err" "${RESULTS_DIR}/logs/" 2>/dev/null || true

echo "Done."
