#!/bin/bash
#SBATCH --job-name=pinn_resume
#SBATCH --partition=GPU-8A100
#SBATCH --qos=gpu_8a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/resume_%j.out
#SBATCH --error=logs/resume_%j.err

# ============================================================================
#  SPATIAL MECHANISTIC PINN — Resume Training from Latest Checkpoint
#
#  Usage:
#    sbatch scripts/resume_mousehead.sh
#        Finds the latest checkpoint automatically under results/*/checkpoints/
#
#    sbatch scripts/resume_mousehead.sh results/mousehead_20240101_120000_job99/checkpoints
#        Resume from a specific checkpoint directory (uses best_model.pt inside it)
#
#  The script resolves the checkpoint in priority order:
#    1. Positional argument $1 — explicit checkpoint directory
#    2. Latest results/*/checkpoints/ directory by modification time
#  Always resumes from best_model.pt inside the resolved directory.
#
#  A new timestamped results directory is created for this run so that
#  the original checkpoint directory is never overwritten.
# ============================================================================

set -eo pipefail

# --- Configuration ---
H5AD="data/mouse_brain.h5ad"
TRAIN_CONFIG="configs/train_config.yaml"
LOSS_WEIGHTS="configs/loss_weights.yaml"
PROJECT_DIR="/home/qukungroup/odorn/spatial_mechanistic_model"

cd "${PROJECT_DIR}"
mkdir -p logs

# --- Resolve checkpoint ---
if [ -n "${1:-}" ]; then
    # Explicit directory provided as first positional argument
    RESUME_CKPT_DIR="${1}"
    echo "Using explicitly provided checkpoint directory: ${RESUME_CKPT_DIR}"
else
    # Auto-discover: find the most recently modified checkpoints/ directory
    RESUME_CKPT_DIR=$(find results -maxdepth 2 -type d -name "checkpoints" \
                      -not -empty 2>/dev/null \
                      | xargs -I{} stat --format="%Y {}" {} 2>/dev/null \
                      | sort -n | tail -1 | awk '{print $2}')
    if [ -z "${RESUME_CKPT_DIR}" ]; then
        echo "ERROR: No checkpoint directories found under results/. "
        echo "       Run submit_mousehead.sh first, or pass an explicit path:"
        echo "       sbatch scripts/resume_mousehead.sh results/.../checkpoints"
        exit 1
    fi
    echo "Auto-discovered latest checkpoint directory: ${RESUME_CKPT_DIR}"
fi

RESUME_CKPT="${RESUME_CKPT_DIR}/best_model.pt"
if [ ! -f "${RESUME_CKPT}" ]; then
    echo "ERROR: best_model.pt not found in ${RESUME_CKPT_DIR}"
    echo "       Available files:"
    ls "${RESUME_CKPT_DIR}" 2>/dev/null || echo "       (directory does not exist)"
    exit 1
fi

# --- New results directory for this continuation run ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/resume_${TIMESTAMP}_job${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}/checkpoints"
mkdir -p "${RESULTS_DIR}/config"

echo "============================================"
echo "  PINN Mouse Head — Resume Training"
echo "  Job ID:      ${SLURM_JOB_ID}"
echo "  Node:        ${HOSTNAME}"
echo "  GPUs:        $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "  Resume from: ${RESUME_CKPT}"
echo "  Results:     ${RESULTS_DIR}"
echo "  Dataset:     ${H5AD}"
echo "  Start:       $(date)"
echo "============================================"

# --- WeChat Notification ---
send_wechat_notification() {
    local status=$1
    SENDKEY="YOUR_SERVERCHAN_SENDKEY"
    curl -s -X POST "https://sctapi.ftqq.com/${SENDKEY}.send" \
        -d "title=MouseHead Resume ${status}: ${SLURM_JOB_ID}" \
        -d "desp=Resumed: ${RESUME_CKPT}%0AResults: ${RESULTS_DIR}%0ANode: ${HOSTNAME}%0ATime: $(date)" > /dev/null 2>&1 || true
}

trap 'send_wechat_notification "FAILED"; echo "FAILED at $(date)"' ERR

# --- Environment ---
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate gdl_env
export PYTHONUNBUFFERED=1

# --- Print epoch resumed from (read from checkpoint without loading model) ---
echo ""
echo "Checkpoint metadata:"
python -c "
import torch, sys
ckpt = torch.load('${RESUME_CKPT}', map_location='cpu', weights_only=False)
print(f'  Saved at epoch:  {ckpt[\"epoch\"]}')
print(f'  Best total loss: {ckpt[\"losses\"][\"total\"]:.6f}')
print(f'  Has scheduler:   {\"scheduler_state_dict\" in ckpt}')
print(f'  Moment scales:   {ckpt.get(\"moment_scales\", \"not saved\")}')
"

# --- Snapshot config ---
echo ""
echo "Saving config snapshot..."
cp "${TRAIN_CONFIG}" "${RESULTS_DIR}/config/"
cp "${LOSS_WEIGHTS}" "${RESULTS_DIR}/config/"

cat > "${RESULTS_DIR}/summary.txt" <<SUMMARY
PINN Mouse Head — Resume Run
=============================
Job ID:           ${SLURM_JOB_ID}
Node:             ${HOSTNAME}
Partition:        GPU-8A100
GPUs:             8x A100
Start Time:       $(date)
Resumed From:     ${RESUME_CKPT}
Dataset:          ${H5AD}
Train Config:     ${TRAIN_CONFIG}
Loss Weights:     ${LOSS_WEIGHTS}
Results Dir:      ${RESULTS_DIR}
Python:           $(python --version 2>&1)
PyTorch:          $(python -c "import torch; print(torch.__version__)")
CUDA:             $(python -c "import torch; print(torch.version.cuda)")
SUMMARY

# --- Train (resumed) ---
echo ""
echo "Starting resumed training..."
echo "============================================"

python scripts/03_train_model.py \
    --config "${TRAIN_CONFIG}" \
    --loss_weights "${LOSS_WEIGHTS}" \
    --h5ad "${H5AD}" \
    --checkpoint_dir "${RESULTS_DIR}/checkpoints" \
    --resume "${RESUME_CKPT}"

# --- Post-training ---
echo ""
echo "============================================"
echo "  Training Complete"
echo "  End:     $(date)"
echo "  Results: ${RESULTS_DIR}"
echo "============================================"

echo "" >> "${RESULTS_DIR}/summary.txt"
echo "End Time:         $(date)" >> "${RESULTS_DIR}/summary.txt"
echo "Exit Status:      SUCCESS" >> "${RESULTS_DIR}/summary.txt"

mkdir -p "${RESULTS_DIR}/logs"
cp "logs/resume_${SLURM_JOB_ID}.out" "${RESULTS_DIR}/logs/" 2>/dev/null || true
cp "logs/resume_${SLURM_JOB_ID}.err" "${RESULTS_DIR}/logs/" 2>/dev/null || true

send_wechat_notification "COMPLETE"
echo "Done."
