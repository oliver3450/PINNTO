#!/bin/bash
#SBATCH --job-name=train_spatial_pinn
#SBATCH --partition=GPU-8A100
#SBATCH --qos=gpu_8a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8           # Just enough for PyTorch DataLoader workers
#SBATCH --gres=gpu:1                # Request exactly 1 A100 GPU
#SBATCH --mem=64G                   # Neural network batches don't need massive RAM
#SBATCH --time=3-00:00:00           # Give PINNs time to converge (up to 3 days)
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
# To chain after preprocessing automatically, submit with:
#   sbatch --dependency=afterok:<PREP_JOB_ID> submit_02_training.sh

send_wechat_notification() {
    local job_name=$1
    local job_id=$2
    local status=$3

    SENDKEY="SCT311572TRDACiXkpHXrwquiUYvoUaJOp"

    curl -X POST "https://sctapi.ftqq.com/${SENDKEY}.send" \
        -d "title=Job ${status}: ${job_name}" \
        -d "desp=Job ID: ${job_id}%0A%0AStatus: ${status}%0A%0ACheck your logs for details."
}

echo "Starting Model Training on GPU Node: $HOSTNAME"
echo "Job ID: $SLURM_JOB_ID"

# --- ENVIRONMENT SETUP ---
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate ENVNAME

trap 'send_wechat_notification "$SLURM_JOB_NAME" "$SLURM_JOB_ID" "FAILED"' ERR

send_wechat_notification "$SLURM_JOB_NAME" "$SLURM_JOB_ID" "STARTED"

# --- EXECUTION ---
echo "Step 3: Training Hybrid RNN-PINN Model..."
python scripts/03_train_model.py \
    --config configs/train_config.yaml \
    --loss_weights configs/loss_weights.yaml

echo "Training Complete. Model weights saved."
send_wechat_notification "$SLURM_JOB_NAME" "$SLURM_JOB_ID" "COMPLETED"
