#!/bin/bash
#SBATCH --job-name=prep_spatial_pinn
#SBATCH --partition=CPU-192C768GB
#SBATCH --qos=qos_cpu_192c768gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48          # Give Palantir/Scanpy plenty of workers
#SBATCH --mem=256G                  # Safe buffer for 10k+ cell spatial distance matrices
#SBATCH --output=logs/prep_%j.out
#SBATCH --error=logs/prep_%j.err
# To chain after Spacemake automatically, submit with:
#   sbatch --dependency=afterok:<SPACEMAKE_JOB_ID> submit_01_preprocessing.sh

send_wechat_notification() {
    local job_name=$1
    local job_id=$2
    local status=$3

    SENDKEY="SCT311572TRDACiXkpHXrwquiUYvoUaJOp"

    curl -X POST "https://sctapi.ftqq.com/${SENDKEY}.send" \
        -d "title=Job ${status}: ${job_name}" \
        -d "desp=Job ID: ${job_id}%0A%0AStatus: ${status}%0A%0ACheck your logs for details."
}

echo "Starting Data Preprocessing on CPU Node: $HOSTNAME"
echo "Job ID: $SLURM_JOB_ID"

# --- ENVIRONMENT SETUP ---
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate ENVNAME

trap 'send_wechat_notification "$SLURM_JOB_NAME" "$SLURM_JOB_ID" "FAILED"' ERR

send_wechat_notification "$SLURM_JOB_NAME" "$SLURM_JOB_ID" "STARTED"

# --- EXECUTION ---
echo "Step 1: Running Palantir Topology Inference..."
python scripts/01_run_palantir.py --config configs/train_config.yaml

echo "Step 2: Building Spatial Forcing Intervals..."
python scripts/02_build_spatial.py --config configs/train_config.yaml

echo "Preprocessing Complete. Data saved to data/processed/"
send_wechat_notification "$SLURM_JOB_NAME" "$SLURM_JOB_ID" "COMPLETED"
