#!/bin/bash
#SBATCH --job-name=spacemake_quantification
#SBATCH --partition=CPU-192C768GB
#SBATCH --qos=qos_cpu_192c768gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48          # Spacemake/Snakemake parallelises alignment heavily
#SBATCH --mem=256G                  # STAR genome index alone can use ~32G
#SBATCH --time=2-00:00:00           # Alignment of large spatial libraries can take hours
#SBATCH --output=logs/spacemake_%j.out
#SBATCH --error=logs/spacemake_%j.err

send_wechat_notification() {
    local job_name=$1
    local job_id=$2
    local status=$3

    SENDKEY="SCT311572TRDACiXkpHXrwquiUYvoUaJOp"

    curl -X POST "https://sctapi.ftqq.com/${SENDKEY}.send" \
        -d "title=Job ${status}: ${job_name}" \
        -d "desp=Job ID: ${job_id}%0A%0AStatus: ${status}%0A%0ACheck your logs for details."
}

echo "Starting Spacemake Quantification on CPU Node: $HOSTNAME"
echo "Job ID: $SLURM_JOB_ID"

# --- PATHS (update these to match your cluster layout) ---
SIF=/path/to/spacemake.sif          # Absolute path to your portable .sif file
PROJECT_DIR=$(pwd)                  # Spacemake writes output relative to where it's run
RAW_DATA_DIR=${PROJECT_DIR}/data/raw
GENOME_DIR=/path/to/genome/reference  # STAR index + GTF (pre-built, cluster-shared)

# --- APPTAINER BIND MOUNTS ---
# --bind makes host directories visible inside the container.
# Add any additional paths your genome/barcode files live on.
BINDS="${PROJECT_DIR}:${PROJECT_DIR},${GENOME_DIR}:${GENOME_DIR}"

trap 'send_wechat_notification "$SLURM_JOB_NAME" "$SLURM_JOB_ID" "FAILED"' ERR

send_wechat_notification "$SLURM_JOB_NAME" "$SLURM_JOB_ID" "STARTED"

# --- EXECUTION ---
# Spacemake is run from the project root. The container is stateless —
# all outputs land on the host via the bind mount.

echo "Running Spacemake pipeline..."
apptainer exec \
    --bind ${BINDS} \
    --pwd  ${PROJECT_DIR} \
    ${SIF} \
    spacemake run \
        --cores ${SLURM_CPUS_PER_TASK} \
        --keep-going              # Don't abort entire run if one sample fails

echo "Spacemake Complete. .h5ad files ready in data/raw/"
send_wechat_notification "$SLURM_JOB_NAME" "$SLURM_JOB_ID" "COMPLETED"
