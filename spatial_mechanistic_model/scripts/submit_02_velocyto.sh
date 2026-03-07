#!/bin/bash
#SBATCH --job-name=openst_velocyto
#SBATCH --partition=CPU-64C256GB
#SBATCH --qos=qos_cpu_64c256gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=logs/velo_%j.out
#SBATCH --error=logs/velo_%j.err

# --- Helper Function ---
send_wechat_notification() {
    local job_name=$1
    local job_id=$2
    local status=$3

    # Paste your actual key here
    SENDKEY="YOUR_SERVERCHAN_SENDKEY"

    curl -s -X POST "https://sctapi.ftqq.com/${SENDKEY}.send" \
        -d "title=Job ${status}: ${job_name}" \
        -d "desp=Job ID: ${job_id}%0A%0AStatus: ${status}%0A%0ACheck your logs for details." > /dev/null
}

# --- Setup ---
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate velocyto

# Ensure logs directory exists (prevents silent failures)
cd /home/qukungroup/odorn/spatial_mechanistic_model/scripts/
mkdir -p logs

# --- Define Absolute Paths ---
RAW_DIR="/home/qukungroup/odorn/spatial_mechanistic_model/data/raw"
PROJ_DIR="${RAW_DIR}/openst_data/spacemake/projects/openst_demo"

# CHANGE THESE:
GENOME_FA="${RAW_DIR}/openst_data/GRCm39vM30.genome.fa"
GENOME_GTF="${RAW_DIR}/openst_data/gencodevM30.annotation.gtf"

CRAM_FILE="${PROJ_DIR}/processed_data/openst_demo_e13_mouse_head/illumina/complete_data/final.polyA_adapter_trimmed.cram"
BAM_FILE="${PROJ_DIR}/processed_data/openst_demo_e13_mouse_head/illumina/complete_data/final_converted.bam"
OUT_DIR="${PROJ_DIR}/velocyto_output"

# --- Execution Step 1: Decompress CRAM to BAM ---
echo "Starting CRAM to BAM conversion at $(date)"

# Samtools requires the original genome sequence to rebuild the BAM
samtools view -b -T "${GENOME_FA}" -@ 32 -o "${BAM_FILE}" "${CRAM_FILE}"

if [ $? -ne 0 ]; then
    echo "samtools conversion FAILED."
    send_wechat_notification "CRAM_to_BAM_FAIL" "$SLURM_JOB_ID" "failed"
    exit 1
fi

# --- Execution Step 2: Extract Spliced/Unspliced Counts ---
echo "Starting Velocyto processing at $(date)"

# -c XC and -U XM tell Velocyto where to find Spacemake's spatial barcodes and UMIs
velocyto run \
    -@ 32 \
    -c XC \
    -U XM \
    -o "${OUT_DIR}" \
    "${BAM_FILE}" \
    "${GENOME_GTF}"

if [ $? -eq 0 ]; then
    echo "Velocyto finished successfully at $(date)."
    
    # Cleanup the massive intermediate BAM file to save hard drive space
    rm "${BAM_FILE}"
    
    send_wechat_notification "openst_velocyto" "$SLURM_JOB_ID" "completed"
else
    echo "Velocyto FAILED at $(date)."
    send_wechat_notification "openst_velocyto_FAIL" "$SLURM_JOB_ID" "failed"
    exit 1
fi
