#!/bin/bash
#SBATCH --job-name=velo_211k
#SBATCH --partition=CPU-192C768GB
#SBATCH --qos=qos_cpu_192c768gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G              # Maxed out to safely aggregate 211k barcodes
#SBATCH --time=48:00:00
#SBATCH --output=logs/velo_211k_%j.out
#SBATCH --error=logs/velo_211k_%j.err

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate velocyto

PROJ_DIR="/home/qukungroup/odorn/spatial_mechanistic_model/data/raw/openst_data/spacemake/projects/openst_demo"
GENOME_GTF="/home/qukungroup/odorn/spatial_mechanistic_model/data/raw/openst_data/gencodevM30.annotation.gtf"
BAM_FILE="${PROJ_DIR}/processed_data/openst_demo_e13_mouse_head/illumina/complete_data/final_converted_sorted.bam"
WHITELIST="/home/qukungroup/odorn/spatial_mechanistic_model/data/processed/velocyto_whitelist_211k.txt"
OUT_DIR="${PROJ_DIR}/velocyto_output"

echo "Starting Velocyto on 211k validated tissue beads at $(date)"

velocyto run \
    -b "${WHITELIST}" \
    -@ 32 \
    -o "${OUT_DIR}" \
    "${BAM_FILE}" \
    "${GENOME_GTF}"

if [ $? -eq 0 ]; then
    echo "Success! High-resolution spatial matrix generated at $(date)."
else
    echo "Velocyto failed. Check logs."
    exit 1
fi
