#!/bin/bash
#SBATCH --job-name=stitchvelo
#SBATCH --output=logs/prop_%j.log
#SBATCH --error=logs/prop_%j.err
#SBATCH --partition=CPU-64C256GB
#SBATCH --qos=qos_cpu_64c256gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --exclude=cnode[66,63,64,60,59,58,12]

# --- Setup ---
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate scvelo-env

# --- Execution ---
# Note: Ensure this python file is in the directory where you run the sbatch command.
# If it is inside the scripts folder, change this to: python scripts/submit_02_propcheck.py
python check_proportions.py
