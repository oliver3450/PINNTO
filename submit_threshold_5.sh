#!/bin/bash
#SBATCH --job-name=velo_t5
#SBATCH --output=logs/thresh5_%j.log
#SBATCH --error=logs/thresh5_%j.err
#SBATCH --partition=CPU-64C256GB
#SBATCH --qos=qos_cpu_64c256gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --exclude=cnode[66,63,64,60,59,58,12]

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate scvelo-env

python -u check_threshold_5.py
