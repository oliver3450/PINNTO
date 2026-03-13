#!/bin/bash
#SBATCH --job-name=momscales
#SBATCH -J submitscalecheck
#SBATCH -o logs/momscales_%j.log
#SBATCH -e logs/momscales_%j.err
#SBATCH --qos=qos_cpu_64c256gb
#SBATCH -N 1 -n 1 -p CPU-64C256GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclude=cnode[66,63,64,60,59,58,12]

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate scvelo-env

python 01_moment_scales.py

echo 'All tasks finished at $(date)'
