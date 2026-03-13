#!/bin/bash
#SBATCH --job-name=kinetics
#SBATCH -J kinetcis
#SBATCH -o logs/plotkinetic_%j.log
#SBATCH -e logs/plotkinetics_%j.err
#SBATCH --qos=qos_cpu_64c256gb
#SBATCH -N 1 -n 1 -p CPU-64C256GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclude=cnode[66,63,64,60,59,58,12]

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate gdl_env

python scripts/diagnostics/02_plot_kinetics.py \
    --checkpoint checkpoints/checkpoint_epoch_500.pt \
    --output logs/kinetic_distributions_epoch500.png

echo 'All tasks finished at $(date)'
