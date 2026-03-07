#!/bin/bash
#SBATCH --job-name=train_spatial_pinn
#SBATCH --partition=GPU-8A100
#SBATCH --qos=gpu_8a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

echo "Starting Model Training on GPU Node: $HOSTNAME"
echo "Job ID: $SLURM_JOB_ID"

# --- ENVIRONMENT SETUP ---
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate gdl_env

# --- EXECUTION ---
echo "Step 3: Training Hybrid RNN-PINN Model on Fake Data..."
export PYTHONUNBUFFERED=1

python scripts/03_train_model.py \
    --config configs/train_config.yaml \
    --loss_weights configs/loss_weights.yaml \
    --h5ad data/processed/fake_spatial.h5ad

echo "Training Complete. Model weights saved."
