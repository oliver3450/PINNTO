#!/bin/bash
#SBATCH --job-name=openst_align
#SBATCH --partition=CPU-192C768GB
#SBATCH --qos=qos_cpu_192c768gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24          
#SBATCH --mem=256G                  
#SBATCH --output=logs/align_%j.out
#SBATCH --error=logs/align_%j.err

echo "Starting Spacemake Alignment Job..."
echo "Running on node: $HOSTNAME"

# 1. Activate the transplanted offline environment
source ~/envs/spacemake/bin/activate

# 2. Navigate into the configured spacemake directory
cd ~/spatial_mechanistic_model/data/raw/openst_data/spacemake

# 3. Execute the pipeline
echo "Executing spacemake run..."
spacemake run --cores 48

echo "Alignment Complete. Awaiting BAM file extraction."
