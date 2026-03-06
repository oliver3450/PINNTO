#!/bin/bash

# Configuration - Update these to your exact WSL/HPC details
HPC_USER="odorn"
HPC_HOST="hanhai22-01"
HPC_DIR="~/spatial_mechanistic_model/"
LOCAL_DIR="./"

case $1 in
    pull)
        echo "Syncing HPC -> WSL..."
        rsync -avzP ${HPC_USER}@${HPC_HOST}:${HPC_DIR} ${LOCAL_DIR}
        ;;
    push-hpc)
        echo "Syncing WSL -> HPC..."
        # Exclude heavy data and logs
        rsync -avzP --exclude 'logs/' --exclude 'checkpoints/' --exclude 'data/' ${LOCAL_DIR} ${HPC_USER}@${HPC_HOST}:${HPC_DIR}
        ;;
    push-github)
        echo "Pushing to GitHub..."
        git add .
        git commit -m "Fix DataParallel gather shape mismatch and scale to 8-A100"
        git push origin main
        ;;
    *)
        echo "Usage: ./manage_hpc.sh {pull|push-hpc|push-github}"
        ;;
esac
