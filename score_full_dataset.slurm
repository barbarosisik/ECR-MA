#!/bin/bash

# To run on gpu-medium: sbatch score_full_dataset.slurm
# To run on gpu-long: change '#SBATCH --partition=gpu-medium' to '#SBATCH --partition=gpu-long' below

#SBATCH --partition=gpu-medium
#SBATCH --gpus=1
#SBATCH --job-name=llama2_score_full
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_outputs/llama2_score_full_%A.out
#SBATCH --error=slurm_outputs/llama2_score_full_%A.err

# Exit on any error
set -e

# Load modules
module purge
module load ALICE/default
module load Miniconda3/23.9.0-0
module load CUDA/12.1.1

# Environment setup
export HF_HOME="/data1/s3905993/cache/huggingface"
export TRANSFORMERS_CACHE="/data1/s3905993/cache/transformers"
mkdir -p slurm_outputs
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE"

# Activate conda environment
source activate /data1/s3905993/conda_envs/ecrhmas_fixed

# Ensure we're using the correct Python
export PATH="/data1/s3905993/conda_envs/ecrhmas_fixed/bin:$PATH"
export PYTHONPATH="/data1/s3905993/conda_envs/ecrhmas_fixed/lib/python3.10/site-packages:$PYTHONPATH"

# Set working directory
cd /data1/s3905993/ECR-main

# Debug info
echo "=== ENVIRONMENT VERIFICATION ==="
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "PyTorch version: $(python -c "import torch; print(torch.__version__)")"
echo "CUDA available: $(python -c "import torch; print(torch.cuda.is_available())")"
echo "Working dir: $(pwd)"
nvidia-smi

echo "=== LLAMA2 FULL DATASET SCORING JOB ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Run the scoring script on full dataset
echo "Starting full dataset scoring..."
python convert_and_score_full_dataset.py

echo ""
echo "Job completed at: $(date)"
echo "Check output file: llama2_scored_full_dataset.jsonl" 