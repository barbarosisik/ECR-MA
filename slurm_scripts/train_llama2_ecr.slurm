#!/bin/bash
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100:1   # Request 1 A100 GPU
#SBATCH --mem=100GB
#SBATCH --job-name=train_llama2_ecr
#SBATCH --ntasks=1
#SBATCH --time=3:59:00
#SBATCH --output=/data1/s3905993/slurm_outputs/train_llama2_ecr_%j.out
#SBATCH --error=/data1/s3905993/slurm_outputs/train_llama2_ecr_%j.err

set -e

module purge
module load ALICE/default
module load Miniconda3/23.9.0-0
module load CUDA/12.1.1

export HF_HOME="/data1/s3905993/cache/huggingface"
export TRANSFORMERS_CACHE="/data1/s3905993/cache/transformers"
mkdir -p /data1/s3905993/slurm_outputs
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE"

source activate /data1/s3905993/conda_envs/ecrhmas_fixed
export PATH="/data1/s3905993/conda_envs/ecrhmas_fixed/bin:$PATH"
export PYTHONPATH="/data1/s3905993/conda_envs/ecrhmas_fixed/lib/python3.10/site-packages:$PYTHONPATH"

cd /data1/s3905993/ECR-main

# Debug info
echo "=== ENVIRONMENT VERIFICATION ==="
echo "Conda Python path: $(which python)"
echo "Python version: $(python --version)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Working dir: $(pwd)"
nvidia-smi

# Run training
python train_llama2_ecr.py

echo "=== Llama2-Chat ECR Training Completed ===" 