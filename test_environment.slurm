#!/bin/bash
#SBATCH --partition=gpu-short
#SBATCH --gpus=1
#SBATCH --job-name=test_env
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --output=/data1/s3905993/slurm_outputs/test_env_%j.out
#SBATCH --error=/data1/s3905993/slurm_outputs/test_env_%j.err

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
mkdir -p /data1/s3905993/slurm_outputs
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE"

# Activate conda environment
source activate /data1/s3905993/conda_envs/ecrhmas_fixed

# Ensure we're using the correct Python
export PATH="/data1/s3905993/conda_envs/ecrhmas_fixed/bin:$PATH"
export PYTHONPATH="/data1/s3905993/conda_envs/ecrhmas_fixed/lib/python3.10/site-packages:$PYTHONPATH"

# Debug info
echo "=== ENVIRONMENT VERIFICATION ==="
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "PyTorch version: $(python -c "import torch; print(torch.__version__)")"
echo "CUDA available: $(python -c "import torch; print(torch.cuda.is_available())")"
echo "Transformers available: $(python -c "import transformers; print('YES')" 2>/dev/null || echo 'NO')"
echo "Working dir: $(pwd)"
nvidia-smi

# Test the critic training script import
echo "=== TESTING CRITIC TRAINING SCRIPT ==="
cd /data1/s3905993/ECR-main
python -c "
import sys
print('Python path:', sys.path)
print('Testing imports...')
import torch
print('✓ PyTorch imported')
import transformers
print('✓ Transformers imported')
from src_emo.rl.critic import CriticAgent
print('✓ CriticAgent imported')
from src_emo.rl.reward_functions import RewardCalculator
print('✓ RewardCalculator imported')
print('All imports successful!')
"

echo "=== ENVIRONMENT TEST COMPLETED ===" 