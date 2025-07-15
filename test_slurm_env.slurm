#!/bin/bash
#SBATCH --job-name=test_slurm_env
#SBATCH --output=../slurm_outputs/test_slurm_env_%j.out
#SBATCH --error=../slurm_outputs/test_slurm_env_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-short

# Load modules
module purge
module load ALICE/default
module load Miniconda3/23.9.0-0
module load CUDA/12.1.1

# Activate conda environment explicitly
source /easybuild/software/Miniconda3/23.9.0-0/etc/profile.d/conda.sh
conda activate /data1/s3905993/conda_envs/ecrhmas_fixed

# Debug environment
echo "=== ENVIRONMENT VERIFICATION ==="
echo "Python path: $(which python)"
python --version
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import transformers; print('Transformers version:', transformers.__version__)"
echo "Working directory: $(pwd)"
nvidia-smi

echo "=== TESTING MODEL LOADING ==="
python -c "
import os
import sys
sys.path.append('src_emo')

# Set cache directory
cache_dir = '/data1/s3905993/cache/huggingface'
os.environ['HF_HOME'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir

# Test model loading
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('roberta-base', cache_dir=cache_dir, local_files_only=True)
model = AutoModel.from_pretrained('roberta-base', cache_dir=cache_dir, local_files_only=True)
print('✅ RoBERTa loaded successfully')

from rl.critic import CriticAgent
print('✅ CriticAgent imported successfully')
"

echo "=== TEST COMPLETED SUCCESSFULLY ===" 