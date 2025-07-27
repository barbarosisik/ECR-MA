#!/bin/bash
#SBATCH --partition=gpu-medium
#SBATCH --gpus=1
#SBATCH --job-name=critic_pretrain
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --output=/data1/s3905993/slurm_outputs/critic_pretrain_%j.out
#SBATCH --error=/data1/s3905993/slurm_outputs/critic_pretrain_%j.err

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

# Move to project directory
cd /data1/s3905993/ECR-main

# Validate critical files
check_file() {
    if [ ! -f "$1" ]; then
        echo "ERROR: Missing file $1"
        exit 1
    fi
}

check_file "src_emo/train_critic_supervised.py"
check_file "llama2_scored_rich.jsonl"

echo "=== STARTING CRITIC PRETRAINING ==="

# Run critic pretraining
# Run critic pretraining on rich-context, 6-head data
python src_emo/train_critic_supervised.py \
    --train_data llama2_scored_rich.jsonl \
    --val_data llama2_scored_rich.jsonl \
    --output_dir ./critic_pretrained_rich \
    --model_name roberta-base \
    --batch_size 4 \
    --learning_rate 2e-5 \
    --num_epochs 5 \
    --max_length 512

echo "=== CRITIC PRETRAINING COMPLETED ==="
+echo "Check output in: ./critic_pretrained_rich/"