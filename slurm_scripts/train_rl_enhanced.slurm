#!/bin/bash
#SBATCH --partition=gpu-medium
#SBATCH --gpus=1
#SBATCH --job-name=rl_training
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --output=/data1/s3905993/slurm_outputs/rl_training_%j.out
#SBATCH --error=/data1/s3905993/slurm_outputs/rl_training_%j.err

# === LATEST RUN: Model saving fix and checkpointing enabled ===
# This run is for 1 epoch to test model saving and checkpointing.

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
echo "Conda Python path: $(which python)"
echo "Python version: $(python --version)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "PyTorch version: $(python -c "import torch; print(torch.__version__)")"
echo "CUDA available: $(python -c "import torch; print(torch.cuda.is_available())")"
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

check_file "src_emo/train_emp_rl.py"
check_file "critic_pretrained_dual_model/critic_final.pth"

echo "=== STARTING RL-ENHANCED TRAINING WITH IMPROVED CRITIC ==="

# Create output directory
mkdir -p ./models/rl_enhanced_ecr_improved

# Run RL-enhanced training with improved critic
python src_emo/train_emp_rl.py \
    --dataset redial \
    --use_rl \
    --critic_pretrained_path ./critic_pretrained_dual_model/critic_final.pth \
    --context_max_length 150 \
    --resp_max_length 150 \
    --rl_learning_rate 1e-5 \
    --rl_batch_size 16 \
    --rl_max_steps 10000 \
    --bleu_weight 1.0 \
    --empathy_weight 2.0 \
    --recommendation_weight 1.5 \
    --distinct_weight 0.5 \
    --ppo_clip_epsilon 0.2 \
    --ppo_epochs 4 \
    --num_train_epochs 1 \
    --output_dir ./models/rl_enhanced_ecr_improved

# Note: --num_train_epochs 1 ensures only 1 epoch is run for this test.

echo "=== RL-ENHANCED TRAINING COMPLETED ==="
echo "Check output in: ./models/rl_enhanced_ecr_improved/" 