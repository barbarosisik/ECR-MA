#!/bin/bash
#SBATCH --partition=gpu-short
#SBATCH --gpus=1
#SBATCH --job-name=model_eval
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --output=/data1/s3905993/slurm_outputs/model_eval_%j.out
#SBATCH --error=/data1/s3905993/slurm_outputs/model_eval_%j.err

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

# Debug info
echo "=== ENVIRONMENT VERIFICATION ==="
echo "Python path: $(which python)"
echo "PyTorch version: $(python -c "import torch; print(torch.__version__)")"
echo "CUDA available: $(python -c "import torch; print(torch.cuda.is_available())")"
echo "Working dir: $(pwd)"

# Move to project directory
cd /data1/s3905993/ECR-main

# Validate critical files
check_file() {
    if [ ! -f "$1" ]; then
        echo "ERROR: Missing file $1"
        exit 1
    fi
}

check_file "src_emo/evaluate_conv.py"
check_file "src_emo/evaluate_rl.py"

echo "=== STARTING MODEL EVALUATION ==="

# Create evaluation directories
mkdir -p ./evaluation_results/baseline
mkdir -p ./evaluation_results/rl_enhanced

# Step 1: Evaluate baseline model (if it exists)
if [ -d "./models/baseline_ecr" ]; then
    echo "Evaluating baseline ECR model..."
    accelerate launch src_emo/evaluate_conv.py \
        --model_path ./models/baseline_ecr \
        --output_dir ./evaluation_results/baseline
    echo "Baseline evaluation completed"
else
    echo "WARNING: Baseline model not found, skipping baseline evaluation"
fi

# Step 2: Evaluate RL-enhanced model (if it exists)
if [ -d "./models/rl_enhanced_ecr" ]; then
    echo "Evaluating RL-enhanced ECR model..."
    accelerate launch src_emo/evaluate_rl.py \
        --model_path ./models/rl_enhanced_ecr \
        --use_rl_eval \
        --output_dir ./evaluation_results/rl_enhanced
    echo "RL-enhanced evaluation completed"
else
    echo "WARNING: RL-enhanced model not found, skipping RL evaluation"
fi

# Step 3: Generate comparison report
echo "Generating comparison report..."
python -c "
import json
import os
from pathlib import Path

def load_eval_results(path):
    results_file = Path(path) / 'evaluation_results.json'
    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)
    return {}

baseline_results = load_eval_results('./evaluation_results/baseline')
rl_results = load_eval_results('./evaluation_results/rl_enhanced')

print('\\n📊 ECR Model Comparison Results')
print('=' * 50)

metrics = ['bleu_1', 'bleu_2', 'distinct_1', 'distinct_2', 'item_ratio']
for metric in metrics:
    baseline_val = baseline_results.get(metric, 'N/A')
    rl_val = rl_results.get(metric, 'N/A')
    print(f'{metric.upper():<15}: Baseline={baseline_val:<10} | RL={rl_val:<10}')

if 'mean_reward' in rl_results:
    print(f'\\n🎯 RL-Specific Metrics:')
    print(f'Mean Reward: {rl_results.get(\"mean_reward\", \"N/A\")}')
    print(f'Critic Value: {rl_results.get(\"critic_value\", \"N/A\")}')

print('\\n✅ Evaluation completed successfully!')
"

echo "=== MODEL EVALUATION COMPLETED ==="
echo "Check results in: ./evaluation_results/" 