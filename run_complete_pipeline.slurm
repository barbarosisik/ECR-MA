#!/bin/bash
#SBATCH --partition=gpu-medium
#SBATCH --gpus=1
#SBATCH --job-name=ecr_rl_pipeline
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --output=/data1/s3905993/slurm_outputs/ecr_rl_pipeline_%j.out
#SBATCH --error=/data1/s3905993/slurm_outputs/ecr_rl_pipeline_%j.err

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

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Validate critical files
check_file() {
    if [ ! -f "$1" ]; then
        print_error "Missing file $1"
        exit 1
    fi
}

check_file "src_emo/train_critic_supervised.py"
check_file "src_emo/train_emp_rl.py"
check_file "data/critic_train.jsonl"
check_file "data/critic_valid.jsonl"

echo "🚀 Starting Complete ECR RL Training Pipeline"
echo "=============================================="

# Step 1: Supervised Critic Pretraining
print_status "Step 1: Starting supervised critic pretraining..."
python src_emo/train_critic_supervised.py \
    --train_data ./data/critic_train.jsonl \
    --val_data ./data/critic_valid.jsonl \
    --output_dir ./critic_pretrained \
    --model_name roberta-base \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --num_epochs 5 \
    --max_length 512

if [ $? -eq 0 ]; then
    print_success "Critic pretraining completed"
else
    print_error "Critic pretraining failed"
    exit 1
fi

# Step 2: Baseline Model Training (if not already done)
if [ ! -d "./models/baseline_ecr" ]; then
    print_status "Step 2: Training baseline ECR model..."
    accelerate launch src_emo/train_emp.py \
        --dataset redial \
        --context_max_length 150 \
        --resp_max_length 150 \
        --per_device_train_batch_size 4 \
        --num_train_epochs 5 \
        --learning_rate 1e-5 \
        --output_dir ./models/baseline_ecr
    
    if [ $? -eq 0 ]; then
        print_success "Baseline ECR training completed"
    else
        print_error "Baseline ECR training failed"
        exit 1
    fi
else
    print_warning "Baseline model already exists, skipping baseline training"
fi

# Step 3: RL-Enhanced Training
print_status "Step 3: Starting RL-enhanced training..."
mkdir -p ./models/rl_enhanced_ecr

accelerate launch src_emo/train_emp_rl.py \
    --dataset redial \
    --use_rl \
    --critic_pretrained_path ./critic_pretrained/critic_pretrained_final.pth \
    --rl_learning_rate 1e-5 \
    --rl_batch_size 8 \
    --rl_max_steps 5000 \
    --bleu_weight 1.0 \
    --empathy_weight 2.0 \
    --recommendation_weight 1.5 \
    --distinct_weight 0.5 \
    --ppo_clip_epsilon 0.2 \
    --ppo_epochs 4 \
    --output_dir ./models/rl_enhanced_ecr

if [ $? -eq 0 ]; then
    print_success "RL-enhanced training completed"
else
    print_error "RL-enhanced training failed"
    exit 1
fi

# Step 4: Evaluation and Comparison
print_status "Step 4: Running comprehensive evaluation..."

# Create evaluation directories
mkdir -p ./evaluation_results/baseline
mkdir -p ./evaluation_results/rl_enhanced

# Evaluate baseline model
print_status "Evaluating baseline ECR model..."
accelerate launch src_emo/evaluate_conv.py \
    --model_path ./models/baseline_ecr \
    --output_dir ./evaluation_results/baseline

# Evaluate RL-enhanced model
print_status "Evaluating RL-enhanced ECR model..."
accelerate launch src_emo/evaluate_rl.py \
    --model_path ./models/rl_enhanced_ecr \
    --use_rl_eval \
    --output_dir ./evaluation_results/rl_enhanced

# Step 5: Generate Comparison Report
print_status "Step 5: Generating comparison report..."

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

print('\\n✅ Training pipeline completed successfully!')
"

print_success "Complete RL training pipeline finished!"
echo ""
echo "📁 Generated files and directories:"
echo "  - ./critic_pretrained/          # Pretrained critic model"
echo "  - ./models/baseline_ecr/        # Baseline ECR model"
echo "  - ./models/rl_enhanced_ecr/     # RL-enhanced ECR model"
echo "  - ./evaluation_results/         # Evaluation results"
echo ""
echo "🎯 Next steps:"
echo "  1. Review evaluation results in ./evaluation_results/"
echo "  2. Compare baseline vs RL-enhanced performance"
echo "  3. Optionally extend to multi-agent PPO (MAPPO/MACPO)" 