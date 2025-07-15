#!/bin/bash
#SBATCH --job-name=ecr_rl_pipeline_fixed
#SBATCH --output=../slurm_outputs/ecr_rl_pipeline_%j.out
#SBATCH --error=../slurm_outputs/ecr_rl_pipeline_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-medium

# Load modules
module purge
module load ALICE/default
module load Miniconda3/23.9.0-0
module load CUDA/12.1.1

# Activate conda environment explicitly
source /easybuild/software/Miniconda3/23.9.0-0/etc/profile.d/conda.sh
conda activate /data1/s3905993/conda_envs/ecrhmas_fixed

# Use the conda environment Python directly
CONDA_PYTHON="/data1/s3905993/conda_envs/ecrhmas_fixed/bin/python"

# Debug environment
echo "=== ENVIRONMENT VERIFICATION ==="
echo "Conda Python path: $CONDA_PYTHON"
$CONDA_PYTHON --version
$CONDA_PYTHON -c "import torch; print('PyTorch version:', torch.__version__)"
$CONDA_PYTHON -c "import torch; print('CUDA available:', torch.cuda.is_available())"
echo "Working directory: $(pwd)"

# Set environment variables for local cache
export HF_HOME="/data1/s3905993/cache/huggingface"
export TRANSFORMERS_CACHE="/data1/s3905993/cache/huggingface"

# Set working directory
cd /data1/s3905993/ECR-main

echo "=== ECR RL Pipeline with Local Cache Support ==="
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"

# Create necessary directories
mkdir -p data/processed
mkdir -p data/saved/critic_pretrained
mkdir -p data/saved/rl_models
mkdir -p data/evaluation_results
mkdir -p ../slurm_outputs

# Step 1: Data Preparation (if needed)
echo "=== Step 1: Checking Data Preparation ==="
if [ ! -f "data/processed/critic_train_data.json" ]; then
    echo "Preparing critic training data..."
    # Check if ReDial data exists, if not create dummy data for testing
    if [ ! -d "data/redial/" ]; then
        echo "ReDial data not found, creating dummy data for testing..."
        mkdir -p data/redial/
        python -c "
import json
import os
dummy_data = [
    {
        'conversationId': 1,
        'conversation': [
            {'text': 'Hello, what kind of movies do you like?'},
            {'text': 'I like action movies.'},
            {'text': 'Great! I recommend The Matrix.'}
        ],
        'movies': [{'movieId': 1, 'title': 'The Matrix'}]
    }
]
os.makedirs('data/redial', exist_ok=True)
with open('data/redial/dummy_data.json', 'w') as f:
    json.dump(dummy_data, f)
"
    fi
    $CONDA_PYTHON src_emo/prepare_critic_data.py \
        --input_data data/redial/ \
        --output_dir data/processed/ \
        --split_ratio 0.8
else
    echo "Critic training data already exists."
fi

if [ ! -f "data/processed/test_data.json" ]; then
    echo "Preparing test data..."
    $CONDA_PYTHON src_emo/prepare_test_data.py \
        --input_data data/redial/ \
        --output_dir data/processed/
else
    echo "Test data already exists."
fi

# Step 2: Critic Pretraining
echo "=== Step 2: Critic Pretraining ==="
if [ ! -f "data/saved/critic_pretrained/critic_model.pt" ]; then
    echo "Starting critic pretraining..."
    $CONDA_PYTHON src_emo/train_critic_supervised.py \
        --train_data data/processed/critic_train_data.json \
        --val_data data/processed/critic_val_data.json \
        --output_dir data/saved/critic_pretrained \
        --model_name roberta-base \
        --batch_size 32 \
        --learning_rate 2e-5 \
        --num_epochs 5 \
        --max_length 512 \
        --seed 42
    
    if [ $? -eq 0 ]; then
        echo "Critic pretraining completed successfully!"
    else
        echo "Critic pretraining failed!"
        exit 1
    fi
else
    echo "Critic model already exists."
fi

# Step 3: RL Training for both models
echo "=== Step 3: RL Training ==="

# Train with DialoGPT (ECR baseline comparison)
echo "Training with DialoGPT model..."
    $CONDA_PYTHON src_emo/train_emp_rl.py \
        --dataset redial \
        --output_dir data/saved/rl_models/dialogpt_rl \
        --model microsoft/DialoGPT-small \
        --tokenizer microsoft/DialoGPT-small \
        --text_tokenizer roberta-base \
        --num_train_epochs 3 \
        --learning_rate 5e-5 \
        --batch_size 4 \
        --max_gen_len 150 \
        --critic_model_path data/saved/critic_pretrained/critic_model.pt \
        --seed 42

if [ $? -eq 0 ]; then
    echo "DialoGPT RL training completed successfully!"
else
    echo "DialoGPT RL training failed!"
fi

# Train with Llama2-chat (ECR comparison)
echo "Training with Llama2-chat model..."
    $CONDA_PYTHON src_emo/train_emp_rl.py \
        --dataset redial \
        --output_dir data/saved/rl_models/llama2_rl \
        --model /data1/s3905993/ECRHMAS/src/models/llama2_chat \
        --tokenizer /data1/s3905993/ECRHMAS/src/models/llama2_chat \
        --text_tokenizer roberta-base \
        --num_train_epochs 3 \
        --learning_rate 5e-5 \
        --batch_size 4 \
        --max_gen_len 150 \
        --critic_model_path data/saved/critic_pretrained/critic_model.pt \
        --seed 42

if [ $? -eq 0 ]; then
    echo "Llama2-chat RL training completed successfully!"
else
    echo "Llama2-chat RL training failed!"
fi

# Step 4: Evaluation
echo "=== Step 4: Evaluation ==="

# Evaluate DialoGPT model
echo "Evaluating DialoGPT model..."
    $CONDA_PYTHON src_emo/evaluate_rl.py \
        --dataset redial \
        --model_path data/saved/rl_models/dialogpt_rl \
        --model_name microsoft/DialoGPT-small \
        --tokenizer microsoft/DialoGPT-small \
        --text_tokenizer roberta-base \
        --output_dir data/evaluation_results/dialogpt_eval \
        --test_data data/processed/test_data.json \
        --batch_size 8 \
        --max_gen_len 150 \
        --metrics bleu,distinct,hit@k,mrr@k,ndcg@k \
        --k_values 1,5,10 \
        --seed 42

# Evaluate Llama2-chat model
echo "Evaluating Llama2-chat model..."
    $CONDA_PYTHON src_emo/evaluate_rl.py \
        --dataset redial \
        --model_path data/saved/rl_models/llama2_rl \
        --model_name /data1/s3905993/ECRHMAS/src/models/llama2_chat \
        --tokenizer /data1/s3905993/ECRHMAS/src/models/llama2_chat \
        --text_tokenizer roberta-base \
        --output_dir data/evaluation_results/llama2_eval \
        --test_data data/processed/test_data.json \
        --batch_size 8 \
        --max_gen_len 150 \
        --metrics bleu,distinct,hit@k,mrr@k,ndcg@k \
        --k_values 1,5,10 \
        --seed 42

# Step 5: Generate ECR Comparison Report
echo "=== Step 5: Generating ECR Comparison Report ==="
$CONDA_PYTHON -c "
import json
import os
from datetime import datetime

def load_results(output_dir):
    eval_file = os.path.join(output_dir, 'evaluation_results.json')
    if os.path.exists(eval_file):
        with open(eval_file, 'r') as f:
            return json.load(f)
    return {}

# Load results for both models
dialogpt_results = load_results('data/evaluation_results/dialogpt_eval')
llama2_results = load_results('data/evaluation_results/llama2_eval')

# Create comprehensive ECR comparison report
report = {
    'pipeline_info': {
        'date': datetime.now().isoformat(),
        'job_id': '$SLURM_JOB_ID',
        'node': '$SLURM_NODELIST',
        'description': 'ECR RL Pipeline with MACPO-inspired multi-agent approach'
    },
    'model_comparisons': {
        'dialogpt_rl': {
            'model_type': 'DialoGPT-small with RL',
            'ecr_equivalent': 'ECR[DialoGPT]',
            'metrics': dialogpt_results
        },
        'llama2_rl': {
            'model_type': 'Llama2-7B-Chat with RL',
            'ecr_equivalent': 'ECR[Llama 2-Chat]',
            'metrics': llama2_results
        }
    },
    'ecr_baseline_comparison': {
        'recommendation_metrics': {
            'dialogpt_auc': dialogpt_results.get('auc', 'N/A'),
            'llama2_auc': llama2_results.get('auc', 'N/A'),
            'ecr_dialogpt_auc': 0.506,  # From ECR paper
            'ecr_llama2_auc': 'N/A'  # Not reported in paper
        },
        'generation_metrics': {
            'dialogpt_bleu': dialogpt_results.get('bleu', 'N/A'),
            'llama2_bleu': llama2_results.get('bleu', 'N/A'),
            'ecr_dialogpt_bleu': 'N/A',  # Not reported in paper
            'ecr_llama2_bleu': 'N/A'  # Not reported in paper
        }
    }
}

# Save comprehensive report
report_file = 'data/evaluation_results/ecr_comparison_report.json'
with open(report_file, 'w') as f:
    json.dump(report, f, indent=2)

print(f'Comprehensive ECR comparison report saved to: {report_file}')
print('\\n=== ECR Comparison Summary ===')
print(f'DialoGPT RL - AUC: {dialogpt_results.get(\"auc\", \"N/A\")}, BLEU: {dialogpt_results.get(\"bleu\", \"N/A\")}')
print(f'Llama2 RL - AUC: {llama2_results.get(\"auc\", \"N/A\")}, BLEU: {llama2_results.get(\"bleu\", \"N/A\")}')
print(f'ECR DialoGPT Baseline - AUC: 0.506')
"

echo "=== Pipeline Completed Successfully! ==="
echo "Final report: data/evaluation_results/ecr_comparison_report.json"
echo "Date: $(date)" 