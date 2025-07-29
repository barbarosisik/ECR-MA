#!/bin/bash
#SBATCH --job-name=rl_llama2_full
#SBATCH --output=../slurm_outputs/rl_llama2_full_%j.out
#SBATCH --error=../slurm_outputs/rl_llama2_full_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-medium

module purge
module load ALICE/default
module load Miniconda3/23.9.0-0
module load CUDA/12.1.1

# Activate conda environment explicitly
source /easybuild/software/Miniconda3/23.9.0-0/etc/profile.d/conda.sh
conda activate /data1/s3905993/conda_envs/ecrhmas_fixed

# Alternative activation if the above fails
if ! python -c "import numpy" 2>/dev/null; then
    echo "Trying alternative conda activation..."
    export PATH="/data1/s3905993/conda_envs/ecrhmas_fixed/bin:$PATH"
    export CONDA_DEFAULT_ENV="/data1/s3905993/conda_envs/ecrhmas_fixed"
fi

# Verify environment activation
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "Numpy available: $(python -c 'import numpy; print("YES")' 2>/dev/null || echo "NO")"

# Exit on error
set -e

# Set environment variables for local cache
export HF_HOME="/data1/s3905993/cache/huggingface"
export TRANSFORMERS_CACHE="/data1/s3905993/cache/huggingface"

# Set working directory
cd /data1/s3905993/ECR-main

# Create output directories
mkdir -p models/rl_enhanced_llama2_optimized
mkdir -p slurm_outputs
mkdir -p data/saved/rl_logs

echo "=== RL Training Configuration ==="
echo "Model: Llama2-7B-Chat with LoRA"
echo "Dataset: Full Redial dataset"
echo "Critic: Pretrained dual-model critic"
echo "Output: models/rl_enhanced_llama2_optimized"
echo "=================================="

# Check if required files exist
echo "Checking required files..."

# Check base model
if [ ! -d "/data1/s3905993/ECRHMAS/src/models/llama2_chat" ]; then
    echo "ERROR: Base Llama2 model not found at /data1/s3905993/ECRHMAS/src/models/llama2_chat"
    exit 1
fi

# Check LoRA model
if [ ! -d "/data1/s3905993/ECRHMAS/models/llama2_finetuned_movie_lora_cpu" ]; then
    echo "ERROR: LoRA model not found at /data1/s3905993/ECRHMAS/models/llama2_finetuned_movie_lora_cpu"
    exit 1
fi

# Check critic model
if [ ! -f "critic_pretrained_dual_model/critic_final.pth" ]; then
    echo "ERROR: Critic model not found at critic_pretrained_dual_model/critic_final.pth"
    exit 1
fi

# Check training data
if [ ! -f "src_emo/data/redial/train_data_processed.jsonl" ]; then
    echo "ERROR: Training data not found at src_emo/data/redial/train_data_processed.jsonl"
    exit 1
fi

if [ ! -f "src_emo/data/redial/valid_data_processed.jsonl" ]; then
    echo "ERROR: Validation data not found at src_emo/data/redial/valid_data_processed.jsonl"
    exit 1
fi

echo "All required files found. Starting RL training..."

# Run RL training with optimized script for 24-hour limit
python src_emo/train_llama2_rl_optimized.py \
    --output_dir models/rl_enhanced_llama2_optimized \
    --base_model /data1/s3905993/ECRHMAS/src/models/llama2_chat \
    --lora_model /data1/s3905993/ECRHMAS/models/llama2_finetuned_movie_lora_cpu \
    --train_file src_emo/data/redial/train_data_processed.jsonl \
    --val_file src_emo/data/redial/valid_data_processed.jsonl \
    --max_train_samples 20000 \
    --max_val_samples 2000 \
    --num_train_epochs 2 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --max_gen_len 150 \
    --context_max_length 512 \
    --critic_pretrained_path critic_pretrained_dual_model/critic_final.pth \
    --use_rl \
    --rl_learning_rate 1e-5 \
    --seed 42 \
    --fp16 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --logging_steps 200 \
    --max_grad_norm 1.0 \
    --weight_decay 0.01

echo "RL training completed successfully!"
echo "Model saved to: models/rl_enhanced_llama2_optimized"
echo "Logs saved to: slurm_outputs/rl_llama2_full_${SLURM_JOB_ID}.out"

# Optional: Run quick evaluation after training
echo "Running post-training evaluation..."
python src_emo/evaluate_ecr_proper.py \
    --model_path models/rl_enhanced_llama2_full \
    --base_model /data1/s3905993/ECRHMAS/src/models/llama2_chat \
    --test_file src_emo/data/redial/test_data_processed.jsonl \
    --max_samples 50 \
    --output_file results/rl_enhanced_optimized_evaluation.json \
    --do_sample \
    --temperature 0.7 \
    --top_p 0.9

echo "Post-training evaluation completed!"
echo "Results saved to: results/rl_enhanced_optimized_evaluation.json" 