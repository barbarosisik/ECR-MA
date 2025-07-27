#!/bin/bash
#SBATCH --job-name=rl_training_fixed
#SBATCH --output=../slurm_outputs/rl_training_%j.out
#SBATCH --error=../slurm_outputs/rl_training_%j.err
#SBATCH --time=12:00:00
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

# Exit on error
set -e

# Set environment variables for local cache
export HF_HOME="/data1/s3905993/cache/huggingface"
export TRANSFORMERS_CACHE="/data1/s3905993/cache/huggingface"

# Set working directory
cd /data1/s3905993/ECR-main

# Create output directories
mkdir -p data/saved/rl_models
mkdir -p data/saved/rl_logs

# Model configuration - can be changed for different experiments
MODEL_TYPE="llama2"  # Only using Llama2 for this run

#if [ "$MODEL_TYPE" = "dialogpt" ]; then
#    echo "Training with DialoGPT model (ECR baseline comparison)"
#    MODEL_NAME="microsoft/DialoGPT-small"
#    TOKENIZER_NAME="microsoft/DialoGPT-small"
#    OUTPUT_DIR="data/saved/rl_models/dialogpt_rl"
if [ "$MODEL_TYPE" = "llama2" ]; then
    echo "Training with Llama2-chat model (ECR comparison)"
    MODEL_NAME="/data1/s3905993/ECR-main/models/llama2_finetuned_movie_lora_cpu"
    TOKENIZER_NAME="/data1/s3905993/ECR-main/models/llama2_finetuned_movie_lora_cpu"
    OUTPUT_DIR="models/rl_enhanced_ecr"
else
    echo "Invalid model type. Please choose 'llama2'"
    exit 1
fi

# Run RL training with local cache
python src_emo/train_emp_rl.py \
    --dataset redial \
    --output_dir $OUTPUT_DIR \
    --model $MODEL_NAME \
    --tokenizer $TOKENIZER_NAME \
    --text_tokenizer roberta-base \
    --num_train_epochs 3 \
    --learning_rate 5e-5 \
    --per_device_train_batch_size 4 \
    --max_gen_len 150 \
    --critic_pretrained_path models/critic_pretrained_dual_model/critic_final.pth \
    --use_rl \
    --seed 42 \
    --debug

echo "RL training completed successfully for $MODEL_TYPE!" 