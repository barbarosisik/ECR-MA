#!/bin/bash
#SBATCH --job-name=rl_training_fixed
#SBATCH --output=../slurm_outputs/rl_training_%j.out
#SBATCH --error=../slurm_outputs/rl_training_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-medium

# Load modules
module load cuda/11.8
module load anaconda3

# Activate conda environment
source /data1/s3905993/conda_envs/ecrhmas_fixed/bin/activate

# Set environment variables for local cache
export HF_HOME="/data1/s3905993/cache/huggingface"
export TRANSFORMERS_CACHE="/data1/s3905993/cache/huggingface"

# Set working directory
cd /data1/s3905993/ECR-main

# Create output directories
mkdir -p data/saved/rl_models
mkdir -p data/saved/rl_logs

# Model configuration - can be changed for different experiments
MODEL_TYPE="dialogpt"  # Options: "dialogpt" or "llama2"

if [ "$MODEL_TYPE" = "dialogpt" ]; then
    echo "Training with DialoGPT model (ECR baseline comparison)"
    MODEL_NAME="microsoft/DialoGPT-small"
    TOKENIZER_NAME="microsoft/DialoGPT-small"
    OUTPUT_DIR="data/saved/rl_models/dialogpt_rl"
elif [ "$MODEL_TYPE" = "llama2" ]; then
    echo "Training with Llama2-chat model (ECR comparison)"
    MODEL_NAME="/data1/s3905993/ECRHMAS/src/models/llama2_chat"
    TOKENIZER_NAME="/data1/s3905993/ECRHMAS/src/models/llama2_chat"
    OUTPUT_DIR="data/saved/rl_models/llama2_rl"
else
    echo "Invalid model type. Please choose 'dialogpt' or 'llama2'"
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
    --batch_size 4 \
    --max_gen_len 150 \
    --critic_model_path data/saved/critic_pretrained/critic_model.pt \
    --seed 42 \
    --debug

echo "RL training completed successfully for $MODEL_TYPE!" 