#!/bin/bash
#SBATCH --job-name=critic_pretrain_fixed
#SBATCH --output=../slurm_outputs/critic_pretrain_%j.out
#SBATCH --error=../slurm_outputs/critic_pretrain_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-short

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

# Create output directory
mkdir -p data/saved/critic_pretrained

# Run critic pretraining with local cache
python src_emo/train_critic_supervised.py \
    --train_data data/processed/critic_train_data.json \
    --val_data data/processed/critic_val_data.json \
    --output_dir data/saved/critic_pretrained \
    --model_name roberta-base \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --num_epochs 5 \
    --max_length 512 \
    --seed 42

echo "Critic pretraining completed successfully!" 