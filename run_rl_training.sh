#!/bin/bash

# RL-Enhanced Training Script for ECR-main
# This script runs the RL-enhanced training with PPO

set -e

# Configuration
DATASET="redial"
MODEL="microsoft/DialoGPT-small"
TOKENIZER="microsoft/DialoGPT-small"
OUTPUT_DIR="data/saved/emp_conv_rl"
LOG_DIR="logs"

# Create directories
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

# Set timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/rl_training_$TIMESTAMP.log"

echo "Starting RL-Enhanced Training at $(date)"
echo "Log file: $LOG_FILE"

# Run RL-enhanced training
accelerate launch src_emo/train_emp_rl.py \
    --dataset $DATASET \
    --model $MODEL \
    --tokenizer $TOKENIZER \
    --output_dir $OUTPUT_DIR \
    --context_max_length 150 \
    --resp_max_length 150 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 5 \
    --learning_rate 1e-5 \
    --num_warmup_steps 1000 \
    --ignore_pad_token_for_loss \
    --seed 42 \
    --use_rl \
    --rl_learning_rate 1e-5 \
    --rl_batch_size 8 \
    --rl_max_steps 5000 \
    --ppo_epochs 4 \
    --ppo_clip_epsilon 0.2 \
    --bleu_weight 1.0 \
    --distinct_weight 0.5 \
    --empathy_weight 2.0 \
    --recommendation_weight 1.5 \
    --use_wandb \
    --project "ecr-rl-enhanced" \
    --name "rl_training_$TIMESTAMP" \
    2>&1 | tee $LOG_FILE

echo "RL-Enhanced Training completed at $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo "Log saved to: $LOG_FILE" 