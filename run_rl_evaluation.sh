#!/bin/bash

# RL-Enhanced Evaluation Script for ECR-main
# This script evaluates both supervised and RL-trained models

set -e

# Configuration
DATASET="redial"
MODEL_PATH="data/saved/emp_conv_rl"  # Path to trained model
OUTPUT_DIR="evaluation_results"
LOG_DIR="logs"

# Create directories
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

# Set timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/rl_evaluation_$TIMESTAMP.log"

echo "Starting RL-Enhanced Evaluation at $(date)"
echo "Log file: $LOG_FILE"

# Run RL-enhanced evaluation
accelerate launch src_emo/evaluate_rl.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --context_max_length 150 \
    --resp_max_length 150 \
    --per_device_eval_batch_size 8 \
    --split test \
    --max_gen_len 150 \
    --use_rl_eval \
    --bleu_weight 1.0 \
    --distinct_weight 0.5 \
    --empathy_weight 2.0 \
    --recommendation_weight 1.5 \
    --do_sample \
    --temperature 0.8 \
    --top_p 0.9 \
    --seed 42 \
    2>&1 | tee $LOG_FILE

echo "RL-Enhanced Evaluation completed at $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo "Log saved to: $LOG_FILE" 