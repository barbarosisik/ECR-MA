#!/bin/bash
#SBATCH --job-name=ecr_reproduce_baseline_fast
#SBATCH --partition=gpu-medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=../slurm_outputs/ecr_reproduce_baseline_fast_%j.out
#SBATCH --error=../slurm_outputs/ecr_reproduce_baseline_fast_%j.err

# Load modules and activate conda
module purge
module load ALICE/default
module load Miniconda3/23.9.0-0
module load CUDA/12.1.1
source /easybuild/software/Miniconda3/23.9.0-0/etc/profile.d/conda.sh
conda activate /data1/s3905993/conda_envs/ecrhmas_fixed

# Set environment variables for local cache
export HF_HOME="/data1/s3905993/cache/huggingface"
export TRANSFORMERS_CACHE="/data1/s3905993/cache/huggingface"

# Set working directory
cd /data1/s3905993/ECR-main

set -e

# === 1. Emotional Semantic Fusion Subtask ===
echo "[ECR] Stage 1: Emotional Semantic Fusion (train_pre.py) - FAST"
cd src_emo
cp -r data/emo_data/* data/redial/
python data/redial/process.py
accelerate launch train_pre.py \
  --dataset redial \
  --num_train_epochs 1 \
  --gradient_accumulation_steps 1 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 8 \
  --num_warmup_steps 100 \
  --max_length 64 \
  --prompt_max_length 64 \
  --entity_max_length 8 \
  --learning_rate 5e-4 \
  --seed 42 \
  --nei_mer
if [ $? -ne 0 ]; then echo "[ECR] Stage 1 failed!"; exit 1; fi

# === 2. Emotion-aware Item Recommendation Training ===
echo "[ECR] Stage 2: Emotion-aware Item Recommendation (train_rec.py) - FAST"
cp -r data/emo_data/* data/redial/
python data/redial/process_mask.py
cp -r data/redial/* data/redial_gen/
python data/redial_gen/merge.py
accelerate launch train_rec.py \
  --dataset redial_gen \
  --n_prefix_rec 5 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --num_warmup_steps 50 \
  --context_max_length 64 \
  --prompt_max_length 64 \
  --entity_max_length 8 \
  --learning_rate 1e-4 \
  --seed 8 \
  --like_score 2.0 \
  --dislike_score 1.0 \
  --notsay_score 0.5 \
  --weighted_loss \
  --nei_mer \
  --use_sentiment
if [ $? -ne 0 ]; then echo "[ECR] Stage 2 failed!"; exit 1; fi

# === 3. Emotion-aligned Response Generation (train and infer) ===
echo "[ECR] Stage 3: Emotion-aligned Response Generation (train_emp.py, infer_emp.py) - FAST"
python data/redial_gen/merge_rec.py
python imdb_review_entity_filter.py
python data/redial/process_empthetic.py
accelerate launch train_emp.py \
  --dataset redial \
  --num_train_epochs 2 \
  --gradient_accumulation_steps 1 \
  --ignore_pad_token_for_loss \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 8 \
  --num_warmup_steps 200 \
  --context_max_length 64 \
  --resp_max_length 64 \
  --learning_rate 1e-4
if [ $? -ne 0 ]; then echo "[ECR] Stage 3 (train) failed!"; exit 1; fi

# Inference
accelerate launch infer_emp.py \
  --dataset redial_gen \
  --split test \
  --per_device_eval_batch_size 16 \
  --context_max_length 64 \
  --resp_max_length 64
if [ $? -ne 0 ]; then echo "[ECR] Stage 3 (infer) failed!"; exit 1; fi

echo "[ECR] FAST baseline reproduction pipeline completed successfully!" 