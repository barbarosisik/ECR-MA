#!/bin/bash
#SBATCH --job-name=ecr_reproduce_baseline_2gpu
#SBATCH --partition=gpu-medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=../slurm_outputs/ecr_reproduce_baseline_2gpu_%j.out
#SBATCH --error=../slurm_outputs/ecr_reproduce_baseline_2gpu_%j.err

# Load modules and activate conda
module purge
module load ALICE/default
module load Miniconda3/23.9.0-0
module load CUDA/12.1.1

# Activate conda environment properly
source /easybuild/software/Miniconda3/23.9.0-0/etc/profile.d/conda.sh
conda activate /data1/s3905993/conda_envs/ecrhmas_fixed

# Verify environment
which python
python --version
pip list | grep numpy

# Set environment variables for local cache
export HF_HOME="/data1/s3905993/cache/huggingface"
export TRANSFORMERS_CACHE="/data1/s3905993/cache/huggingface"

# Set CUDA environment variables
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

# Set working directory
cd /data1/s3905993/ECR-main

set -e

# === 1. Emotional Semantic Fusion Subtask ===
echo "[ECR] Stage 1: Emotional Semantic Fusion (train_pre.py) - Single GPU Test"
cd src_emo
cp -r data/emo_data/* data/redial/
/data1/s3905993/conda_envs/ecrhmas_fixed/bin/python data/redial/process.py
/data1/s3905993/conda_envs/ecrhmas_fixed/bin/python train_pre.py \
  --dataset redial \
  --num_train_epochs 1 \
  --gradient_accumulation_steps 8 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 4 \
  --num_warmup_steps 200 \
  --max_length 128 \
  --prompt_max_length 128 \
  --entity_max_length 16 \
  --learning_rate 1e-5 \
  --max_grad_norm 1.0 \
  --seed 42 \
  --nei_mer \
  --fp16
if [ $? -ne 0 ]; then echo "[ECR] Stage 1 failed!"; exit 1; fi

echo "[ECR] Stage 1 completed successfully!"

# === 2. Emotion-aware Item Recommendation Training ===
echo "[ECR] Stage 2: Emotion-aware Item Recommendation (train_rec.py)"
cp -r data/emo_data/* data/redial/
# Ensure we're in the correct directory for Stage 2
cd /data1/s3905993/ECR-main/src_emo
/data1/s3905993/conda_envs/ecrhmas_fixed/bin/python data/redial/process_mask.py
cp -r data/redial/* data/redial_gen/
/data1/s3905993/conda_envs/ecrhmas_fixed/bin/python data/redial_gen/merge.py
/data1/s3905993/conda_envs/ecrhmas_fixed/bin/python train_rec.py \
  --dataset redial_gen \
  --n_prefix_rec 10 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_warmup_steps 100 \
  --context_max_length 128 \
  --prompt_max_length 128 \
  --entity_max_length 16 \
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
echo "[ECR] Stage 3: Emotion-aligned Response Generation (train_emp.py, infer_emp.py)"
/data1/s3905993/conda_envs/ecrhmas_fixed/bin/python data/redial_gen/merge_rec.py
/data1/s3905993/conda_envs/ecrhmas_fixed/bin/python imdb_review_entity_filter.py
/data1/s3905993/conda_envs/ecrhmas_fixed/bin/python data/redial/process_empthetic.py
/data1/s3905993/conda_envs/ecrhmas_fixed/bin/python train_emp.py \
  --dataset redial \
  --num_train_epochs 1 \
  --gradient_accumulation_steps 4 \
  --ignore_pad_token_for_loss \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 4 \
  --num_warmup_steps 200 \
  --context_max_length 128 \
  --resp_max_length 128 \
  --learning_rate 1e-4
if [ $? -ne 0 ]; then echo "[ECR] Stage 3 (train) failed!"; exit 1; fi

# Inference
/data1/s3905993/conda_envs/ecrhmas_fixed/bin/python infer_emp.py \
  --dataset redial_gen \
  --split test \
  --per_device_eval_batch_size 8 \
  --context_max_length 128 \
  --resp_max_length 128
if [ $? -ne 0 ]; then echo "[ECR] Stage 3 (infer) failed!"; exit 1; fi

echo "[ECR] Full ECR baseline reproduction pipeline completed successfully!" 