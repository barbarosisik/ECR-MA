#!/bin/bash
#SBATCH --job-name=ecr_stage3_only
#SBATCH --partition=gpu-medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=../slurm_outputs/ecr_stage3_only_%j.out
#SBATCH --error=../slurm_outputs/ecr_stage3_only_%j.err

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

echo "[ECR] Stages 1 and 2 already completed successfully - proceeding to Stage 3 only"

# === 3. Emotion-aligned Response Generation (train and infer) ===
echo "[ECR] Stage 3: Emotion-aligned Response Generation (train_emp.py, infer_emp.py)"

# First, run merge_rec.py from the correct directory
cd /data1/s3905993/ECR-main/src_emo/data/redial_gen
/data1/s3905993/conda_envs/ecrhmas_fixed/bin/python merge_rec.py
if [ $? -ne 0 ]; then echo "[ECR] merge_rec.py failed!"; exit 1; fi

# Go back to src_emo directory for the rest of Stage 3
cd /data1/s3905993/ECR-main/src_emo
/data1/s3905993/conda_envs/ecrhmas_fixed/bin/python imdb_review_entity_filter.py
if [ $? -ne 0 ]; then echo "[ECR] imdb_review_entity_filter.py failed!"; exit 1; fi

/data1/s3905993/conda_envs/ecrhmas_fixed/bin/python data/redial/process_empthetic.py
if [ $? -ne 0 ]; then echo "[ECR] process_empthetic.py failed!"; exit 1; fi

/data1/s3905993/conda_envs/ecrhmas_fixed/bin/python train_emp.py \
  --dataset redial \
  --num_train_epochs 1 \
  --gradient_accumulation_steps 1 \
  --ignore_pad_token_for_loss \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 32 \
  --num_warmup_steps 25 \
  --context_max_length 32 \
  --resp_max_length 32 \
  --learning_rate 5e-4
if [ $? -ne 0 ]; then echo "[ECR] Stage 3 (train) failed!"; exit 1; fi

# Inference
/data1/s3905993/conda_envs/ecrhmas_fixed/bin/python infer_emp.py \
  --dataset redial_gen \
  --split test \
  --per_device_eval_batch_size 64 \
  --context_max_length 32 \
  --resp_max_length 32
if [ $? -ne 0 ]; then echo "[ECR] Stage 3 (infer) failed!"; exit 1; fi

echo "[ECR] Stage 3 completed successfully! Full ECR baseline reproduction pipeline finished!" 