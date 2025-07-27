#!/bin/bash
#SBATCH --job-name=ecr_stage2_3_only
#SBATCH --partition=gpu-medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=../slurm_outputs/ecr_stage2_3_only_%j.out
#SBATCH --error=../slurm_outputs/ecr_stage2_3_only_%j.err

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

echo "[ECR] Stage 1 already completed successfully - proceeding to Stage 2 and 3"

# === 2. Emotion-aware Item Recommendation Training ===
echo "[ECR] Stage 2: Emotion-aware Item Recommendation (train_rec.py)"
cd src_emo
cp -r data/emo_data/* data/redial/
# Ensure we're in the correct directory for Stage 2
cd /data1/s3905993/ECR-main/src_emo/data/redial
/data1/s3905993/conda_envs/ecrhmas_fixed/bin/python process_mask.py
cp -r * ../redial_gen/
cd /data1/s3905993/ECR-main/src_emo/data/redial_gen
/data1/s3905993/conda_envs/ecrhmas_fixed/bin/python merge.py
cd /data1/s3905993/ECR-main/src_emo
/data1/s3905993/conda_envs/ecrhmas_fixed/bin/python train_rec.py \
  --dataset redial_gen \
  --n_prefix_rec 10 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --num_warmup_steps 25 \
  --context_max_length 64 \
  --prompt_max_length 64 \
  --entity_max_length 8 \
  --learning_rate 3e-4 \
  --seed 8 \
  --like_score 2.0 \
  --dislike_score 1.0 \
  --notsay_score 0.5 \
  --weighted_loss \
  --nei_mer \
  --use_sentiment
if [ $? -ne 0 ]; then echo "[ECR] Stage 2 failed!"; exit 1; fi

echo "[ECR] Stage 2 completed successfully!"

# === 3. Emotion-aligned Response Generation (train and infer) ===
echo "[ECR] Stage 3: Emotion-aligned Response Generation (train_emp.py, infer_emp.py)"
/data1/s3905993/conda_envs/ecrhmas_fixed/bin/python data/redial_gen/merge_rec.py
/data1/s3905993/conda_envs/ecrhmas_fixed/bin/python imdb_review_entity_filter.py
/data1/s3905993/conda_envs/ecrhmas_fixed/bin/python data/redial/process_empthetic.py
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

echo "[ECR] Stages 2 and 3 completed successfully! Full ECR baseline reproduction pipeline finished!" 