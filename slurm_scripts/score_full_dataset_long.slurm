#!/bin/bash

# To run on gpu-long: sbatch score_full_dataset_long.slurm

#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=llama2_score_full_long
#SBATCH --ntasks=1
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_outputs/llama2_score_full_long_%A.out
#SBATCH --error=slurm_outputs/llama2_score_full_long_%A.err

set -e

module purge
module load ALICE/default
module load Miniconda3/23.9.0-0
module load CUDA/12.1.1

export HF_HOME="/data1/s3905993/cache/huggingface"
export TRANSFORMERS_CACHE="/data1/s3905993/cache/transformers"
mkdir -p slurm_outputs
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /data1/s3905993/conda_envs/ecrhmas_fixed
export PATH="/data1/s3905993/conda_envs/ecrhmas_fixed/bin:$PATH"
export PYTHONPATH="/data1/s3905993/conda_envs/ecrhmas_fixed/lib/python3.10/site-packages:$PYTHONPATH"

cd /data1/s3905993/ECR-main

echo "=== ENVIRONMENT VERIFICATION ==="
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo 'PyTorch version: ' $(python -c "import torch; print(torch.__version__)")
echo 'CUDA available: ' $(python -c "import torch; print(torch.cuda.is_available())")
echo "Working dir: $(pwd)"
nvidia-smi

echo "=== LLAMA2 FULL DATASET SCORING JOB (LONG) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

echo "Starting full dataset scoring with llama2_score_responses.py..."
python -u src_emo/scoring/llama2_score_responses.py --input src_emo/data/redial_gen/train_data_processed.jsonl --output llama2_scored_fullset_long.jsonl

echo ""
echo "Job completed at: $(date)"
echo "Check output file: llama2_scored_fullset_long.jsonl" 