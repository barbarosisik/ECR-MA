#!/bin/bash

# To run on gpu-short: sbatch score_full_dataset_test.slurm

#SBATCH --partition=gpu-short
#SBATCH --gpus=1
#SBATCH --job-name=llama2_score_test
#SBATCH --ntasks=1
#SBATCH --time=00:20:00
#SBATCH --output=slurm_outputs/llama2_score_test_%A.out
#SBATCH --error=slurm_outputs/llama2_score_test_%A.err

set -e

module purge
module load ALICE/default
module load Miniconda3/23.9.0-0
module load CUDA/12.1.1

export HF_HOME="/data1/s3905993/cache/huggingface"
export TRANSFORMERS_CACHE="/data1/s3905993/cache/transformers"
mkdir -p slurm_outputs
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE"

source activate /data1/s3905993/conda_envs/ecrhmas_fixed
export PATH="/data1/s3905993/conda_envs/ecrhmas_fixed/bin:$PATH"
export PYTHONPATH="/data1/s3905993/conda_envs/ecrhmas_fixed/lib/python3.10/site-packages:$PYTHONPATH"

cd /data1/s3905993/ECR-main

echo "=== ENVIRONMENT VERIFICATION ==="
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "PyTorch version: $(python -c "import torch; print(torch.__version__)")"
echo "CUDA available: $(python -c "import torch; print(torch.cuda.is_available())")"
echo "Working dir: $(pwd)"
nvidia-smi

echo "=== LLAMA2 TEST DATASET SCORING JOB ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

echo "Starting test dataset scoring with llama2_score_responses.py..."
python -u src_emo/scoring/llama2_score_responses.py --input src_emo/data/redial_gen/train_data_processed.jsonl --output llama2_scored_testset.jsonl --max 20

echo ""
echo "Job completed at: $(date)"
echo "Check output file: llama2_scored_testset.jsonl" 