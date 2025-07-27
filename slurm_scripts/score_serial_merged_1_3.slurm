#!/bin/bash

# To run: sbatch score_serial_merged_1_3.slurm <BATCH_NUM>
# Each batch processes two splits: batch 1 = parts 1+2, batch 2 = parts 3+4, ...

#SBATCH --partition=gpu-medium
#SBATCH --gpus=1
#SBATCH --job-name=llama2_score_serial_1_3
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_outputs/llama2_score_serial_1_3_%A_%x.out
#SBATCH --error=slurm_outputs/llama2_score_serial_1_3_%A_%x.err

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

# Get batch number from command line
BATCH_NUM=$1
if [ -z "$BATCH_NUM" ]; then
  echo "Error: Must provide batch number (1-5) as argument."
  exit 1
fi

PART1=$((2*BATCH_NUM-1))
PART2=$((2*BATCH_NUM))

for PART in $PART1 $PART2; do
  INPUT_FILE="src_emo/data/redial_gen/train_merged_1_3_part_${PART}.jsonl"
  OUTPUT_FILE="llama2_scored_merged_1_3_part_${PART}.jsonl"

  echo "=== ENVIRONMENT VERIFICATION ==="
  echo "Python path: $(which python)"
  echo "Python version: $(python --version)"
  echo "Conda env: $CONDA_DEFAULT_ENV"
  echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
  echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
  echo "Working dir: $(pwd)"
  nvidia-smi

  echo "=== LLAMA2 SERIAL SCORING JOB ==="
  echo "Job ID: $SLURM_JOB_ID"
  echo "Node: $SLURM_NODELIST"
  echo "Batch: $BATCH_NUM, Part: $PART"
  echo "Input file: $INPUT_FILE"
  echo "Output file: $OUTPUT_FILE"
  echo "Start time: $(date)"
  echo ""

  echo "Starting scoring for $INPUT_FILE..."
  python -u src_emo/scoring/llama2_score_responses.py --input $INPUT_FILE --output $OUTPUT_FILE

  echo ""
  echo "Finished $INPUT_FILE at: $(date)"
  echo "Check output file: $OUTPUT_FILE"
  echo "--------------------------------------"
done

echo "Batch $BATCH_NUM completed at: $(date)" 