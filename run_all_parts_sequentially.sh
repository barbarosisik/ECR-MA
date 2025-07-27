#!/bin/bash

# Usage: bash run_all_parts_sequentially.sh
# This script submits scoring jobs for parts 1-10 sequentially, waiting for each to finish.

for PART in {1..10}; do
  echo "Submitting part $PART..."
  JOBID=$(sbatch --parsable score_one_merged_1_3.slurm $PART)
  echo "Submitted job $JOBID for part $PART. Waiting for it to finish..."
  # Wait for the job to finish
  while squeue -j $JOBID | grep -q $JOBID; do
    sleep 60
  done
  echo "Job $JOBID for part $PART finished."
done

echo "All parts processed sequentially." 