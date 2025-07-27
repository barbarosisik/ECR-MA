#!/bin/bash

# Monitor all scoring jobs
echo "=== ALL SCORING JOBS STATUS ==="
echo ""

# Show all jobs
echo "=== ACTIVE JOBS ==="
squeue -u $USER

echo ""
echo "=== RECENT COMPLETED JOBS ==="
sacct -u $USER --starttime=2025-07-18 --format=JobID,JobName,State,Elapsed,MaxRSS

echo ""
echo "=== OUTPUT FILES SUMMARY ==="
echo "Fast scoring outputs:"
ls -la llama2_scored_fast_merged_1_3_part_*.jsonl 2>/dev/null || echo "No fast scoring outputs yet"

echo ""
echo "Original scoring outputs:"
ls -la llama2_scored_merged_1_3_part_*.jsonl 2>/dev/null || echo "No original scoring outputs yet"

echo ""
echo "=== SLURM OUTPUTS ==="
ls -la slurm_outputs/ | grep -E "(llama2_score|4461009)" | tail -5

echo ""
echo "=== TO MONITOR SPECIFIC JOB ==="
echo "./track_jobs.sh <job_id>"
echo ""
echo "=== TO MONITOR ALL OUTPUTS ==="
echo "tail -f slurm_outputs/*.out" 