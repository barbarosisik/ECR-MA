#!/bin/bash

# Job tracking script
# Usage: ./track_jobs.sh [job_id]

JOB_ID=${1:-4461009}

echo "=== JOB TRACKING SCRIPT ==="
echo "Job ID: $JOB_ID"
echo ""

# Check job status
echo "=== JOB STATUS ==="
squeue -j $JOB_ID 2>/dev/null || echo "Job not found or completed"

echo ""
echo "=== OUTPUT FILES ==="
ls -la slurm_outputs/ | grep $JOB_ID || echo "No output files found"

echo ""
echo "=== LATEST OUTPUT (last 20 lines) ==="
OUTPUT_FILE=$(ls slurm_outputs/*${JOB_ID}*.out 2>/dev/null | head -1)
if [ -n "$OUTPUT_FILE" ]; then
    tail -20 "$OUTPUT_FILE"
else
    echo "No output file found"
fi

echo ""
echo "=== LATEST ERRORS (last 10 lines) ==="
ERROR_FILE=$(ls slurm_outputs/*${JOB_ID}*.err 2>/dev/null | head -1)
if [ -n "$ERROR_FILE" ]; then
    tail -10 "$ERROR_FILE"
else
    echo "No error file found"
fi

echo ""
echo "=== OUTPUT FILE SIZE ==="
if [ -n "$OUTPUT_FILE" ]; then
    echo "Output file: $(du -h "$OUTPUT_FILE" | cut -f1)"
    echo "Lines in output: $(wc -l < "$OUTPUT_FILE")"
fi

echo ""
echo "=== TO CONTINUOUSLY MONITOR, RUN: ==="
echo "tail -f slurm_outputs/*${JOB_ID}*.out" 