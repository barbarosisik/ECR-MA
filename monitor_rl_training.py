#!/usr/bin/env python3
"""
RL Training Monitor
Monitors the progress of RL training and provides real-time updates
"""

import os
import time
import subprocess
import json
from datetime import datetime

def get_job_status(job_id):
    """Get the status of a SLURM job"""
    try:
        result = subprocess.run(['squeue', '-j', str(job_id)], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:  # Job is still running
                return lines[1].split()[4]  # Status column
        return "COMPLETED"  # Job finished
    except:
        return "UNKNOWN"

def monitor_training(job_id, output_dir="/data1/s3905993/slurm_outputs"):
    """Monitor RL training progress"""
    print(f"🔍 Monitoring RL Training Job {job_id}")
    print("=" * 50)
    
    out_file = f"{output_dir}/rl_training_{job_id}.out"
    err_file = f"{output_dir}/rl_training_{job_id}.err"
    
    last_size = 0
    last_error_size = 0
    
    while True:
        # Check job status
        status = get_job_status(job_id)
        
        if status == "COMPLETED":
            print(f"✅ Job {job_id} completed!")
            break
        elif status == "FAILED":
            print(f"❌ Job {job_id} failed!")
            break
        elif status == "CANCELLED":
            print(f"⏹️ Job {job_id} was cancelled!")
            break
        
        # Check output file for new content
        if os.path.exists(out_file):
            current_size = os.path.getsize(out_file)
            if current_size > last_size:
                with open(out_file, 'r') as f:
                    f.seek(last_size)
                    new_content = f.read()
                    if new_content.strip():
                        print(f"\n📤 New output ({datetime.now().strftime('%H:%M:%S')}):")
                        print(new_content.strip())
                last_size = current_size
        
        # Check error file for new content
        if os.path.exists(err_file):
            current_error_size = os.path.getsize(err_file)
            if current_error_size > last_error_size:
                with open(err_file, 'r') as f:
                    f.seek(last_error_size)
                    new_error_content = f.read()
                    if new_error_content.strip():
                        print(f"\n⚠️ New errors ({datetime.now().strftime('%H:%M:%S')}):")
                        print(new_error_content.strip())
                last_error_size = current_error_size
        
        # Wait before next check
        time.sleep(10)
        
        # Print status indicator
        print(f"⏳ Job {job_id} is {status}... ({datetime.now().strftime('%H:%M:%S')})", end='\r')

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python monitor_rl_training.py <job_id>")
        print("Example: python monitor_rl_training.py 4429412")
        sys.exit(1)
    
    job_id = sys.argv[1]
    monitor_training(job_id) 