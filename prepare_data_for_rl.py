#!/usr/bin/env python3
"""
Data Preparation Script for ECR RL Training

This script copies and prepares the processed data from ECRHMAS for RL training in ECR-main.
"""

import os
import json
import shutil
import argparse
from pathlib import Path

def copy_data_files():
    """Copy processed data files from ECRHMAS to ECR-main."""
    
    # Source and destination paths
    source_dir = Path("../ECRHMAS/data/redial_gen")
    dest_dir = Path("./data/processed")
    
    # Create destination directory
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Files to copy
    files_to_copy = [
        "train_data_processed.jsonl",
        "valid_data_processed.jsonl", 
        "test_data_processed.jsonl",
        "train_data_dbpedia_emo.jsonl",
        "valid_data_dbpedia_emo.jsonl",
        "test_data_dbpedia_emo.jsonl"
    ]
    
    print("Copying data files from ECRHMAS to ECR-main...")
    
    for file_name in files_to_copy:
        source_file = source_dir / file_name
        dest_file = dest_dir / file_name
        
        if source_file.exists():
            shutil.copy2(source_file, dest_file)
            print(f"✓ Copied {file_name}")
        else:
            print(f"✗ File not found: {source_file}")
    
    # Copy additional files
    additional_files = [
        "movie_knowledge_base.json",
        "entity2id.json",
        "relation2id.json",
        "movie_ids.json",
        "movie_genres_full.json"
    ]
    
    for file_name in additional_files:
        source_file = source_dir / file_name
        dest_file = dest_dir / file_name
        
        if source_file.exists():
            shutil.copy2(source_file, dest_file)
            print(f"✓ Copied {file_name}")
        else:
            print(f"✗ File not found: {source_file}")

def create_critic_training_data():
    """Create training data specifically for critic pretraining."""
    
    print("Creating critic training data...")
    
    # Load processed data
    train_file = Path("./data/processed/train_data_processed.jsonl")
    valid_file = Path("./data/processed/valid_data_processed.jsonl")
    test_file = Path("./data/processed/test_data_processed.jsonl")
    
    if not train_file.exists():
        print("Error: Training data not found. Please run copy_data_files() first.")
        return
    
    # Create critic training data
    critic_train_file = Path("./data/critic_train.jsonl")
    critic_valid_file = Path("./data/critic_valid.jsonl")
    
    # Process training data for critic
    with open(train_file, 'r', encoding='utf-8') as f_in, \
         open(critic_train_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            data = json.loads(line.strip())
            
            # Extract context and response
            context = data.get('context', '')
            response = data.get('response', '')
            
            # Create critic training sample
            critic_sample = {
                'context': context,
                'response': response,
                'quality_label': 1.0  # Will be calculated by reward function during training
            }
            
            f_out.write(json.dumps(critic_sample, ensure_ascii=False) + '\n')
    
    # Process validation data for critic
    with open(valid_file, 'r', encoding='utf-8') as f_in, \
         open(critic_valid_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            data = json.loads(line.strip())
            
            context = data.get('context', '')
            response = data.get('response', '')
            
            critic_sample = {
                'context': context,
                'response': response,
                'quality_label': 1.0
            }
            
            f_out.write(json.dumps(critic_sample, ensure_ascii=False) + '\n')
    
    print(f"✓ Created critic training data: {critic_train_file}")
    print(f"✓ Created critic validation data: {critic_valid_file}")

def main():
    parser = argparse.ArgumentParser(description='Prepare data for ECR RL training')
    parser.add_argument('--copy-only', action='store_true',
                       help='Only copy data files, skip critic data creation')
    parser.add_argument('--critic-only', action='store_true',
                       help='Only create critic training data, skip file copying')
    
    args = parser.parse_args()
    
    if not args.critic_only:
        copy_data_files()
    
    if not args.copy_only:
        create_critic_training_data()
    
    print("\nData preparation completed!")
    print("\nNext steps:")
    print("1. Run critic pretraining:")
    print("   python src_emo/train_critic_supervised.py \\")
    print("       --train_data ./data/critic_train.jsonl \\")
    print("       --val_data ./data/critic_valid.jsonl \\")
    print("       --output_dir ./critic_pretrained")
    print("\n2. Run RL training:")
    print("   ./run_rl_training.sh")

if __name__ == '__main__':
    main() 