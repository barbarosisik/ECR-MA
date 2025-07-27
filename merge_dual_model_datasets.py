#!/usr/bin/env python3
"""
Merge Dual-Model Scored Datasets for Critic Training
Combines Llama2 and Mistral7B scored datasets into unified training format
"""

import json
import argparse
from typing import Dict, List, Any
from tqdm import tqdm
import os


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    """Save data to JSONL file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def merge_datasets(llama2_data: List[Dict], mistral7b_data: List[Dict]) -> List[Dict]:
    """
    Merge Llama2 and Mistral7B datasets into unified format
    
    Each output sample will contain:
    - Original conversation data
    - llama2_scores field
    - mistral7b_scores field
    """
    print(f"Merging {len(llama2_data)} Llama2 samples with {len(mistral7b_data)} Mistral7B samples...")
    
    # Create lookup dictionaries for quick matching
    llama2_lookup = {}
    for item in llama2_data:
        # Use context and response as key for matching
        key = (tuple(item.get('context', [])), item.get('resp', ''))
        llama2_lookup[key] = item
    
    mistral7b_lookup = {}
    for item in mistral7b_data:
        key = (tuple(item.get('context', [])), item.get('resp', ''))
        mistral7b_lookup[key] = item
    
    # Find common samples
    common_keys = set(llama2_lookup.keys()) & set(mistral7b_lookup.keys())
    print(f"Found {len(common_keys)} common samples between datasets")
    
    # Merge common samples
    merged_data = []
    for key in tqdm(common_keys, desc="Merging samples"):
        llama2_item = llama2_lookup[key]
        mistral7b_item = mistral7b_lookup[key]
        
        # Create merged sample
        merged_item = {
            'role': llama2_item.get('role'),
            'context': llama2_item.get('context'),
            'resp': llama2_item.get('resp'),
            'rec': llama2_item.get('rec'),
            'entity': llama2_item.get('entity'),
            'emotion_entity': llama2_item.get('emotion_entity'),
            'emotion_probs_entity': llama2_item.get('emotion_probs_entity'),
            'llama2_scores': llama2_item.get('llama2_scores'),
            'mistral7b_scores': mistral7b_item.get('mistral7b_scores')
        }
        
        merged_data.append(merged_item)
    
    return merged_data


def create_training_format(merged_data: List[Dict]) -> List[Dict]:
    """
    Convert merged data into training format for critic agent
    
    Training format:
    - context: conversation context
    - response: generated response
    - target_scores: combined scores from both models
    """
    training_data = []
    
    for item in tqdm(merged_data, desc="Creating training format"):
        # Combine context into single string
        context = " ".join(item.get('context', []))
        response = item.get('resp', '')
        
        # Get scores from both models
        llama2_scores = item.get('llama2_scores', {})
        mistral7b_scores = item.get('mistral7b_scores', {})
        
        # Create target scores (average of both models)
        target_scores = {}
        score_keys = ['empathy_score', 'informativeness_score', 'recommendation_score', 'engagement_score', 'overall_score']
        
        for key in score_keys:
            llama2_val = llama2_scores.get(key, 0.0)
            mistral7b_val = mistral7b_scores.get(key, 0.0)
            # Average the scores from both models
            target_scores[key] = (llama2_val + mistral7b_val) / 2.0
        
        training_item = {
            'context': context,
            'response': response,
            'target_scores': target_scores,
            'llama2_scores': llama2_scores,
            'mistral7b_scores': mistral7b_scores
        }
        
        training_data.append(training_item)
    
    return training_data


def main():
    parser = argparse.ArgumentParser(description="Merge dual-model scored datasets")
    parser.add_argument("--llama2_data", type=str, required=True, 
                       help="Path to Llama2 scored dataset")
    parser.add_argument("--mistral7b_data", type=str, required=True,
                       help="Path to Mistral7B scored dataset")
    parser.add_argument("--output_dir", type=str, default="src_emo/data/redial_gen/scored_datasets",
                       help="Output directory for merged datasets")
    parser.add_argument("--train_split", type=float, default=0.8,
                       help="Training split ratio")
    
    args = parser.parse_args()
    
    # Load datasets
    print("Loading Llama2 dataset...")
    llama2_data = load_jsonl(args.llama2_data)
    
    print("Loading Mistral7B dataset...")
    mistral7b_data = load_jsonl(args.mistral7b_data)
    
    # Merge datasets
    merged_data = merge_datasets(llama2_data, mistral7b_data)
    
    # Create training format
    training_data = create_training_format(merged_data)
    
    # Split into train/validation
    split_idx = int(len(training_data) * args.train_split)
    train_data = training_data[:split_idx]
    val_data = training_data[split_idx:]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save datasets
    train_path = os.path.join(args.output_dir, "critic_train_dual_model.jsonl")
    val_path = os.path.join(args.output_dir, "critic_val_dual_model.jsonl")
    full_path = os.path.join(args.output_dir, "critic_full_dual_model.jsonl")
    
    print(f"Saving training data ({len(train_data)} samples) to {train_path}")
    save_jsonl(train_data, train_path)
    
    print(f"Saving validation data ({len(val_data)} samples) to {val_path}")
    save_jsonl(val_data, val_path)
    
    print(f"Saving full dataset ({len(training_data)} samples) to {full_path}")
    save_jsonl(training_data, full_path)
    
    # Print statistics
    print("\n" + "="*50)
    print("DATASET MERGING COMPLETED")
    print("="*50)
    print(f"Llama2 samples: {len(llama2_data)}")
    print(f"Mistral7B samples: {len(mistral7b_data)}")
    print(f"Common samples: {len(merged_data)}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Total training samples: {len(training_data)}")
    
    # Sample score statistics
    if training_data:
        sample = training_data[0]
        print(f"\nSample target scores: {sample['target_scores']}")
        print(f"Sample Llama2 scores: {sample['llama2_scores']}")
        print(f"Sample Mistral7B scores: {sample['mistral7b_scores']}")
    
    print("\nReady for critic agent training!")


if __name__ == "__main__":
    main() 