#!/usr/bin/env python3
"""
Prepare critic training data from ReDial dataset
"""

import os
import json
import argparse
from typing import Dict, List, Tuple
import random

def load_redial_data(data_path: str) -> List[Dict]:
    """Load ReDial dataset"""
    data = []
    if os.path.isdir(data_path):
        # Load from directory structure
        for filename in os.listdir(data_path):
            if filename.endswith('.json'):
                with open(os.path.join(data_path, filename), 'r') as f:
                    data.extend(json.load(f))
    else:
        # Load from single file
        with open(data_path, 'r') as f:
            data = json.load(f)
    return data

def create_critic_training_data(data: List[Dict], split_ratio: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
    """Create critic training data with quality labels"""
    
    critic_data = []
    
    for conversation in data:
        # Extract dialogue turns
        dialogue_history = []
        for turn in conversation.get('conversation', []):
            if 'text' in turn:
                dialogue_history.append(turn['text'])
        
        # Create training examples
        for i in range(len(dialogue_history) - 1):
            context = ' '.join(dialogue_history[:i+1])
            response = dialogue_history[i+1]
            
            # Simple quality scoring based on response length and content
            # In a real implementation, you'd use more sophisticated metrics
            quality_score = min(1.0, len(response.split()) / 20.0)  # Normalize by expected length
            
            # Create quality labels (BLEU, Distinct, Empathy, Recommendation)
            quality_labels = {
                'bleu_score': quality_score,
                'distinct_score': min(1.0, len(set(response.split())) / len(response.split()) if response.split() else 0),
                'empathy_score': 0.5,  # Placeholder - would need emotion analysis
                'recommendation_score': 0.5  # Placeholder - would need recommendation analysis
            }
            
            critic_data.append({
                'context': context,
                'response': response,
                'quality_labels': quality_labels,
                'overall_score': sum(quality_labels.values()) / len(quality_labels)
            })
    
    # Split into train/val
    random.shuffle(critic_data)
    split_idx = int(len(critic_data) * split_ratio)
    
    train_data = critic_data[:split_idx]
    val_data = critic_data[split_idx:]
    
    return train_data, val_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, required=True, help='Path to ReDial data')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='Train/val split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading data from {args.input_data}...")
    data = load_redial_data(args.input_data)
    print(f"Loaded {len(data)} conversations")
    
    print("Creating critic training data...")
    train_data, val_data = create_critic_training_data(data, args.split_ratio)
    
    print(f"Created {len(train_data)} training examples and {len(val_data)} validation examples")
    
    # Save data
    train_file = os.path.join(args.output_dir, 'critic_train_data.json')
    val_file = os.path.join(args.output_dir, 'critic_val_data.json')
    
    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(val_file, 'w') as f:
        json.dump(val_data, f, indent=2)
    
    print(f"Saved training data to {train_file}")
    print(f"Saved validation data to {val_file}")

if __name__ == "__main__":
    main() 