#!/usr/bin/env python3
"""
Test script for critic training
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src_emo'))

from train_critic_supervised import CriticPretrainingDataset

def test_dataset():
    """Test the dataset loading"""
    print("Testing dataset loading...")
    
    # Test with a small subset
    dataset = CriticPretrainingDataset('data/critic_train.jsonl', max_length=512)
    print(f"Dataset size: {len(dataset)}")
    
    # Test first item
    first_item = dataset[0]
    print(f"First item keys: {list(first_item.keys())}")
    print(f"Context: {first_item['context'][:100]}...")
    print(f"Response: {first_item['response'][:100]}...")
    print(f"Quality scores: {first_item['quality_scores']}")
    print(f"Overall score: {first_item['overall_score']}")
    
    print("Dataset test passed!")

if __name__ == '__main__':
    test_dataset() 