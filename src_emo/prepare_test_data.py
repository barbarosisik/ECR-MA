#!/usr/bin/env python3
"""
Prepare test data from ReDial dataset
"""

import os
import json
import argparse
from typing import Dict, List

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

def create_test_data(data: List[Dict]) -> List[Dict]:
    """Create test data for evaluation"""
    
    test_data = []
    
    for conversation in data:
        # Extract dialogue turns
        dialogue_history = []
        for turn in conversation.get('conversation', []):
            if 'text' in turn:
                dialogue_history.append(turn['text'])
        
        # Create test examples (use last few turns for testing)
        if len(dialogue_history) >= 2:
            # Use the last turn as the target response
            context = ' '.join(dialogue_history[:-1])
            target_response = dialogue_history[-1]
            
            # Extract movie recommendations if available
            movies = conversation.get('movies', [])
            movie_ids = [movie.get('movieId') for movie in movies if movie.get('movieId')]
            
            test_data.append({
                'conversation_id': conversation.get('conversationId', len(test_data)),
                'context': context,
                'target_response': target_response,
                'movie_ids': movie_ids,
                'movie_names': [movie.get('title', '') for movie in movies],
                'user_rating': conversation.get('user_rating', 0)
            })
    
    return test_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, required=True, help='Path to ReDial data')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading data from {args.input_data}...")
    data = load_redial_data(args.input_data)
    print(f"Loaded {len(data)} conversations")
    
    print("Creating test data...")
    test_data = create_test_data(data)
    
    print(f"Created {len(test_data)} test examples")
    
    # Save data
    test_file = os.path.join(args.output_dir, 'test_data.json')
    
    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Saved test data to {test_file}")

if __name__ == "__main__":
    main() 