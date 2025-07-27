#!/usr/bin/env python3
"""
Quick test of Mixtral8x7B scoring functionality.
"""

import os
import json
from src_emo.scoring.mixtral8x7b_score_responses_ultra_fast import load_model_and_tokenizer, score_response

# Set cache directories
os.environ['HF_HOME'] = "/data1/s3905993/cache/huggingface"
os.environ['TRANSFORMERS_CACHE'] = "/data1/s3905993/cache/transformers"

def test_mixtral_scoring():
    print("=== TESTING MIXTRAL8X7B SCORING ===")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Test cases
    test_cases = [
        {
            "context": ["Hello", "What kind of movies do you like?"],
            "response": "I love action movies, especially Marvel films!"
        },
        {
            "context": ["I'm feeling sad today"],
            "response": "I'm sorry to hear that. Would you like to talk about what's bothering you?"
        }
    ]
    
    print("\nTesting scoring on sample responses...")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Context: {test_case['context']}")
        print(f"Response: {test_case['response']}")
        
        scores = score_response(model, tokenizer, test_case['context'], test_case['response'])
        
        print("Scores:")
        for key, value in scores.items():
            print(f"  {key}: {value}")
    
    print("\n✅ Mixtral8x7B scoring test completed successfully!")

if __name__ == "__main__":
    test_mixtral_scoring() 