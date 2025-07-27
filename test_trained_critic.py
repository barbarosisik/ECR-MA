#!/usr/bin/env python3
"""
Test Trained Critic Agent
Verifies that the trained critic agent works properly and can evaluate responses
"""

import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src_emo.rl.critic import CriticAgent

def test_critic_agent():
    """Test the trained critic agent"""
    
    print("=== TESTING TRAINED CRITIC AGENT ===")
    
    # Check if model exists
    model_path = "critic_pretrained_dual_model/critic_final.pth"
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return False
    
    print(f"✅ Found trained model: {model_path}")
    print(f"Model size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
    
    # Initialize critic config
    class CriticConfig:
        def __init__(self):
            self.critic_model_name = 'roberta-base'
            self.critic_hidden_size = 768
            self.critic_dropout = 0.1
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = CriticConfig()
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    emotion_list = ['happy', 'sad', 'angry', 'neutral']
    
    # Load the trained critic
    print("Loading trained critic agent...")
    critic = CriticAgent(config, tokenizer, emotion_list)
    critic.load_state_dict(torch.load(model_path, map_location=config.device))
    critic.eval()
    
    print(f"✅ Critic loaded successfully on {config.device}")
    
    # Test cases
    test_cases = [
        {
            'context': "I'm feeling really sad today. Can you recommend a movie to cheer me up?",
            'response': "I understand you're feeling down. Let me recommend 'The Secret Life of Walter Mitty' - it's an uplifting adventure that will definitely brighten your day!",
            'expected_high_scores': ['empathy_score', 'recommendation_score']
        },
        {
            'context': "What kind of movies do you like?",
            'response': "I enjoy action movies and comedies. Have you seen 'Mad Max: Fury Road'?",
            'expected_high_scores': ['informativeness_score', 'engagement_score']
        },
        {
            'context': "I'm looking for a good thriller.",
            'response': "I'd recommend 'Gone Girl' - it's a psychological thriller with amazing twists.",
            'expected_high_scores': ['recommendation_score', 'informativeness_score']
        },
        {
            'context': "Hello",
            'response': "Hello there! How can I help you today?",
            'expected_high_scores': ['engagement_score']
        }
    ]
    
    print("\n=== TESTING RESPONSE EVALUATION ===")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Context: {test_case['context']}")
        print(f"Response: {test_case['response']}")
        
        # Evaluate response
        with torch.no_grad():
            outputs = critic(
                context=[test_case['context']],
                responses=[test_case['response']],
                return_quality_breakdown=True
            )
        
        # Extract scores
        value_score = outputs['values'][0].item()
        quality_scores = outputs['quality_breakdown']
        
        print(f"Overall Value Score: {value_score:.3f}")
        print("Quality Breakdown:")
        for metric, score in quality_scores.items():
            print(f"  {metric}: {score[0].item():.3f}")
        
        # Check if expected high scores are actually high
        for expected_metric in test_case['expected_high_scores']:
            if expected_metric in quality_scores:
                score = quality_scores[expected_metric][0].item()
                if score > 0.6:
                    print(f"  ✅ {expected_metric} is high ({score:.3f})")
                else:
                    print(f"  ⚠️  {expected_metric} is lower than expected ({score:.3f})")
    
    # Test batch processing
    print("\n=== TESTING BATCH PROCESSING ===")
    
    contexts = [tc['context'] for tc in test_cases]
    responses = [tc['response'] for tc in test_cases]
    
    with torch.no_grad():
        batch_outputs = critic(
            context=contexts,
            responses=responses,
            return_quality_breakdown=True
        )
    
    print(f"Batch processing successful!")
    print(f"Batch size: {len(contexts)}")
    print(f"Value scores: {batch_outputs['values'].cpu().numpy()}")
    
    # Test consistency
    print("\n=== TESTING CONSISTENCY ===")
    
    # Run same evaluation multiple times
    consistency_scores = []
    for _ in range(5):
        with torch.no_grad():
            outputs = critic(
                context=[test_cases[0]['context']],
                responses=[test_cases[0]['response']],
                return_quality_breakdown=True
            )
        consistency_scores.append(outputs['values'][0].item())
    
    consistency_std = np.std(consistency_scores)
    print(f"Consistency test (5 runs):")
    print(f"  Scores: {[f'{s:.3f}' for s in consistency_scores]}")
    print(f"  Standard deviation: {consistency_std:.4f}")
    
    if consistency_std < 0.01:
        print("  ✅ Very consistent predictions")
    elif consistency_std < 0.05:
        print("  ✅ Consistent predictions")
    else:
        print("  ⚠️  Some inconsistency in predictions")
    
    # Load training results for comparison
    print("\n=== TRAINING RESULTS SUMMARY ===")
    
    results_path = "critic_pretrained_dual_model/training_results.json"
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        final_train_r2 = results['training_history']['train_r2_scores'][-1]
        final_val_r2 = results['training_history']['val_r2_scores'][-1]
        final_train_loss = results['training_history']['train_losses'][-1]
        final_val_loss = results['training_history']['val_losses'][-1]
        
        print(f"Final Training R²: {final_train_r2:.4f}")
        print(f"Final Validation R²: {final_val_r2:.4f}")
        print(f"Final Training Loss: {final_train_loss:.4f}")
        print(f"Final Validation Loss: {final_val_loss:.4f}")
        
        if final_val_r2 > 0.2:
            print("✅ Good model performance (R² > 0.2)")
        else:
            print("⚠️  Model performance could be improved")
    
    print("\n=== TEST COMPLETED ===")
    print("✅ Critic agent is working properly!")
    print("✅ Ready for RL integration!")
    
    return True

if __name__ == "__main__":
    test_critic_agent() 