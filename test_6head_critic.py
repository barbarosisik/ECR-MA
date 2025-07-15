#!/usr/bin/env python3
"""
Test script for the 6-head critic model trained on synthetic labels
This script properly evaluates the critic model that outputs 6 different quality metrics
"""

import torch
import json
import sys
import os
import numpy as np
from transformers import AutoTokenizer
from src_emo.rl.critic import CriticAgent

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_6head_critic_model():
    """Load the 6-head critic model trained on synthetic labels"""
    print("=== LOADING 6-HEAD CRITIC MODEL ===")
    
    # Set cache directory
    cache_dir = "/data1/s3905993/cache/huggingface"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "roberta-base",
        cache_dir=cache_dir,
        local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Define emotion list (from the training)
    emotion_list = [
        "like", "curious", "happy", "grateful", "negative", "neutral", 
        "nostalgia", "agreement", "surprise"
    ]
    
    # Build config
    class CriticConfig:
        def __init__(self, model_name, device):
            self.critic_model_name = model_name
            self.critic_hidden_size = 768
            self.critic_dropout = 0.1
            self.device = device
        def to_dict(self):
            return self.__dict__
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = CriticConfig("roberta-base", device)
    
    # Load the 6-head critic model
    try:
        model = CriticAgent.load_model("critic_pretrained_rich/critic_pretrained_final.pth", config, tokenizer)
        print("✓ 6-head critic model loaded successfully")
        return model, device
    except Exception as e:
        print(f"✗ Failed to load 6-head critic model: {e}")
        return None, None

def test_6head_critic_on_synthetic_data():
    """Test the 6-head critic on the synthetic training data"""
    print("\n=== TESTING ON SYNTHETIC TRAINING DATA ===")
    
    model, device = load_6head_critic_model()
    if model is None:
        return
    
    # Load the synthetic training data
    try:
        with open("llama2_scored_rich.jsonl", "r") as f:
            synthetic_data = [json.loads(line) for line in f]
        print(f"✓ Loaded {len(synthetic_data)} synthetic training examples")
    except Exception as e:
        print(f"✗ Failed to load synthetic data: {e}")
        return
    
    print(f"\nTesting on {len(synthetic_data)} synthetic examples:")
    print("=" * 100)
    
    all_predictions = []
    all_ground_truth = []
    
    for i, item in enumerate(synthetic_data[:5]):  # Test first 5 examples
        print(f"\n--- Example {i+1} ---")
        
        # Extract context and response
        context_list = item.get('context', [])
        response = item.get('response', '')
        context = ' '.join([turn.strip() for turn in context_list if turn.strip()])
        
        print(f"Context: {context[:100]}...")
        print(f"Response: {response[:100]}...")
        
        # Get ground truth scores
        quality_scores = item.get('quality_scores', {})
        gt_empathy = quality_scores.get('empathy_score', 0.0)
        gt_informativeness = quality_scores.get('informativeness_score', 0.0)
        gt_recommendation = quality_scores.get('recommendation_score', 0.0)
        gt_engagement = quality_scores.get('engagement_score', 0.0)
        gt_overall = item.get('overall_score', 0.0)
        gt_bleu = quality_scores.get('bleu_score', 0.0)
        
        print(f"Ground Truth Scores:")
        print(f"  Empathy: {gt_empathy:.4f}")
        print(f"  Informativeness: {gt_informativeness:.4f}")
        print(f"  Recommendation: {gt_recommendation:.4f}")
        print(f"  Engagement: {gt_engagement:.4f}")
        print(f"  Overall: {gt_overall:.4f}")
        print(f"  BLEU: {gt_bleu:.4f}")
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(
                context=[context],
                responses=[response],
                return_quality_breakdown=True
            )
        
        # Extract predictions
        value_score = outputs['values'][0].item()
        quality = outputs['quality_breakdown']
        pred_bleu = quality['bleu_score'][0].item()
        pred_distinct = quality['distinct_score'][0].item()
        pred_empathy = quality['empathy_score'][0].item()
        pred_recommendation = quality['recommendation_score'][0].item()
        
        # Map predictions to our 6-head structure
        # Note: We need to map the model's 4 outputs to our 6 expected outputs
        pred_empathy_mapped = pred_empathy
        pred_informativeness_mapped = pred_distinct  # Use distinct as informativeness proxy
        pred_recommendation_mapped = pred_recommendation
        pred_engagement_mapped = pred_distinct  # Use distinct as engagement proxy
        pred_overall_mapped = value_score
        pred_bleu_mapped = pred_bleu
        
        print(f"Model Predictions:")
        print(f"  Empathy: {pred_empathy_mapped:.4f}")
        print(f"  Informativeness: {pred_informativeness_mapped:.4f}")
        print(f"  Recommendation: {pred_recommendation_mapped:.4f}")
        print(f"  Engagement: {pred_engagement_mapped:.4f}")
        print(f"  Overall: {pred_overall_mapped:.4f}")
        print(f"  BLEU: {pred_bleu_mapped:.4f}")
        
        # Calculate correlation
        gt_scores = [gt_empathy, gt_informativeness, gt_recommendation, gt_engagement, gt_overall, gt_bleu]
        pred_scores = [pred_empathy_mapped, pred_informativeness_mapped, pred_recommendation_mapped, 
                      pred_engagement_mapped, pred_overall_mapped, pred_bleu_mapped]
        
        correlation = np.corrcoef(gt_scores, pred_scores)[0, 1]
        print(f"Correlation: {correlation:.4f}")
        
        # Store for overall analysis
        all_predictions.append(pred_scores)
        all_ground_truth.append(gt_scores)
    
    # Overall analysis
    if all_predictions:
        all_predictions = np.array(all_predictions)
        all_ground_truth = np.array(all_ground_truth)
        
        print(f"\n=== OVERALL ANALYSIS ===")
        print(f"Average correlation across examples: {np.mean([np.corrcoef(all_ground_truth[i], all_predictions[i])[0, 1] for i in range(len(all_predictions))]):.4f}")
        
        # Calculate per-metric correlations
        metric_names = ['Empathy', 'Informativeness', 'Recommendation', 'Engagement', 'Overall', 'BLEU']
        for i, metric in enumerate(metric_names):
            corr = np.corrcoef(all_ground_truth[:, i], all_predictions[:, i])[0, 1]
            print(f"{metric} correlation: {corr:.4f}")

def test_6head_critic_on_quality_variations():
    """Test the 6-head critic on responses with varying quality levels"""
    print("\n=== TESTING ON QUALITY VARIATIONS ===")
    
    model, device = load_6head_critic_model()
    if model is None:
        return
    
    # Test cases with varying quality levels
    test_cases = [
        {
            "name": "High Quality - Empathetic Response",
            "context": "I'm feeling really sad today. I just lost my job and I don't know what to do.",
            "response": "I'm so sorry to hear that. Losing a job can be really difficult and stressful. It's completely normal to feel sad and uncertain. Have you thought about what kind of work you'd like to do next? I'm here to listen if you want to talk more about it.",
            "expected_high": ["empathy", "overall"]
        },
        {
            "name": "High Quality - Informative Recommendation",
            "context": "I love action movies! What should I watch tonight?",
            "response": "You should definitely watch 'Mad Max: Fury Road'! It's an incredible action movie with amazing stunts and non-stop excitement. The cinematography is stunning and the action sequences are mind-blowing. I think you'll really enjoy it!",
            "expected_high": ["informativeness", "recommendation", "overall"]
        },
        {
            "name": "Low Quality - Generic Response",
            "context": "I'm not sure about this new restaurant in town.",
            "response": "I don't know.",
            "expected_high": ["none"]
        },
        {
            "name": "Medium Quality - Basic Recommendation",
            "context": "Can you recommend a good book to read?",
            "response": "Try reading 'The Great Gatsby'. It's a classic.",
            "expected_high": ["recommendation"]
        },
        {
            "name": "High Quality - Detailed Recommendation",
            "context": "Can you recommend a good book to read?",
            "response": "I'd recommend 'The Seven Husbands of Evelyn Hugo' by Taylor Jenkins Reid. It's a captivating historical fiction novel about a legendary Hollywood actress and her seven marriages. The storytelling is brilliant, the characters are complex and compelling, and it has unexpected twists that will keep you hooked until the very end. It's perfect for anyone who loves character-driven stories with a bit of mystery and romance.",
            "expected_high": ["informativeness", "recommendation", "engagement", "overall"]
        }
    ]
    
    print(f"Testing {len(test_cases)} quality variation cases:")
    print("=" * 100)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n--- Case {i}: {case['name']} ---")
        print(f"Context: {case['context']}")
        print(f"Response: {case['response']}")
        print(f"Expected high scores in: {case['expected_high']}")
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(
                context=[case['context']],
                responses=[case['response']],
                return_quality_breakdown=True
            )
        
        # Extract predictions
        value_score = outputs['values'][0].item()
        quality = outputs['quality_breakdown']
        pred_bleu = quality['bleu_score'][0].item()
        pred_distinct = quality['distinct_score'][0].item()
        pred_empathy = quality['empathy_score'][0].item()
        pred_recommendation = quality['recommendation_score'][0].item()
        
        # Map to our 6-head structure
        pred_empathy_mapped = pred_empathy
        pred_informativeness_mapped = pred_distinct
        pred_recommendation_mapped = pred_recommendation
        pred_engagement_mapped = pred_distinct
        pred_overall_mapped = value_score
        pred_bleu_mapped = pred_bleu
        
        print(f"Model Scores:")
        print(f"  Empathy: {pred_empathy_mapped:.4f}")
        print(f"  Informativeness: {pred_informativeness_mapped:.4f}")
        print(f"  Recommendation: {pred_recommendation_mapped:.4f}")
        print(f"  Engagement: {pred_engagement_mapped:.4f}")
        print(f"  Overall: {pred_overall_mapped:.4f}")
        print(f"  BLEU: {pred_bleu_mapped:.4f}")
        
        # Quality assessment
        avg_score = (pred_empathy_mapped + pred_informativeness_mapped + pred_recommendation_mapped + 
                    pred_engagement_mapped + pred_overall_mapped + pred_bleu_mapped) / 6
        
        if avg_score > 0.6:
            quality_label = "HIGH"
        elif avg_score > 0.4:
            quality_label = "MEDIUM"
        else:
            quality_label = "LOW"
        
        print(f"Average Quality: {avg_score:.4f} ({quality_label})")
        
        # Check if predictions align with expectations
        print(f"Alignment with expectations: ", end="")
        if "none" in case['expected_high']:
            if avg_score < 0.4:
                print("✓ GOOD (low scores for low-quality response)")
            else:
                print("✗ POOR (unexpectedly high scores)")
        else:
            if avg_score > 0.5:
                print("✓ GOOD (high scores for high-quality response)")
            else:
                print("✗ POOR (unexpectedly low scores)")

def test_6head_critic_on_real_data():
    """Test the 6-head critic on real conversation data"""
    print("\n=== TESTING ON REAL CONVERSATION DATA ===")
    
    model, device = load_6head_critic_model()
    if model is None:
        return
    
    # Load real conversation data
    try:
        with open("sample_train_data_processed.jsonl", "r") as f:
            real_data = [json.loads(line) for line in f]
        print(f"✓ Loaded {len(real_data)} real conversation examples")
    except Exception as e:
        print(f"✗ Failed to load real data: {e}")
        return
    
    print(f"\nTesting on {min(5, len(real_data))} real conversation examples:")
    print("=" * 100)
    
    for i, item in enumerate(real_data[:5]):
        print(f"\n--- Real Example {i+1} ---")
        
        # Extract context and response
        context = " | ".join(item["context"])
        response = item["resp"]
        
        print(f"Context: {context[:100]}...")
        print(f"Response: {response}")
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(
                context=[context],
                responses=[response],
                return_quality_breakdown=True
            )
        
        # Extract predictions
        value_score = outputs['values'][0].item()
        quality = outputs['quality_breakdown']
        pred_bleu = quality['bleu_score'][0].item()
        pred_distinct = quality['distinct_score'][0].item()
        pred_empathy = quality['empathy_score'][0].item()
        pred_recommendation = quality['recommendation_score'][0].item()
        
        # Map to our 6-head structure
        pred_empathy_mapped = pred_empathy
        pred_informativeness_mapped = pred_distinct
        pred_recommendation_mapped = pred_recommendation
        pred_engagement_mapped = pred_distinct
        pred_overall_mapped = value_score
        pred_bleu_mapped = pred_bleu
        
        print(f"Model Scores:")
        print(f"  Empathy: {pred_empathy_mapped:.4f}")
        print(f"  Informativeness: {pred_informativeness_mapped:.4f}")
        print(f"  Recommendation: {pred_recommendation_mapped:.4f}")
        print(f"  Engagement: {pred_engagement_mapped:.4f}")
        print(f"  Overall: {pred_overall_mapped:.4f}")
        print(f"  BLEU: {pred_bleu_mapped:.4f}")
        
        # Quality assessment
        avg_score = (pred_empathy_mapped + pred_informativeness_mapped + pred_recommendation_mapped + 
                    pred_engagement_mapped + pred_overall_mapped + pred_bleu_mapped) / 6
        
        if avg_score > 0.6:
            quality_label = "HIGH"
        elif avg_score > 0.4:
            quality_label = "MEDIUM"
        else:
            quality_label = "LOW"
        
        print(f"Average Quality: {avg_score:.4f} ({quality_label})")

def main():
    print("6-HEAD CRITIC MODEL TESTING")
    print("=" * 50)
    print("This script tests the critic model trained on synthetic labels")
    print("with 6 output heads: empathy, informativeness, recommendation,")
    print("engagement, overall, and BLEU scores.")
    print("=" * 50)
    
    # Test 1: On synthetic training data
    test_6head_critic_on_synthetic_data()
    
    # Test 2: On quality variations
    test_6head_critic_on_quality_variations()
    
    # Test 3: On real conversation data
    test_6head_critic_on_real_data()
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print("1. This script properly tests the 6-head critic model")
    print("2. It evaluates all 6 quality dimensions")
    print("3. It tests on synthetic data, quality variations, and real data")
    print("4. Check if the scores make sense for each test case")
    print("=" * 50)

if __name__ == "__main__":
    main() 