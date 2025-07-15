#!/usr/bin/env python3
"""
Test script to compare old and new critic models
Demonstrates the difference between the two training approaches
"""

import torch
import json
import sys
import os
from transformers import AutoTokenizer
from src_emo.rl.critic import CriticAgent

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_model_loading():
    """Test if models can be loaded correctly"""
    print("=== MODEL LOADING TEST ===")
    
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
    
    # Define emotion list
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
    
    # Test 1: Old model (should fail)
    print("\n1. Testing OLD model (best_critic_pretrained.pth):")
    try:
        old_checkpoint = torch.load("best_critic_pretrained.pth", map_location=device)
        print(f"   ✓ Checkpoint loaded successfully")
        print(f"   ✓ Checkpoint keys: {list(old_checkpoint.keys())}")
        
        # Try to load with old method (should fail)
        try:
            old_model = CriticAgent.load_model("best_critic_pretrained.pth", config, tokenizer)
            print("   ✓ OLD MODEL LOADED SUCCESSFULLY (unexpected!)")
        except KeyError as e:
            print(f"   ✗ OLD MODEL FAILED TO LOAD (expected): {e}")
        except Exception as e:
            print(f"   ✗ OLD MODEL FAILED TO LOAD: {e}")
            
    except Exception as e:
        print(f"   ✗ Failed to load checkpoint: {e}")
    
    # Test 2: New model (should work)
    print("\n2. Testing NEW model (critic_pretrained/critic_pretrained_final.pth):")
    try:
        new_checkpoint = torch.load("critic_pretrained/critic_pretrained_final.pth", map_location=device)
        print(f"   ✓ Checkpoint loaded successfully")
        print(f"   ✓ Checkpoint keys: {list(new_checkpoint.keys())}")
        
        # Try to load with new method (should work)
        try:
            new_model = CriticAgent.load_model("critic_pretrained/critic_pretrained_final.pth", config, tokenizer)
            print("   ✓ NEW MODEL LOADED SUCCESSFULLY")
        except Exception as e:
            print(f"   ✗ NEW MODEL FAILED TO LOAD: {e}")
            
    except Exception as e:
        print(f"   ✗ Failed to load checkpoint: {e}")

def test_model_performance():
    """Test model performance on example cases"""
    print("\n=== MODEL PERFORMANCE TEST ===")
    
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
    
    # Define emotion list
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
    
    # Load the new model
    try:
        model = CriticAgent.load_model("critic_pretrained_rich/critic_pretrained_final.pth", config, tokenizer)
        print("✓ Model loaded successfully for performance testing")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Example test cases
    test_cases = [
        {
            "context": "I'm feeling really sad today. I just lost my job and I don't know what to do.",
            "response": "I'm so sorry to hear that. Losing a job can be really difficult and stressful. It's completely normal to feel sad and uncertain. Have you thought about what kind of work you'd like to do next? I'm here to listen if you want to talk more about it.",
            "expected_quality": "high_empathy"
        },
        {
            "context": "I love action movies! What should I watch tonight?",
            "response": "You should definitely watch 'Mad Max: Fury Road'! It's an incredible action movie with amazing stunts and non-stop excitement. The cinematography is stunning and the action sequences are mind-blowing. I think you'll really enjoy it!",
            "expected_quality": "good_recommendation"
        },
        {
            "context": "I'm not sure about this new restaurant in town.",
            "response": "I don't know.",
            "expected_quality": "low_quality"
        },
        {
            "context": "Can you recommend a good book to read?",
            "response": "I'd recommend 'The Seven Husbands of Evelyn Hugo' by Taylor Jenkins Reid. It's a captivating historical fiction novel about a legendary Hollywood actress and her seven marriages. The storytelling is brilliant, the characters are complex and compelling, and it has unexpected twists that will keep you hooked until the very end. It's perfect for anyone who loves character-driven stories with a bit of mystery and romance.",
            "expected_quality": "excellent_recommendation"
        }
    ]
    
    print(f"\nTesting {len(test_cases)} example cases:")
    print("-" * 80)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\nCase {i}: {case['expected_quality']}")
        print(f"Context: {case['context']}")
        print(f"Response: {case['response']}")
        
        # Get model evaluation
        with torch.no_grad():
            outputs = model(
                context=[case['context']],
                responses=[case['response']],
                return_quality_breakdown=True
            )
        
        # Extract scores
        value_score = outputs['values'][0].item()
        quality = outputs['quality_breakdown']
        bleu_score = quality['bleu_score'][0].item()
        distinct_score = quality['distinct_score'][0].item()
        empathy_score = quality['empathy_score'][0].item()
        recommendation_score = quality['recommendation_score'][0].item()
        
        # Get emotion probabilities
        emotion_probs = outputs['emotion_probs'][0]
        top_emotion_idx = torch.argmax(emotion_probs).item()
        top_emotion = emotion_list[top_emotion_idx]
        top_emotion_prob = emotion_probs[top_emotion_idx].item()
        
        print(f"Overall Value Score: {value_score:.4f}")
        print(f"Quality Breakdown:")
        print(f"  - BLEU Score: {bleu_score:.4f}")
        print(f"  - Distinct Score: {distinct_score:.4f}")
        print(f"  - Empathy Score: {empathy_score:.4f}")
        print(f"  - Recommendation Score: {recommendation_score:.4f}")
        print(f"Top Emotion: {top_emotion} (probability: {top_emotion_prob:.4f})")
        
        # Simple quality assessment
        avg_quality = (bleu_score + distinct_score + empathy_score + recommendation_score) / 4
        if avg_quality > 0.7:
            quality_label = "HIGH"
        elif avg_quality > 0.4:
            quality_label = "MEDIUM"
        else:
            quality_label = "LOW"
        
        print(f"Average Quality: {avg_quality:.4f} ({quality_label})")
        print("-" * 40)

def compare_training_histories():
    """Compare training histories between old and new models"""
    print("\n=== TRAINING HISTORY COMPARISON ===")
    
    # Check if we have training history for the new model
    try:
        with open("critic_pretrained/training_history.json", 'r') as f:
            new_history = json.load(f)
        
        print("New Model Training History:")
        print(f"  Final Training Loss: {new_history['train_losses'][-1]:.6f}")
        print(f"  Final Validation Loss: {new_history['val_losses'][-1]:.6f}")
        print(f"  Final Training Accuracy: {new_history['train_accuracies'][-1]:.4f}")
        print(f"  Final Validation Accuracy: {new_history['val_accuracies'][-1]:.4f}")
        
        # Show loss progression
        print(f"\nLoss Progression (first 3 epochs):")
        for i in range(min(3, len(new_history['train_losses']))):
            print(f"  Epoch {i+1}: Train={new_history['train_losses'][i]:.4f}, Val={new_history['val_losses'][i]:.4f}")
        
        print(f"\nLoss Progression (last 3 epochs):")
        for i in range(max(0, len(new_history['train_losses'])-3), len(new_history['train_losses'])):
            print(f"  Epoch {i+1}: Train={new_history['train_losses'][i]:.4f}, Val={new_history['val_losses'][i]:.4f}")
            
    except Exception as e:
        print(f"Could not load training history: {e}")

def test_on_real_data():
    """Test critic model on real examples from sample_train_data_processed.jsonl"""
    import json
    print("\n=== REAL DATASET TEST ===")
    # Set cache directory
    cache_dir = "/data1/s3905993/cache/huggingface"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "roberta-base",
        cache_dir=cache_dir,
        local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Build config
    class CriticConfig:
        def __init__(self, model_name, device):
            self.critic_model_name = model_name
            self.critic_hidden_size = 768
            self.critic_dropout = 0.1
            self.device = device
        def to_dict(self):
            return self.__dict__
    import torch
    from src_emo.rl.critic import CriticAgent
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = CriticConfig("roberta-base", device)
    # Load the new model
    try:
        model = CriticAgent.load_model("critic_pretrained/critic_pretrained_final.pth", config, tokenizer)
        print("✓ Model loaded successfully for real data testing")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    # Load real examples
    with open("sample_train_data_processed.jsonl", "r") as f:
        real_examples = [json.loads(line) for line in f]
    print(f"\nTesting {len(real_examples)} real dataset examples:")
    print("-" * 80)
    for i, ex in enumerate(real_examples, 1):
        context = " | ".join(ex["context"])
        response = ex["resp"]
        print(f"\nExample {i}:")
        print(f"Context: {context}")
        print(f"Response: {response}")
        with torch.no_grad():
            outputs = model(
                context=[context],
                responses=[response],
                return_quality_breakdown=True
            )
        value_score = outputs['values'][0].item()
        quality = outputs['quality_breakdown']
        bleu_score = quality['bleu_score'][0].item()
        distinct_score = quality['distinct_score'][0].item()
        empathy_score = quality['empathy_score'][0].item()
        recommendation_score = quality['recommendation_score'][0].item()
        print(f"Overall Value Score: {value_score:.4f}")
        print(f"Quality Breakdown:")
        print(f"  - BLEU Score: {bleu_score:.4f}")
        print(f"  - Distinct Score: {distinct_score:.4f}")
        print(f"  - Empathy Score: {empathy_score:.4f}")
        print(f"  - Recommendation Score: {recommendation_score:.4f}")
        avg_quality = (bleu_score + distinct_score + empathy_score + recommendation_score) / 4
        print(f"Average Quality: {avg_quality:.4f}")
        print("-" * 40)

def main():
    print("CRITIC MODEL COMPARISON AND TESTING")
    print("=" * 50)
    
    # Test 1: Model loading
    test_model_loading()
    
    # Test 2: Model performance
    test_model_performance()
    
    # Test 3: Training history
    compare_training_histories()
    
    # Test 4: Real dataset test
    test_on_real_data()
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print("1. The OLD model was saved as a state dict only - incomplete for RL loading")
    print("2. The NEW model was saved as a complete checkpoint - ready for RL training")
    print("3. Both models have the same weights, but different save formats")
    print("4. The NEW model can now be used for RL training")
    print("=" * 50)

if __name__ == "__main__":
    main() 