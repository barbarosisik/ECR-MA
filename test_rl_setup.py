#!/usr/bin/env python3
"""
Test script to verify RL training setup
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    try:
        from src_emo.rl import RLConfig, CriticAgent, RewardCalculator, SimplePPOTrainer
        print("✅ RL modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import RL modules: {e}")
        return False

def test_model_loading():
    """Test if Llama2 model and LoRA adapter can be loaded."""
    print("\nTesting model loading...")
    try:
        base_model_path = "/data1/s3905993/ECRHMAS/src/models/llama2_chat"
        lora_model_path = "/data1/s3905993/ECRHMAS/models/llama2_finetuned_movie_lora_cpu"
        
        # Check if paths exist
        if not os.path.exists(base_model_path):
            print(f"❌ Base model not found: {base_model_path}")
            return False
        
        if not os.path.exists(lora_model_path):
            print(f"❌ LoRA model not found: {lora_model_path}")
            return False
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        print("✅ Tokenizer loaded successfully")
        
        # Load base model (just config to save memory)
        config = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16, device_map="auto")
        print("✅ Base model loaded successfully")
        
        return True
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

def test_critic_loading():
    """Test if critic model can be loaded."""
    print("\nTesting critic loading...")
    try:
        critic_path = "critic_pretrained_dual_model/critic_final.pth"
        
        if not os.path.exists(critic_path):
            print(f"❌ Critic model not found: {critic_path}")
            return False
        
        # Try to load critic
        from src_emo.rl import CriticAgent, RLConfig
        from transformers import AutoTokenizer
        
        # Create config and tokenizer
        config = RLConfig()
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        
        critic = CriticAgent.load_model(critic_path, config, tokenizer)
        print("✅ Critic model loaded successfully")
        
        return True
    except Exception as e:
        print(f"❌ Failed to load critic: {e}")
        return False

def test_data_files():
    """Test if training data files exist."""
    print("\nTesting data files...")
    
    required_files = [
        "src_emo/data/redial/train_data_processed.jsonl",
        "src_emo/data/redial/valid_data_processed.jsonl",
        "src_emo/data/redial/test_data_processed.jsonl"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"✅ {file_path} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {file_path} not found")
            all_exist = False
    
    return all_exist

def test_gpu():
    """Test GPU availability."""
    print("\nTesting GPU...")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"✅ GPU available: {gpu_name} ({gpu_memory:.1f} GB)")
        return True
    else:
        print("❌ No GPU available")
        return False

def main():
    print("=== RL Training Setup Test ===\n")
    
    tests = [
        ("Imports", test_imports),
        ("Model Loading", test_model_loading),
        ("Critic Loading", test_critic_loading),
        ("Data Files", test_data_files),
        ("GPU", test_gpu)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n=== Test Summary ===")
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    print(f"\nOverall Status: {'✅ READY FOR RL TRAINING' if all_passed else '❌ SETUP ISSUES DETECTED'}")
    
    if all_passed:
        print("\n🎉 All tests passed! You can now run RL training with:")
        print("sbatch slurm_scripts/rl_training_llama2_full.slurm")
    else:
        print("\n⚠️  Please fix the issues above before running RL training.")

if __name__ == "__main__":
    main() 