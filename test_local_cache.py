#!/usr/bin/env python3
"""
Test script to verify local cache functionality
"""

import os
import sys
sys.path.append('src_emo')

# Set cache directory
cache_dir = "/data1/s3905993/cache/huggingface"
os.environ['HF_HOME'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir

def test_model_loading():
    """Test loading all required models from local cache"""
    
    print("=== Testing Local Cache Model Loading ===")
    
    # Test 1: RoBERTa base
    print("\n1. Testing RoBERTa base...")
    try:
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(
            'roberta-base',
            cache_dir=cache_dir,
            local_files_only=True
        )
        model = AutoModel.from_pretrained(
            'roberta-base',
            cache_dir=cache_dir,
            local_files_only=True
        )
        print("✅ RoBERTa base loaded successfully")
    except Exception as e:
        print(f"❌ RoBERTa base failed: {e}")
        return False
    
    # Test 2: DialoGPT small
    print("\n2. Testing DialoGPT small...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            'microsoft/DialoGPT-small',
            cache_dir=cache_dir,
            local_files_only=True
        )
        model = AutoModel.from_pretrained(
            'microsoft/DialoGPT-small',
            cache_dir=cache_dir,
            local_files_only=True
        )
        print("✅ DialoGPT small loaded successfully")
    except Exception as e:
        print(f"❌ DialoGPT small failed: {e}")
        return False
    
    # Test 3: Llama2-chat (from local path)
    print("\n3. Testing Llama2-chat...")
    try:
        llama2_path = "/data1/s3905993/ECRHMAS/src/models/llama2_chat"
        tokenizer = AutoTokenizer.from_pretrained(
            llama2_path,
            cache_dir=cache_dir,
            local_files_only=True
        )
        model = AutoModel.from_pretrained(
            llama2_path,
            cache_dir=cache_dir,
            local_files_only=True
        )
        print("✅ Llama2-chat loaded successfully")
    except Exception as e:
        print(f"❌ Llama2-chat failed: {e}")
        return False
    
    # Test 4: CriticAgent import
    print("\n4. Testing CriticAgent import...")
    try:
        from rl.critic import CriticAgent
        print("✅ CriticAgent imported successfully")
    except Exception as e:
        print(f"❌ CriticAgent import failed: {e}")
        return False
    
    # Test 5: RewardCalculator import
    print("\n5. Testing RewardCalculator import...")
    try:
        from rl.reward_functions import RewardCalculator
        print("✅ RewardCalculator imported successfully")
    except Exception as e:
        print(f"❌ RewardCalculator import failed: {e}")
        return False
    
    print("\n=== All Tests Passed! ===")
    print("✅ Local cache is working correctly")
    print("✅ All models can be loaded without internet access")
    print("✅ Scripts are ready for SLURM execution")
    
    return True

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("\n🎉 Ready to run the ECR RL pipeline!")
        print("Use: sbatch slurm_scripts/ecr_rl_pipeline_fixed.slurm")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1) 