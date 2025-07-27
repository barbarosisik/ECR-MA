#!/usr/bin/env python3
"""
Test Mixtral8x7B-Instruct access and basic functionality.
"""

import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def test_mixtral_access():
    print("=== TESTING MIXTRAL8X7B-INSTRUCT ACCESS ===")
    
    # Set cache directories
    os.environ['HF_HOME'] = "/data1/s3905993/cache/huggingface"
    os.environ['TRANSFORMERS_CACHE'] = "/data1/s3905993/cache/transformers"
    
    try:
        print("1. Testing tokenizer loading...")
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
        print("✅ Tokenizer loaded successfully!")
        
        print("\n2. Testing model loading with 8-bit quantization + CPU offloading...")
        from transformers import BitsAndBytesConfig
        
        # Use 8-bit quantization with CPU offloading
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_threshold=6.0
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        print("✅ Model loaded successfully!")
        print(f"   Device: {model.device}")
        if torch.cuda.is_available():
            print(f"   GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        
        print("\n3. Testing basic generation...")
        messages = [
            {"role": "user", "content": "Rate this response from 0.0 to 1.0 for empathy: 'I understand how you feel.'"}
        ]
        
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=32,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Generation successful!")
        print(f"   Output: {result}")
        
        print("\n=== ACCESS TEST COMPLETE ===")
        print("Mixtral8x7B-Instruct is ready for scoring!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_mixtral_access() 