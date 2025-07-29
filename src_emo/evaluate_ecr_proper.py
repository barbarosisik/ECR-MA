#!/usr/bin/env python3
"""
Proper Evaluation Script for ECR LoRA Model
Evaluates the LoRA-enhanced Llama2 model on conversation data
"""

import argparse
import json
import os
import sys
import time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from tqdm.auto import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk import ngrams
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the LoRA model.")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-2-7b-chat-hf", help="Base model for LoRA.")
    parser.add_argument("--output_file", type=str, default="results/ecr_evaluation.json", help="Output file for results.")
    parser.add_argument("--test_file", type=str, default="data/redial/test_data_processed.jsonl", help="Test data file.")
    parser.add_argument("--max_samples", type=int, default=100, help="Maximum number of samples to evaluate.")
    parser.add_argument("--max_gen_len", type=int, default=150, help="Maximum generation length.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for evaluation.")
    parser.add_argument("--do_sample", action="store_true", help="Whether to use sampling for generation.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p for generation.")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for generation.")
    return parser.parse_args()

def load_lora_model_and_tokenizer(model_path, base_model):
    """Load the LoRA model and tokenizer."""
    print(f"Loading base model: {base_model}")
    print(f"Loading LoRA adapter from: {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set padding side to left for decoder-only models
    tokenizer.padding_side = "left"
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    return model, tokenizer

def load_test_data(test_file, max_samples):
    """Load test data from JSONL file."""
    print(f"Loading test data from {test_file}")
    
    data = []
    with open(test_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            data.append(json.loads(line.strip()))
    
    print(f"Loaded {len(data)} samples")
    return data

def format_conversation_for_llama(context, response=None):
    """Format conversation context for Llama2 chat format."""
    # Build conversation history
    messages = []
    
    # Add context turns
    for i, turn in enumerate(context):
        if i % 2 == 0:  # User turn
            messages.append({"role": "user", "content": turn})
        else:  # Assistant turn
            messages.append({"role": "assistant", "content": turn})
    
    # Add current response if provided
    if response:
        messages.append({"role": "assistant", "content": response})
    
    # Format for Llama2
    formatted = ""
    for message in messages:
        if message["role"] == "user":
            formatted += f"[INST] {message['content']} [/INST]"
        else:
            formatted += f" {message['content']}"
    
    return formatted

def compute_bleu_score(predicted, reference, smoothing=True):
    """Compute BLEU score with smoothing."""
    try:
        pred_tokens = predicted.split()
        ref_tokens = [reference.split()]
        
        if smoothing:
            smoothing_function = SmoothingFunction().method1
            return sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing_function)
        else:
            return sentence_bleu(ref_tokens, pred_tokens)
    except Exception as e:
        print(f"BLEU computation error: {e}")
        return 0.0

def compute_distinct_score(text, n=1):
    """Compute distinct-n score."""
    try:
        tokens = text.split()
        if len(tokens) < n:
            return 0.0
        ngrams_list = list(ngrams(tokens, n))
        if len(ngrams_list) == 0:
            return 0.0
        unique_ngrams = len(set(ngrams_list))
        return unique_ngrams / len(ngrams_list)
    except Exception as e:
        print(f"Distinct computation error: {e}")
        return 0.0

def evaluate_model(model, tokenizer, test_data, args):
    """Evaluate the model on test data."""
    print("Starting evaluation...")
    
    device = next(model.parameters()).device
    print(f"Using device: {device}")
    
    results = {
        'bleu_scores': [],
        'distinct_1_scores': [],
        'distinct_2_scores': [],
        'generated_responses': [],
        'reference_responses': [],
        'contexts': [],
        'formatted_prompts': []
    }
    
    # Process in batches
    for i in tqdm(range(0, len(test_data), args.batch_size), desc="Evaluating"):
        batch = test_data[i:i+args.batch_size]
        
        # Prepare inputs
        formatted_prompts = []
        references = []
        contexts = []
        
        for item in batch:
            # Format conversation for Llama2
            formatted_prompt = format_conversation_for_llama(item['context'])
            formatted_prompts.append(formatted_prompt)
            references.append(item['resp'])
            contexts.append(item['context'])
        
        # Tokenize inputs
        inputs = tokenizer(
            formatted_prompts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        # Generate responses
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=inputs['input_ids'].shape[1] + args.max_gen_len,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True if args.num_beams > 1 else False
            )
        
        # Extract generated responses
        generated_responses = []
        for j, gen_ids in enumerate(generated_ids):
            # Remove input tokens
            response_ids = gen_ids[inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            generated_responses.append(response)
        
        # Compute metrics
        for pred, ref, ctx, prompt in zip(generated_responses, references, contexts, formatted_prompts):
            bleu = compute_bleu_score(pred, ref, smoothing=True)
            distinct_1 = compute_distinct_score(pred, 1)
            distinct_2 = compute_distinct_score(pred, 2)
            
            results['bleu_scores'].append(bleu)
            results['distinct_1_scores'].append(distinct_1)
            results['distinct_2_scores'].append(distinct_2)
            results['generated_responses'].append(pred)
            results['reference_responses'].append(ref)
            results['contexts'].append(ctx)
            results['formatted_prompts'].append(prompt)
    
    return results

def compute_final_metrics(results):
    """Compute final evaluation metrics."""
    metrics = {
        'bleu_1': np.mean(results['bleu_scores']),
        'bleu_1_std': np.std(results['bleu_scores']),
        'distinct_1': np.mean(results['distinct_1_scores']),
        'distinct_1_std': np.std(results['distinct_1_scores']),
        'distinct_2': np.mean(results['distinct_2_scores']),
        'distinct_2_std': np.std(results['distinct_2_scores']),
        'num_samples': len(results['bleu_scores']),
        'avg_response_length': np.mean([len(resp.split()) for resp in results['generated_responses']])
    }
    
    return metrics

def save_results(results, metrics, output_file):
    """Save evaluation results."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save detailed results
    detailed_results = {
        'metrics': metrics,
        'samples': []
    }
    
    for i in range(min(20, len(results['contexts']))):  # Save first 20 samples
        sample = {
            'context': results['contexts'][i],
            'formatted_prompt': results['formatted_prompts'][i],
            'reference': results['reference_responses'][i],
            'generated': results['generated_responses'][i],
            'bleu': results['bleu_scores'][i],
            'distinct_1': results['distinct_1_scores'][i],
            'distinct_2': results['distinct_2_scores'][i]
        }
        detailed_results['samples'].append(sample)
    
    with open(output_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"Results saved to {output_file}")

def main():
    args = parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_lora_model_and_tokenizer(args.model_path, args.base_model)
    
    # Load test data
    test_data = load_test_data(args.test_file, args.max_samples)
    
    # Evaluate model
    results = evaluate_model(model, tokenizer, test_data, args)
    
    # Compute final metrics
    metrics = compute_final_metrics(results)
    
    # Print results
    print("\n" + "="*60)
    print("ECR MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"Number of samples: {metrics['num_samples']}")
    print(f"BLEU-1: {metrics['bleu_1']:.4f} ± {metrics['bleu_1_std']:.4f}")
    print(f"Distinct-1: {metrics['distinct_1']:.4f} ± {metrics['distinct_1_std']:.4f}")
    print(f"Distinct-2: {metrics['distinct_2']:.4f} ± {metrics['distinct_2_std']:.4f}")
    print(f"Average response length: {metrics['avg_response_length']:.1f} words")
    print("="*60)
    
    # Save results
    save_results(results, metrics, args.output_file)
    
    # Print some sample outputs
    print("\nSAMPLE OUTPUTS:")
    print("-"*60)
    for i in range(min(5, len(results['contexts']))):
        print(f"Sample {i+1}:")
        print(f"Context: {' | '.join(results['contexts'][i][-2:])}")  # Last 2 turns
        print(f"Reference: {results['reference_responses'][i]}")
        print(f"Generated: {results['generated_responses'][i]}")
        print(f"BLEU: {results['bleu_scores'][i]:.4f}")
        print("-"*40)

if __name__ == "__main__":
    main() 