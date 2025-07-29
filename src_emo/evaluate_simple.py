#!/usr/bin/env python3
"""
Simple Evaluation Script for RL-Enhanced ECR Model
Evaluates the model on conversation data format
"""

import argparse
import json
import os
import sys
import time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
from nltk.translate.bleu_score import sentence_bleu
from nltk import ngrams

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model.")
    parser.add_argument("--output_file", type=str, default="results/simple_evaluation.json", help="Output file for results.")
    parser.add_argument("--test_file", type=str, default="data/redial/test_data_processed.jsonl", help="Test data file.")
    parser.add_argument("--max_samples", type=int, default=100, help="Maximum number of samples to evaluate.")
    parser.add_argument("--max_gen_len", type=int, default=150, help="Maximum generation length.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation.")
    parser.add_argument("--do_sample", action="store_true", help="Whether to use sampling for generation.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for generation.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p for generation.")
    return parser.parse_args()

def load_model_and_tokenizer(model_path):
    """Load the model and tokenizer."""
    print(f"Loading model from {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_path)
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

def compute_bleu_score(predicted, reference):
    """Compute BLEU score for a single prediction."""
    try:
        pred_tokens = predicted.split()
        ref_tokens = [reference.split()]
        return sentence_bleu(ref_tokens, pred_tokens)
    except:
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
    except:
        return 0.0

def evaluate_model(model, tokenizer, test_data, args):
    """Evaluate the model on test data."""
    print("Starting evaluation...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    results = {
        'bleu_scores': [],
        'distinct_1_scores': [],
        'distinct_2_scores': [],
        'generated_responses': [],
        'reference_responses': [],
        'contexts': []
    }
    
    # Process in batches
    for i in tqdm(range(0, len(test_data), args.batch_size), desc="Evaluating"):
        batch = test_data[i:i+args.batch_size]
        
        # Prepare inputs
        contexts = []
        references = []
        
        for item in batch:
            # Build context from conversation history
            context_parts = []
            if item['context']:
                context_parts.extend(item['context'])
            context_parts.append(item['resp'])
            context = " ".join(context_parts)
            contexts.append(context)
            references.append(item['resp'])
        
        # Tokenize inputs
        inputs = tokenizer(
            contexts,
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
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True
            )
        
        # Extract generated responses
        generated_responses = []
        for j, gen_ids in enumerate(generated_ids):
            # Remove input tokens
            response_ids = gen_ids[inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(response_ids, skip_special_tokens=True)
            generated_responses.append(response)
        
        # Compute metrics
        for pred, ref, ctx in zip(generated_responses, references, contexts):
            bleu = compute_bleu_score(pred, ref)
            distinct_1 = compute_distinct_score(pred, 1)
            distinct_2 = compute_distinct_score(pred, 2)
            
            results['bleu_scores'].append(bleu)
            results['distinct_1_scores'].append(distinct_1)
            results['distinct_2_scores'].append(distinct_2)
            results['generated_responses'].append(pred)
            results['reference_responses'].append(ref)
            results['contexts'].append(ctx)
    
    return results

def compute_final_metrics(results):
    """Compute final evaluation metrics."""
    metrics = {
        'bleu_1': np.mean(results['bleu_scores']),
        'distinct_1': np.mean(results['distinct_1_scores']),
        'distinct_2': np.mean(results['distinct_2_scores']),
        'num_samples': len(results['bleu_scores'])
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
    
    for i in range(min(10, len(results['contexts']))):  # Save first 10 samples
        sample = {
            'context': results['contexts'][i],
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
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    
    # Load test data
    test_data = load_test_data(args.test_file, args.max_samples)
    
    # Evaluate model
    results = evaluate_model(model, tokenizer, test_data, args)
    
    # Compute final metrics
    metrics = compute_final_metrics(results)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Number of samples: {metrics['num_samples']}")
    print(f"BLEU-1: {metrics['bleu_1']:.4f}")
    print(f"Distinct-1: {metrics['distinct_1']:.4f}")
    print(f"Distinct-2: {metrics['distinct_2']:.4f}")
    print("="*50)
    
    # Save results
    save_results(results, metrics, args.output_file)
    
    # Print some sample outputs
    print("\nSAMPLE OUTPUTS:")
    print("-"*50)
    for i in range(min(3, len(results['contexts']))):
        print(f"Sample {i+1}:")
        print(f"Context: {results['contexts'][i][:100]}...")
        print(f"Reference: {results['reference_responses'][i]}")
        print(f"Generated: {results['generated_responses'][i]}")
        print(f"BLEU: {results['bleu_scores'][i]:.4f}")
        print("-"*30)

if __name__ == "__main__":
    main() 