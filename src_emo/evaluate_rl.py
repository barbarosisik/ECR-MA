#!/usr/bin/env python3
"""
Enhanced Evaluation Script for RL-Enhanced ECR-main
Evaluates both supervised and RL-trained models with comprehensive metrics
"""

import argparse
import json
import os
import sys
import time
import torch
import numpy as np
from accelerate import Accelerator
from accelerate.utils import set_seed
from loguru import logger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel

from config import gpt2_special_tokens_dict, Emo_List
from dataset_emp import CRSEmpDataCollator, CRSEmpDataset
from dataset_dbpedia import DBpedia
from evaluate_conv import ConvEvaluator
from model_gpt2 import PromptGPT2forCRS

# Import RL components
from rl import RLConfig, CriticAgent, RewardCalculator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible evaluation.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model.")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Where to store evaluation results.")
    parser.add_argument("--debug", action='store_true', help="Debug mode.")
    
    # Data arguments
    parser.add_argument("--dataset", type=str, default="redial", help="Dataset to evaluate on.")
    parser.add_argument('--context_max_length', type=int, default=150, help="Max length of context.")
    parser.add_argument('--resp_max_length', type=int, default=150, help="Max length of response.")
    parser.add_argument("--tokenizer", type=str, default="microsoft/DialoGPT-small")
    parser.add_argument("--model_name", type=str, default="microsoft/DialoGPT-small")
    
    # Evaluation arguments
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size for evaluation.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate on.")
    parser.add_argument("--max_gen_len", type=int, default=150, help="Maximum generation length.")
    
    # RL evaluation arguments
    parser.add_argument("--use_rl_eval", action="store_true", help="Whether to use RL-based evaluation")
    parser.add_argument("--critic_path", type=str, help="Path to critic model for RL evaluation")
    parser.add_argument("--bleu_weight", type=float, default=1.0, help="BLEU reward weight")
    parser.add_argument("--distinct_weight", type=float, default=0.5, help="Distinct reward weight")
    parser.add_argument("--empathy_weight", type=float, default=2.0, help="Empathy reward weight")
    parser.add_argument("--recommendation_weight", type=float, default=1.5, help="Recommendation reward weight")
    
    # Generation arguments
    parser.add_argument("--do_sample", action="store_true", help="Whether to use sampling for generation.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for generation.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p for generation.")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search.")
    
    # Additional arguments
    parser.add_argument("--wk", action="store_true", default=False)
    parser.add_argument("--wt", action="store_true", default=False)
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    # Initialize accelerator
    accelerator = Accelerator(device_placement=False)
    device = accelerator.device
    
    # Setup logging
    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    logger.remove()
    logger.add(sys.stderr, level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.add(f'log/eval_{local_time}.log', level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.info(accelerator.state)
    logger.info(vars(args))
    
    # Set seed
    if args.seed is not None:
        set_seed(args.seed)
    
    # Create output directory
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Set cache directory to avoid internet access issues
    cache_dir = "/data1/s3905993/cache/huggingface"
    
    # Initialize tokenizer and model from local cache
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        cache_dir=cache_dir,
        local_files_only=True
    )
    tokenizer.add_special_tokens(gpt2_special_tokens_dict)
    
    # Load model from local cache
    if os.path.isdir(args.model_path):
        model = PromptGPT2forCRS.from_pretrained(args.model_path)
    else:
        model = PromptGPT2forCRS.from_pretrained(
            args.model_name,
            cache_dir=cache_dir,
            local_files_only=True
        )
        # Load checkpoint if provided
        if args.model_path.endswith('.pt'):
            checkpoint = torch.load(args.model_path, map_location=device)
            if 'policy_model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['policy_model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
    
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)
    
    # Initialize knowledge graph
    kg = DBpedia(dataset=args.dataset, debug=args.debug)
    
    # Create dataset
    eval_dataset = CRSEmpDataset(
        args.dataset, args.split, tokenizer, debug=args.debug,
        context_max_length=args.context_max_length, resp_max_length=args.resp_max_length, kg=kg, wk=args.wk, wt=args.wt
    )
    
    # Create dataloader
    data_collator = CRSEmpDataCollator(
        tokenizer=tokenizer, device=device, gen=True, use_amp=False, debug=args.debug,
        ignore_pad_token_for_loss=True,
        context_max_length=args.context_max_length, resp_max_length=args.resp_max_length,
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.num_workers,
        collate_fn=data_collator,
    )
    
    # Initialize RL components if using RL evaluation
    if args.use_rl_eval:
        logger.info("Initializing RL evaluation components...")
        
        rl_config = RLConfig(
            bleu_weight=args.bleu_weight,
            distinct_weight=args.distinct_weight,
            empathy_weight=args.empathy_weight,
            recommendation_weight=args.recommendation_weight,
            device=device
        )
        
        reward_calculator = RewardCalculator(rl_config, tokenizer, Emo_List)
        
        if args.critic_path and os.path.exists(args.critic_path):
            critic = CriticAgent.load_model(args.critic_path, rl_config, tokenizer)
        else:
            critic = CriticAgent(rl_config, tokenizer, Emo_List)
    
    # Initialize evaluator
    gen_file_path = os.path.join(args.output_dir, f'generations_{local_time}.jsonl')
    evaluator = ConvEvaluator(tokenizer=tokenizer, log_file_path=gen_file_path)
    
    # Evaluation loop
    logger.info("***** Running evaluation *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Batch size = {args.per_device_eval_batch_size}")
    
    model.eval()
    all_rewards = []
    all_critic_values = []
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            # Generate responses
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.shape[1] + args.max_gen_len,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True
            )
            
            # Extract generated responses
            generated_responses = generated_ids[:, input_ids.shape[1]:]
            
            # Evaluate with standard metrics
            evaluator.evaluate(
                preds=generated_responses,
                labels=batch['labels'],
                log=True,
                context=input_ids
            )
            
            # RL-based evaluation if enabled
            if args.use_rl_eval:
                # Decode responses for reward calculation
                decoded_responses = tokenizer.batch_decode(generated_responses, skip_special_tokens=True)
                decoded_contexts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                decoded_targets = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
                
                # Calculate rewards
                rewards = reward_calculator.calculate_reward(
                    context=decoded_contexts,
                    generated_responses=decoded_responses,
                    target_responses=decoded_targets
                )
                all_rewards.extend(rewards.cpu().numpy())
                
                # Get critic values
                critic_outputs = critic(decoded_contexts, decoded_responses)
                all_critic_values.extend(critic_outputs['values'].cpu().numpy())
    
    # Get evaluation results
    eval_results = evaluator.report()
    
    # Add RL metrics if using RL evaluation
    if args.use_rl_eval:
        eval_results['rl_mean_reward'] = float(np.mean(all_rewards))
        eval_results['rl_std_reward'] = float(np.std(all_rewards))
        eval_results['rl_mean_critic_value'] = float(np.mean(all_critic_values))
        eval_results['rl_std_critic_value'] = float(np.std(all_critic_values))
        
        # Calculate reward breakdown for a sample
        sample_context = decoded_contexts[0] if 'decoded_contexts' in locals() else "Sample context"
        sample_response = decoded_responses[0] if 'decoded_responses' in locals() else "Sample response"
        sample_target = decoded_targets[0] if 'decoded_targets' in locals() else "Sample target"
        
        reward_breakdown = reward_calculator.get_reward_breakdown(
            sample_context, sample_response, sample_target
        )
        eval_results['reward_breakdown'] = reward_breakdown
    
    # Save results
    results_file = os.path.join(args.output_dir, f'evaluation_results_{local_time}.json')
    with open(results_file, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    # Print results
    logger.info("***** Evaluation Results *****")
    for key, value in eval_results.items():
        if key != 'reward_breakdown':
            logger.info(f"  {key}: {value}")
        else:
            logger.info(f"  {key}:")
            for sub_key, sub_value in value.items():
                logger.info(f"    {sub_key}: {sub_value}")
    
    logger.info(f"Results saved to {results_file}")
    logger.info(f"Generations saved to {gen_file_path}")


if __name__ == "__main__":
    main() 