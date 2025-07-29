#!/usr/bin/env python3
"""
Optimized Llama2 RL Training Script for ECR-main
Faster training with optimized settings for 24-hour limit
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

from rl import RLConfig, CriticAgent, RewardCalculator, SimplePPOTrainer


class OptimizedRedialDataset(Dataset):
    """Optimized dataset for Redial data with sampling for faster training"""
    
    def __init__(self, data_file, tokenizer, max_length=512, max_samples=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        print(f"Loading data from {data_file}")
        with open(data_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                self.data.append(json.loads(line.strip()))
        
        print(f"Loaded {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format conversation for Llama2
        context = item['context']
        response = item['resp']
        
        # Format as Llama2 chat
        formatted_context = self.format_conversation_for_llama(context)
        
        return {
            'context': formatted_context,
            'response': response,
            'original_context': context
        }
    
    def format_conversation_for_llama(self, context):
        """Format conversation context for Llama2 chat format."""
        if isinstance(context, list):
            # Build conversation history
            messages = []
            for i, turn in enumerate(context):
                if i % 2 == 0:  # User turn
                    messages.append({"role": "user", "content": turn})
                else:  # Assistant turn
                    messages.append({"role": "assistant", "content": turn})
            
            # Format for Llama2
            formatted = ""
            for message in messages:
                if message["role"] == "user":
                    formatted += f"[INST] {message['content']} [/INST]"
                else:
                    formatted += f" {message['content']}"
            
            return formatted
        else:
            # If context is already a string, just wrap it
            return f"[INST] {context} [/INST]"


def simple_collate_fn(batch):
    """Custom collate function to handle variable-length sequences."""
    contexts = [item['context'] for item in batch]
    responses = [item['response'] for item in batch]
    original_contexts = [item['original_context'] for item in batch]
    
    return {
        'context': contexts,
        'response': responses,
        'original_context': original_contexts
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, help="Where to store the final model.", default="models/rl_enhanced_llama2_optimized")
    parser.add_argument("--debug", action='store_true', help="Debug mode.")
    
    # Data arguments
    parser.add_argument("--train_file", type=str, default="src_emo/data/redial/train_data_processed.jsonl", help="Training data file")
    parser.add_argument("--val_file", type=str, default="src_emo/data/redial/valid_data_processed.jsonl", help="Validation data file")
    parser.add_argument('--context_max_length', type=int, default=512, help="max length of context input.")
    parser.add_argument('--resp_max_length', type=int, default=150, help="max length of response output.")
    parser.add_argument('--max_train_samples', type=int, default=20000, help="Maximum training samples to use")
    parser.add_argument('--max_val_samples', type=int, default=2000, help="Maximum validation samples to use")
    
    # Model arguments
    parser.add_argument("--base_model", type=str, default="/data1/s3905993/ECRHMAS/src/models/llama2_chat", help="Base Llama2 model path")
    parser.add_argument("--lora_model", type=str, default="/data1/s3905993/ECRHMAS/models/llama2_finetuned_movie_lora_cpu", help="LoRA adapter path")
    parser.add_argument("--max_gen_len", type=int, default=150)
    
    # Training arguments - OPTIMIZED FOR SPEED
    parser.add_argument("--num_train_epochs", type=int, default=2, help="Total number of training epochs to perform.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--fp16', action='store_true', help='use automatic mixed precision')
    
    # RL arguments
    parser.add_argument("--use_rl", action="store_true", help="Whether to use RL training")
    parser.add_argument("--rl_learning_rate", type=float, default=1e-5, help="RL learning rate")
    parser.add_argument("--critic_pretrained_path", type=str, default=None, help="Path to pretrained critic model")
    
    # Logging arguments - OPTIMIZED FOR SPEED
    parser.add_argument('--save_steps', type=int, default=1000, help='Save checkpoint every X steps')
    parser.add_argument('--eval_steps', type=int, default=1000, help='Evaluate every X steps')
    parser.add_argument('--logging_steps', type=int, default=200, help='Log every X steps')
    
    args = parser.parse_args()
    return args


def load_llama2_model_and_tokenizer(base_model_path, lora_model_path):
    """Load Llama2 model with LoRA adapter."""
    print(f"Loading base model from: {base_model_path}")
    print(f"Loading LoRA adapter from: {lora_model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set padding side to left for decoder-only models
    tokenizer.padding_side = "left"
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    model.eval()
    
    return model, tokenizer


def compute_bleu_score(predicted, reference):
    """Simple BLEU score computation."""
    try:
        pred_words = predicted.split()
        ref_words = reference.split()
        
        # Simple n-gram overlap
        matches = 0
        for word in pred_words:
            if word in ref_words:
                matches += 1
        
        if len(pred_words) == 0:
            return 0.0
        
        return matches / len(pred_words)
    except:
        return 0.0


def compute_distinct_score(text, n=1):
    """Compute distinct-n score."""
    try:
        tokens = text.split()
        if len(tokens) < n:
            return 0.0
        
        if n == 1:
            unique_tokens = len(set(tokens))
            return unique_tokens / len(tokens)
        else:
            # Simple bigram distinctness
            bigrams = []
            for i in range(len(tokens) - 1):
                bigrams.append((tokens[i], tokens[i + 1]))
            
            if len(bigrams) == 0:
                return 0.0
            
            unique_bigrams = len(set(bigrams))
            return unique_bigrams / len(bigrams)
    except:
        return 0.0


class OptimizedRLTrainer:
    """Optimized RL trainer for Llama2 model."""
    
    def __init__(self, args, model, tokenizer, critic_agent=None):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.critic_agent = critic_agent
        self.device = next(model.parameters()).device
        
        print(f"Using device: {self.device}")
    
    def train_step(self, batch):
        """Single training step."""
        contexts = batch['context']
        references = batch['response']
        
        # Tokenize inputs
        inputs = self.tokenizer(
            contexts,
            padding=True,
            truncation=True,
            max_length=self.args.context_max_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate responses
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=inputs['input_ids'].shape[1] + self.args.max_gen_len,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Extract generated responses
        generated_responses = []
        for j, gen_ids in enumerate(generated_ids):
            response_ids = gen_ids[inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            generated_responses.append(response)
        
        # Calculate rewards
        rewards = []
        for pred, ref in zip(generated_responses, references):
            if self.critic_agent:
                # Use critic if available
                reward = self.critic_agent.get_value_estimate(ref, pred)
            else:
                # Simple reward based on BLEU and distinctness
                bleu = compute_bleu_score(pred, ref)
                distinct = compute_distinct_score(pred, 1)
                reward = bleu + 0.5 * distinct
            rewards.append(reward)
        
        # For now, just return the loss as 0 (placeholder for PPO)
        loss = 0.0
        
        return loss, generated_responses, rewards


def main():
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Load Llama2 model and tokenizer
    model, tokenizer = load_llama2_model_and_tokenizer(args.base_model, args.lora_model)
    
    # Load critic agent if provided
    critic_agent = None
    if args.critic_pretrained_path and os.path.exists(args.critic_pretrained_path):
        print(f"Loading critic from: {args.critic_pretrained_path}")
        from transformers import AutoTokenizer as CriticTokenizer
        config = RLConfig()
        tokenizer_critic = CriticTokenizer.from_pretrained("roberta-base")
        critic_agent = CriticAgent.load_model(args.critic_pretrained_path, config, tokenizer_critic)
    else:
        print("No critic model provided. Using simple reward function.")
    
    # Load datasets with sampling for faster training
    print("Loading datasets...")
    train_dataset = OptimizedRedialDataset(
        args.train_file, tokenizer, args.context_max_length, args.max_train_samples
    )
    val_dataset = OptimizedRedialDataset(
        args.val_file, tokenizer, args.context_max_length, args.max_val_samples
    )
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=simple_collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=simple_collate_fn
    )
    
    # Initialize RL trainer
    rl_trainer = OptimizedRLTrainer(args, model, tokenizer, critic_agent)
    
    # Training loop
    print("Starting optimized training...")
    model.train()
    
    total_steps = 0
    best_reward = -float('inf')
    
    for epoch in range(args.num_train_epochs):
        print(f"Epoch {epoch + 1}/{args.num_train_epochs}")
        
        progress_bar = tqdm(train_dataloader, desc=f"Training epoch {epoch + 1}")
        
        for step, batch in enumerate(progress_bar):
            # Training step
            loss, generated_responses, rewards = rl_trainer.train_step(batch)
            
            total_steps += 1
            
            # Log progress
            if total_steps % args.logging_steps == 0:
                avg_reward = np.mean(rewards) if rewards else 0.0
                print(f"Step {total_steps}: Loss = {loss:.4f}, Avg Reward = {avg_reward:.4f}")
                
                # Show sample output
                if len(generated_responses) > 0:
                    print(f"Sample: {generated_responses[0][:100]}...")
            
            # Save checkpoint
            if total_steps % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{total_steps}")
                os.makedirs(save_path, exist_ok=True)
                model.save_pretrained(save_path, safe_serialization=False)
                tokenizer.save_pretrained(save_path)
                print(f"Checkpoint saved to {save_path}")
            
            # Evaluate
            if total_steps % args.eval_steps == 0:
                print(f"Running evaluation at step {total_steps}...")
                model.eval()
                eval_rewards = []
                
                with torch.no_grad():
                    for eval_batch in val_dataloader:
                        _, _, rewards = rl_trainer.train_step(eval_batch)
                        eval_rewards.extend(rewards)
                
                if len(eval_rewards) > 0:
                    avg_eval_reward = np.mean(eval_rewards)
                    print(f"Evaluation at step {total_steps}: Average reward = {avg_eval_reward:.4f}")
                    
                    # Save best model
                    if avg_eval_reward > best_reward:
                        best_reward = avg_eval_reward
                        best_path = os.path.join(args.output_dir, "best_model")
                        os.makedirs(best_path, exist_ok=True)
                        model.save_pretrained(best_path, safe_serialization=False)
                        tokenizer.save_pretrained(best_path)
                        print(f"New best model saved to {best_path}")
                
                model.train()
    
    # Save final model
    print(f"Saving final model to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=False)
    tokenizer.save_pretrained(args.output_dir)
    
    print("Optimized training completed successfully!")


if __name__ == "__main__":
    main() 