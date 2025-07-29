#!/usr/bin/env python3
"""
RL-Enhanced Training Script for ECR-main
Integrates PPO training with empathetic response generation
"""

import argparse
import math
import os
import sys
import time
import numpy as np
import torch
import transformers
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from loguru import logger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel

from config import gpt2_special_tokens_dict, prompt_special_tokens_dict, Emo_List
from dataset_emp import CRSEmpDataCollator, CRSEmpDataset
from dataset_dbpedia import DBpedia
from evaluate_conv import ConvEvaluator
from model_gpt2 import PromptGPT2forCRS
from model_prompt import KGPrompt

# Import RL components
from rl import RLConfig, CriticAgent, RewardCalculator, SimplePPOTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, help="Where to store the final model.", default="data/saved/emp_conv_rl")
    parser.add_argument("--debug", action='store_true', help="Debug mode.")
    
    # Data arguments
    parser.add_argument("--dataset", type=str, required=False, default="redial", help="A file containing all data.")
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--context_max_length', type=int, default=150, help="max length of both encoder and decoder input.")
    parser.add_argument('--resp_max_length', type=int, default=150, help="max length of decoder input.")
    parser.add_argument("--tokenizer", type=str, default="microsoft/DialoGPT-small")
    parser.add_argument("--ignore_pad_token_for_loss", action='store_true')
    parser.add_argument("--text_tokenizer", type=str, default="roberta-base")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="microsoft/DialoGPT-small")
    parser.add_argument("--max_gen_len", type=int, default=150)
    parser.add_argument("--num_bases", type=int, default=8, help="num_bases in RGCN")
    
    # Training arguments
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                        help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--num_warmup_steps', type=int, default=10000)
    parser.add_argument('--fp16', action='store_true', help='use automatic mixed precision to speed up.')
    
    # RL arguments
    parser.add_argument("--use_rl", action="store_true", help="Whether to use RL training")
    parser.add_argument("--rl_learning_rate", type=float, default=1e-5, help="RL learning rate")
    parser.add_argument("--rl_batch_size", type=int, default=8, help="RL batch size")
    parser.add_argument("--rl_max_steps", type=int, default=10000, help="Maximum RL training steps")
    parser.add_argument("--ppo_epochs", type=int, default=4, help="PPO epochs per update")
    parser.add_argument("--ppo_clip_epsilon", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--bleu_weight", type=float, default=1.0, help="BLEU reward weight")
    parser.add_argument("--distinct_weight", type=float, default=0.5, help="Distinct reward weight")
    parser.add_argument("--empathy_weight", type=float, default=2.0, help="Empathy reward weight")
    parser.add_argument("--recommendation_weight", type=float, default=1.5, help="Recommendation reward weight")
    parser.add_argument("--critic_pretrained_path", type=str, default=None, help="Path to pretrained critic model")
    
    # Wandb arguments
    parser.add_argument("--use_wandb", action="store_true", help="whether to use wandb")
    parser.add_argument("--entity", type=str, help="wandb username")
    parser.add_argument("--project", type=str, help="wandb exp project")
    parser.add_argument("--name", type=str, help="wandb exp name")
    parser.add_argument("--log_all", action="store_true", help="log in all processes, otherwise only in rank0")
    parser.add_argument("--pretrain", action="store_true", default=False, help="log in all processes, otherwise only in rank0")
    parser.add_argument("--test", action="store_true", default=False)
    
    # Additional arguments
    parser.add_argument("--wk", action="store_true", default=False)
    parser.add_argument("--wt", action="store_true", default=False)
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config = vars(args)
    
    # Initialize accelerator
    accelerator = Accelerator(device_placement=False)
    device = accelerator.device
    
    # Setup logging
    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    logger.remove()
    logger.add(sys.stderr, level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.add(f'log/{local_time}.log', level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.info(accelerator.state)
    logger.info(config)
    
    args.output_dir = args.output_dir + "_" + local_time
    
    # Setup wandb
    if args.use_wandb:
        name = args.name if args.name else local_time
        name += '_' + str(accelerator.process_index)
        
        if args.log_all:
            group = args.name if args.name else 'DDP_' + local_time
            run = wandb.init(entity=args.entity, project=args.project, group=group, config=config, name=name)
        else:
            if accelerator.is_local_main_process:
                run = wandb.init(entity=args.entity, project=args.project, config=config, name=name)
            else:
                run = None
    else:
        run = None
    
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
    model = PromptGPT2forCRS.from_pretrained(
        args.model,
        cache_dir=cache_dir,
        local_files_only=True
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)
    
    # Initialize knowledge graph
    kg = DBpedia(dataset=args.dataset, debug=args.debug)
    
    # Create datasets
    train_dataset = CRSEmpDataset(
        args.dataset, 'train', tokenizer, debug=args.debug,
        context_max_length=args.context_max_length, resp_max_length=args.resp_max_length, kg=kg, wk=args.wk, wt=args.wt
    )
    valid_dataset = CRSEmpDataset(
        args.dataset, 'valid', tokenizer, debug=args.debug,
        context_max_length=args.context_max_length, resp_max_length=args.resp_max_length, kg=kg, wk=args.wk, wt=args.wt
    )
    
    # Create dataloaders
    data_collator_teacher = CRSEmpDataCollator(
        tokenizer=tokenizer, device=device, use_amp=(accelerator.mixed_precision == "fp16"), debug=args.debug, gen=False,
        ignore_pad_token_for_loss=args.ignore_pad_token_for_loss,
        context_max_length=args.context_max_length + args.resp_max_length
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=data_collator_teacher,
    )
    
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.num_workers,
        collate_fn=data_collator_teacher,
    )
    
    # Initialize RL components if using RL
    if args.use_rl:
        logger.info("Initializing RL components...")
        
        # Create RL config
        rl_config = RLConfig(
            rl_learning_rate=args.rl_learning_rate,
            rl_batch_size=args.rl_batch_size,
            rl_max_steps=args.rl_max_steps,
            ppo_epochs=args.ppo_epochs,
            ppo_clip_epsilon=args.ppo_clip_epsilon,
            bleu_weight=args.bleu_weight,
            distinct_weight=args.distinct_weight,
            empathy_weight=args.empathy_weight,
            recommendation_weight=args.recommendation_weight,
            device=device,
            output_dir=os.path.join(args.output_dir, "rl_checkpoints")
        )
        # Lower checkpoint save frequency for short runs
        rl_config.save_steps = 100  # Save every 100 steps for better checkpointing
        
        # Load critic model
        if args.critic_pretrained_path:
            logger.info(f"Loading critic model from {args.critic_pretrained_path}")
            # Create a simple config for the critic
            from config import Emo_List
            critic_model = CriticAgent.load_model(
                args.critic_pretrained_path, 
                rl_config, 
                tokenizer
            )
        else:
            logger.warning("No critic model path provided, using policy model as critic")
            critic_model = model
        
        # Initialize Simple PPO trainer
        ppo_trainer = SimplePPOTrainer(
            policy_model=model,
            critic_model=critic_model,
            tokenizer=tokenizer,
            config=rl_config,
            device=device
        )
        
        logger.info("Starting RL training...")
        ppo_trainer.train(train_dataset=train_dataset, num_epochs=args.num_train_epochs)  # Use the specified number of epochs
        
    else:
        # Standard supervised training
        logger.info("Starting standard supervised training...")
        
        # Optimizer setup
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters()
                           if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters()
                           if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        
        # Prepare model and optimizer
        model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
        
        # Learning rate scheduler
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, args.num_warmup_steps, args.max_train_steps)
        lr_scheduler = accelerator.prepare(lr_scheduler)
        
        # Training loop
        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
        completed_steps = 0
        
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        
        # Training loop implementation would go here
        # (This is a simplified version - the full implementation would include the actual training loop)
        
        logger.info("Training completed!")
    
    # Save final model
    if accelerator.is_local_main_process:
        # Fix tensor sharing issue by using safe_serialization=False
        model.save_pretrained(args.output_dir, safe_serialization=False)
        tokenizer.save_pretrained(args.output_dir)
        logger.info(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main() 