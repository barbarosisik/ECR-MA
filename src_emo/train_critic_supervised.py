#!/usr/bin/env python3
"""
Supervised Critic Pretraining for ECR RL Enhancement

This script pretrains the critic agent using supervised learning on quality-labeled data
before starting RL training. This provides stable value estimates for PPO.
"""

import os
import json
import argparse
import logging
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, 
    get_linear_schedule_with_warmup,
    set_seed
)
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import wandb

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.critic import CriticAgent
from utils import setup_logging

class CriticPretrainingDataset(Dataset):
    """Dataset for supervised critic pretraining."""
    
    def __init__(self, data_path: str, max_length: int = 512):
        self.max_length = max_length
        self.data = self._load_data(data_path)
        
    def _load_data(self, data_path: str) -> List[Dict]:
        """Load and process data for critic pretraining."""
        data = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                
                # Extract context and response
                context_list = item.get('context', [])
                response = item.get('response', '')
                quality_label = item.get('quality_label', 0.0)
                
                # Convert context list to string (join conversation turns)
                context = ' '.join([turn.strip() for turn in context_list if turn.strip()])
                
                # Use quality_label as the target score
                overall_score = quality_label
                # Use the detailed quality scores from the data
                quality_scores = item.get('quality_scores', {
                    'bleu_score': quality_label * 0.25,
                    'distinct_score': quality_label * 0.25,
                    'empathy_score': quality_label * 0.25,
                    'recommendation_score': quality_label * 0.25
                })
                
                # Create training sample
                sample = {
                    'context': context,
                    'response': response,
                    'quality_scores': quality_scores,
                    'overall_score': overall_score
                }
                data.append(sample)
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Return raw text data - the critic will handle tokenization
        return {
            'context': item['context'],
            'response': item['response'],
            'quality_scores': torch.tensor(list(item['quality_scores'].values()), dtype=torch.float),
            'overall_score': torch.tensor(item['overall_score'], dtype=torch.float)
        }

def train_critic_supervised(
    model: CriticAgent,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler,
    device: torch.device,
    num_epochs: int,
    use_wandb: bool = False
) -> Dict:
    """Train the critic using supervised learning."""
    
    model.train()
    best_val_loss = float('inf')
    training_history = {
        'train_losses': [],
        'val_losses': [],
        'train_accuracies': [],
        'val_accuracies': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_targets = []
        
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs} - Training')
        
        for batch in progress_bar:
            # Extract batch data
            contexts = batch['context']
            responses = batch['response']
            quality_scores = batch['quality_scores'].to(device)
            overall_score = batch['overall_score'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            outputs = model(
                context=contexts,
                responses=responses,
                return_quality_breakdown=True
            )
            
            value_output = outputs['values']
            quality_output = torch.stack([
                outputs['quality_breakdown']['bleu_score'],
                outputs['quality_breakdown']['distinct_score'],
                outputs['quality_breakdown']['empathy_score'],
                outputs['quality_breakdown']['recommendation_score']
            ], dim=1)
            emotion_output = outputs.get('emotion_probs', None)
            
            # Calculate losses
            value_loss = nn.MSELoss()(value_output.squeeze(), overall_score)
            quality_loss = nn.MSELoss()(quality_output, quality_scores)
            
            # Emotion classification loss (if applicable)
            emotion_loss = 0.0
            if emotion_output is not None:
                # For now, we'll skip emotion loss in supervised pretraining
                pass
            
            total_loss = value_loss + quality_loss + emotion_loss
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            train_loss += total_loss.item()
            train_predictions.extend(value_output.squeeze().detach().cpu().numpy())
            train_targets.extend(overall_score.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'value_loss': f'{value_loss.item():.4f}',
                'quality_loss': f'{quality_loss.item():.4f}'
            })
        
        # Calculate training metrics
        train_loss /= len(train_dataloader)
        train_accuracy = accuracy_score(
            [1 if p > 0.5 else 0 for p in train_predictions],
            [1 if t > 0.5 else 0 for t in train_targets]
        )
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation'):
                contexts = batch['context']
                responses = batch['response']
                quality_scores = batch['quality_scores'].to(device)
                overall_score = batch['overall_score'].to(device)
                
                outputs = model(
                    context=contexts,
                    responses=responses,
                    return_quality_breakdown=True
                )
                
                value_output = outputs['values']
                quality_output = torch.stack([
                    outputs['quality_breakdown']['bleu_score'],
                    outputs['quality_breakdown']['distinct_score'],
                    outputs['quality_breakdown']['empathy_score'],
                    outputs['quality_breakdown']['recommendation_score']
                ], dim=1)
                
                # Calculate losses
                value_loss = nn.MSELoss()(value_output.squeeze(), overall_score)
                quality_loss = nn.MSELoss()(quality_output, quality_scores)
                total_loss = value_loss + quality_loss
                
                val_loss += total_loss.item()
                val_predictions.extend(value_output.squeeze().cpu().numpy())
                val_targets.extend(overall_score.cpu().numpy())
        
        val_loss /= len(val_dataloader)
        val_accuracy = accuracy_score(
            [1 if p > 0.5 else 0 for p in val_predictions],
            [1 if t > 0.5 else 0 for t in val_targets]
        )
        
        # Log metrics
        training_history['train_losses'].append(train_loss)
        training_history['val_losses'].append(val_loss)
        training_history['train_accuracies'].append(train_accuracy)
        training_history['val_accuracies'].append(val_accuracy)
        
        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'learning_rate': scheduler.get_last_lr()[0]
            })
        
        logging.info(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        logging.info(f'Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_model('best_critic_pretrained.pth')
            logging.info(f'New best model saved with validation loss: {val_loss:.4f}')
    
    return training_history

def main():
    parser = argparse.ArgumentParser(description='Supervised Critic Pretraining')
    parser.add_argument('--train_data', type=str, required=True,
                       help='Path to training data file')
    parser.add_argument('--val_data', type=str, required=True,
                       help='Path to validation data file')
    parser.add_argument('--output_dir', type=str, default='./critic_pretrained',
                       help='Output directory for pretrained critic')
    parser.add_argument('--model_name', type=str, default='roberta-base',
                       help='Base model for critic')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--warmup_steps', type=int, default=100,
                       help='Number of warmup steps')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use wandb for logging')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    setup_logging()
    
    if args.use_wandb:
        wandb.init(project="ecr-critic-pretraining", config=vars(args))
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Set cache directory to avoid internet access issues
    cache_dir = "/data1/s3905993/cache/huggingface"
    os.environ['HF_HOME'] = cache_dir
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    
    # Load tokenizer and model from local cache
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=cache_dir,
        local_files_only=True  # Force use of local cache only
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Define emotion list (should match your use case, here is a common set)
    emotion_list = [
        "like", "curious", "happy", "grateful", "negative", "neutral", "nostalgia", "agreement", "surprise"
    ]

    # Build config object for CriticAgent
    class CriticConfig:
        def __init__(self, model_name, device):
            self.critic_model_name = model_name
            self.critic_hidden_size = 768  # for roberta-base
            self.critic_dropout = 0.1
            self.device = device
        def to_dict(self):
            return self.__dict__
    config = CriticConfig(args.model_name, device)

    # Initialize critic
    critic = CriticAgent(
        config,
        tokenizer,
        emotion_list
    ).to(device)
    
    # Create datasets
    train_dataset = CriticPretrainingDataset(args.train_data, args.max_length)
    val_dataset = CriticPretrainingDataset(args.val_data, args.max_length)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    logging.info(f'Training samples: {len(train_dataset)}')
    logging.info(f'Validation samples: {len(val_dataset)}')
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(critic.parameters(), lr=args.learning_rate)
    total_steps = len(train_dataloader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Train the critic
    logging.info('Starting supervised critic pretraining...')
    training_history = train_critic_supervised(
        model=critic,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.num_epochs,
        use_wandb=args.use_wandb
    )
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'critic_pretrained_final.pth')
    critic.save_model(final_model_path)
    
    # Save training history
    history_path = os.path.join(args.output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Save configuration
    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    logging.info(f'Critic pretraining completed. Model saved to {args.output_dir}')
    
    if args.use_wandb:
        wandb.finish()

if __name__ == '__main__':
    main() 