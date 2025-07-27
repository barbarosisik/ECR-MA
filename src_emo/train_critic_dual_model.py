#!/usr/bin/env python3
"""
Supervised Critic Training for Dual-Model Dataset

This script trains the critic agent using supervised learning on dual-model scored data
(Llama2 + Mistral7B) for enhanced robustness and alignment.
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import wandb

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.critic import CriticAgent
from utils import setup_logging

class DualModelCriticDataset(Dataset):
    """Dataset for critic training on dual-model scored data."""
    
    def __init__(self, data_path: str, max_length: int = 512):
        self.max_length = max_length
        self.data = self._load_data(data_path)
        
    def _load_data(self, data_path: str) -> List[Dict]:
        """Load and process dual-model scored data."""
        data = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                
                # Extract context and response
                context = item.get('context', '')
                response = item.get('response', '')
                
                # Get target scores (averaged from both models)
                target_scores = item.get('target_scores', {})
                
                # Get individual model scores for analysis
                llama2_scores = item.get('llama2_scores', {})
                mistral7b_scores = item.get('mistral7b_scores', {})
                
                # Create training sample
                sample = {
                    'context': context,
                    'response': response,
                    'target_scores': target_scores,
                    'llama2_scores': llama2_scores,
                    'mistral7b_scores': mistral7b_scores
                }
                data.append(sample)
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Convert target scores to tensor
        target_scores = item['target_scores']
        score_tensor = torch.tensor([
            target_scores.get('empathy_score', 0.0),
            target_scores.get('informativeness_score', 0.0),
            target_scores.get('recommendation_score', 0.0),
            target_scores.get('engagement_score', 0.0)
        ], dtype=torch.float)
        
        overall_score = torch.tensor(target_scores.get('overall_score', 0.0), dtype=torch.float)
        
        # Convert individual model scores to tensors to avoid collation issues
        llama2_scores = item['llama2_scores']
        mistral7b_scores = item['mistral7b_scores']
        
        llama2_tensor = torch.tensor([
            llama2_scores.get('empathy_score', 0.0),
            llama2_scores.get('informativeness_score', 0.0),
            llama2_scores.get('recommendation_score', 0.0),
            llama2_scores.get('engagement_score', 0.0),
            llama2_scores.get('overall_score', 0.0)
        ], dtype=torch.float)
        
        mistral7b_tensor = torch.tensor([
            mistral7b_scores.get('empathy_score', 0.0),
            mistral7b_scores.get('informativeness_score', 0.0),
            mistral7b_scores.get('recommendation_score', 0.0),
            mistral7b_scores.get('engagement_score', 0.0),
            mistral7b_scores.get('overall_score', 0.0)
        ], dtype=torch.float)
        
        return {
            'context': item['context'],
            'response': item['response'],
            'quality_scores': score_tensor,
            'overall_score': overall_score,
            'llama2_scores': llama2_tensor,
            'mistral7b_scores': mistral7b_tensor
        }

def train_critic_dual_model(
    model: CriticAgent,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler,
    device: torch.device,
    num_epochs: int,
    use_wandb: bool = False
) -> Dict:
    """Train the critic using dual-model supervised learning."""
    
    model.train()
    best_val_loss = float('inf')
    training_history = {
        'train_losses': [],
        'val_losses': [],
        'train_r2_scores': [],
        'val_r2_scores': []
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
            
            # Calculate losses
            value_loss = nn.MSELoss()(value_output.squeeze(), overall_score)
            quality_loss = nn.MSELoss()(quality_output, quality_scores)
            
            total_loss = value_loss + quality_loss
            
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
        train_r2 = r2_score(train_targets, train_predictions)
        
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
                
                # Update metrics
                val_loss += total_loss.item()
                val_predictions.extend(value_output.squeeze().cpu().numpy())
                val_targets.extend(overall_score.cpu().numpy())
        
        # Calculate validation metrics
        val_loss /= len(val_dataloader)
        val_r2 = r2_score(val_targets, val_predictions)
        
        # Log metrics
        training_history['train_losses'].append(train_loss)
        training_history['val_losses'].append(val_loss)
        training_history['train_r2_scores'].append(train_r2)
        training_history['val_r2_scores'].append(val_r2)
        
        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_r2': train_r2,
                'val_r2': val_r2
            })
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train R²: {train_r2:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val R²: {val_r2:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_critic_dual_model.pth')
            print(f'  New best model saved! (Val Loss: {val_loss:.4f})')
    
    return training_history

def evaluate_critic_dual_model(model: CriticAgent, test_dataloader: DataLoader, device: torch.device) -> Dict:
    """Evaluate the trained critic on test data."""
    model.eval()
    predictions = []
    targets = []
    llama2_scores = []
    mistral7b_scores = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc='Evaluating'):
            contexts = batch['context']
            responses = batch['response']
            overall_score = batch['overall_score'].to(device)
            
            outputs = model(
                context=contexts,
                responses=responses,
                return_quality_breakdown=True
            )
            
            value_output = outputs['values']
            predictions.extend(value_output.squeeze().cpu().numpy())
            targets.extend(overall_score.cpu().numpy())
            
            # Collect individual model scores for comparison
            for i in range(len(contexts)):
                # llama2_scores tensor: [empathy, informativeness, recommendation, engagement, overall]
                llama2_scores.append(batch['llama2_scores'][i][4].item())  # overall_score is at index 4
                # mistral7b_scores tensor: [empathy, informativeness, recommendation, engagement, overall]
                mistral7b_scores.append(batch['mistral7b_scores'][i][4].item())  # overall_score is at index 4
    
    # Calculate metrics
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    
    # Compare with individual models
    llama2_r2 = r2_score(targets, llama2_scores)
    mistral7b_r2 = r2_score(targets, mistral7b_scores)
    
    results = {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'llama2_r2': llama2_r2,
        'mistral7b_r2': mistral7b_r2,
        'predictions': predictions,
        'targets': targets
    }
    
    print(f'\nEvaluation Results:')
    print(f'  MSE: {mse:.4f}')
    print(f'  MAE: {mae:.4f}')
    print(f'  R²: {r2:.4f}')
    print(f'  Llama2 R²: {llama2_r2:.4f}')
    print(f'  Mistral7B R²: {mistral7b_r2:.4f}')
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Train critic agent on dual-model dataset')
    parser.add_argument('--train_data', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--val_data', type=str, required=True,
                       help='Path to validation data')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data')
    parser.add_argument('--output_dir', type=str, default='critic_pretrained_dual_model',
                       help='Output directory for trained model')
    parser.add_argument('--model_name', type=str, default='roberta-base',
                       help='Base model for critic')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use wandb for logging')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Setup logging
    setup_logging()
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(project='ecr-critic-dual-model', config=vars(args))
    
    # Create datasets
    print('Loading datasets...')
    train_dataset = DualModelCriticDataset(args.train_data, args.max_length)
    val_dataset = DualModelCriticDataset(args.val_data, args.max_length)
    test_dataset = DualModelCriticDataset(args.test_data, args.max_length)
    
    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')
    print(f'Test samples: {len(test_dataset)}')
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize critic model
    print('Initializing critic model...')
    
    class CriticConfig:
        def __init__(self, model_name, device):
            self.critic_model_name = model_name
            self.critic_hidden_size = 768  # RoBERTa base hidden size
            self.critic_dropout = 0.1
            self.device = device
        
        def to_dict(self):
            return {
                'critic_model_name': self.critic_model_name,
                'critic_hidden_size': self.critic_hidden_size,
                'critic_dropout': self.critic_dropout,
                'device': self.device
            }
    
    config = CriticConfig(args.model_name, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    emotion_list = ['happy', 'sad', 'angry', 'neutral']  # Default emotion list
    
    model = CriticAgent(config, tokenizer, emotion_list)
    model.to(device)
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(train_dataloader) * 2,  # 2 epochs warmup
        num_training_steps=len(train_dataloader) * args.num_epochs
    )
    
    # Train the model
    print('Starting training...')
    training_history = train_critic_dual_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.num_epochs,
        use_wandb=args.use_wandb
    )
    
    # Load best model for evaluation
    print('Loading best model for evaluation...')
    model.load_state_dict(torch.load('best_critic_dual_model.pth'))
    
    # Evaluate on test set
    print('Evaluating on test set...')
    test_results = evaluate_critic_dual_model(model, test_dataloader, device)
    
    # Save final model and results
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'critic_final.pth'))
    
    # Convert numpy values to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    # Save training history and results
    results = {
        'training_history': convert_numpy(training_history),
        'test_results': convert_numpy(test_results),
        'config': config.to_dict()
    }
    
    with open(os.path.join(args.output_dir, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'Training completed! Model saved to {args.output_dir}')
    
    if args.use_wandb:
        wandb.finish()

if __name__ == '__main__':
    main() 