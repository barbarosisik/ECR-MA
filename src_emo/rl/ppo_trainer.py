"""
PPO Trainer for RL Training
Implements Proximal Policy Optimization for empathetic response generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from typing import Dict, List, Optional, Tuple
import numpy as np
import json
import os
from tqdm import tqdm
import wandb

from .critic import CriticAgent
from .reward_functions import RewardCalculator
from .rl_config import RLConfig


class PPOTrainer:
    """PPO Trainer for empathetic response generation"""
    
    def __init__(self, 
                 config: RLConfig,
                 policy_model,  # The main response generation model
                 tokenizer,
                 emotion_list,
                 train_dataset,
                 valid_dataset,
                 output_dir: str,
                 critic_pretrained_path: Optional[str] = None):
        
        self.config = config
        self.policy_model = policy_model
        self.tokenizer = tokenizer
        self.emotion_list = emotion_list
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.output_dir = output_dir
        
        # Initialize critic - load pretrained if available
        if critic_pretrained_path and os.path.exists(critic_pretrained_path):
            print(f"Loading pretrained critic from {critic_pretrained_path}")
            self.critic = CriticAgent.load_model(critic_pretrained_path, config, tokenizer)
        else:
            print("Initializing new critic (no pretrained model found)")
            self.critic = CriticAgent(config, tokenizer, emotion_list)
        
        self.reward_calculator = RewardCalculator(config, tokenizer, emotion_list)
        
        # Optimizers
        self.policy_optimizer = AdamW(
            self.policy_model.parameters(),
            lr=config.rl_learning_rate,
            weight_decay=0.01
        )
        
        self.critic_optimizer = AdamW(
            self.critic.parameters(),
            lr=config.rl_learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate schedulers
        self.policy_scheduler = get_linear_schedule_with_warmup(
            self.policy_optimizer,
            num_warmup_steps=config.rl_warmup_steps,
            num_training_steps=config.rl_max_steps
        )
        
        self.critic_scheduler = get_linear_schedule_with_warmup(
            self.critic_optimizer,
            num_warmup_steps=config.rl_warmup_steps,
            num_training_steps=config.rl_max_steps
        )
        
        # Training state
        self.global_step = 0
        self.best_reward = -float('inf')
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def train(self, num_epochs: int = 1):
        """Main training loop"""
        print(f"Starting PPO training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training phase
            train_metrics = self._train_epoch()
            
            # Evaluation phase
            if self.global_step % self.config.eval_frequency == 0:
                eval_metrics = self._evaluate()
                self._log_metrics(train_metrics, eval_metrics, is_eval=True)
                
                # Save best model
                if eval_metrics['mean_reward'] > self.best_reward:
                    self.best_reward = eval_metrics['mean_reward']
                    self._save_checkpoint('best_model')
            
            # Regular checkpointing
            if self.global_step % self.config.save_frequency == 0:
                self._save_checkpoint(f'checkpoint_step_{self.global_step}')
            
            self._log_metrics(train_metrics, {}, is_eval=False)
            
            if self.global_step >= self.config.rl_max_steps:
                break
        
        print("PPO training completed!")
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.policy_model.train()
        self.critic.train()
        
        # Create dataloader
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.rl_batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        epoch_metrics = {
            'policy_loss': 0.0,
            'critic_loss': 0.0,
            'total_reward': 0.0,
            'mean_reward': 0.0,
            'num_batches': 0
        }
        
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            # Generate responses using current policy
            responses, log_probs = self._generate_responses(batch)
            
            # Calculate rewards
            rewards = self.reward_calculator.calculate_reward(
                context=batch['context'],
                generated_responses=responses,
                target_responses=batch['target'],
                emotion_labels=batch.get('emotion', None)
            )
            
            # Get value estimates from critic
            critic_outputs = self.critic(
                context=batch['context'],
                responses=responses
            )
            values = critic_outputs['values']
            
            # Compute advantages
            advantages = self.critic.compute_advantage(rewards, values)
            
            # PPO update
            policy_loss, critic_loss = self._ppo_update(
                batch, responses, log_probs, rewards, values, advantages
            )
            
            # Update metrics
            epoch_metrics['policy_loss'] += policy_loss
            epoch_metrics['critic_loss'] += critic_loss
            epoch_metrics['total_reward'] += rewards.sum().item()
            epoch_metrics['num_batches'] += 1
            
            self.global_step += 1
            
            # Early stopping
            if self.global_step >= self.config.rl_max_steps:
                break
        
        # Average metrics
        if epoch_metrics['num_batches'] > 0:
            epoch_metrics['policy_loss'] /= epoch_metrics['num_batches']
            epoch_metrics['critic_loss'] /= epoch_metrics['num_batches']
            epoch_metrics['mean_reward'] = epoch_metrics['total_reward'] / (
                epoch_metrics['num_batches'] * self.config.rl_batch_size
            )
        
        return epoch_metrics
    
    def _ppo_update(self, 
                   batch: Dict,
                   responses: List[str],
                   log_probs: torch.Tensor,
                   rewards: torch.Tensor,
                   values: torch.Tensor,
                   advantages: torch.Tensor) -> Tuple[float, float]:
        """Perform PPO update"""
        
        # Store old log probs for ratio calculation
        old_log_probs = log_probs.detach()
        
        policy_losses = []
        critic_losses = []
        
        for ppo_epoch in range(self.config.ppo_epochs):
            # Forward pass to get new log probs
            new_log_probs = self._get_log_probs(batch, responses)
            
            # Calculate ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # PPO clipped objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 
                               1 - self.config.ppo_clip_epsilon,
                               1 + self.config.ppo_clip_epsilon) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values, rewards)
            
            # Entropy bonus for exploration
            entropy = self._calculate_entropy(new_log_probs)
            entropy_bonus = self.config.ppo_entropy_coef * entropy
            
            # Total policy loss
            total_policy_loss = policy_loss - entropy_bonus
            
            # Update policy
            self.policy_optimizer.zero_grad()
            total_policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy_model.parameters(), 
                self.config.ppo_max_grad_norm
            )
            self.policy_optimizer.step()
            self.policy_scheduler.step()
            
            # Update critic
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(), 
                self.config.ppo_max_grad_norm
            )
            self.critic_optimizer.step()
            self.critic_scheduler.step()
            
            policy_losses.append(policy_loss.item())
            critic_losses.append(value_loss.item())
        
        return np.mean(policy_losses), np.mean(critic_losses)
    
    def _generate_responses(self, batch: Dict) -> Tuple[List[str], torch.Tensor]:
        """Generate responses using current policy"""
        self.policy_model.eval()
        
        with torch.no_grad():
            # Prepare inputs
            input_ids = batch['input_ids'].to(self.config.device)
            attention_mask = batch['attention_mask'].to(self.config.device)
            
            # Generate responses
            outputs = self.policy_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.config.rl_batch_size + 50,  # Add some buffer
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
            
            # Decode responses
            responses = self.tokenizer.batch_decode(
                outputs.sequences, 
                skip_special_tokens=True
            )
            
            # Calculate log probabilities
            log_probs = self._calculate_log_probs(outputs.sequences, outputs.scores)
        
        return responses, log_probs
    
    def _calculate_log_probs(self, sequences: torch.Tensor, scores: Tuple[torch.Tensor]) -> torch.Tensor:
        """Calculate log probabilities for generated sequences"""
        log_probs = []
        
        for i, seq in enumerate(sequences):
            seq_log_probs = []
            for j, score in enumerate(scores):
                if j < len(seq) - 1:  # Skip the last token
                    probs = F.softmax(score[i], dim=-1)
                    token_id = seq[j + 1]
                    log_prob = torch.log(probs[token_id] + 1e-8)
                    seq_log_probs.append(log_prob)
            
            if seq_log_probs:
                log_probs.append(torch.stack(seq_log_probs).mean())
            else:
                log_probs.append(torch.tensor(0.0))
        
        return torch.stack(log_probs)
    
    def _get_log_probs(self, batch: Dict, responses: List[str]) -> torch.Tensor:
        """Get log probabilities for given responses"""
        # This is a simplified version - in practice, you'd need to tokenize responses
        # and calculate log probs through the model
        return torch.randn(len(responses), device=self.config.device) * 0.1
    
    def _calculate_entropy(self, log_probs: torch.Tensor) -> torch.Tensor:
        """Calculate entropy of log probabilities"""
        probs = torch.exp(log_probs)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy.mean()
    
    def _evaluate(self) -> Dict[str, float]:
        """Evaluate the current model"""
        self.policy_model.eval()
        self.critic.eval()
        
        eval_dataloader = DataLoader(
            self.valid_dataset,
            batch_size=self.config.rl_batch_size,
            shuffle=False,
            collate_fn=self._collate_fn
        )
        
        total_reward = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                responses, _ = self._generate_responses(batch)
                
                rewards = self.reward_calculator.calculate_reward(
                    context=batch['context'],
                    generated_responses=responses,
                    target_responses=batch['target'],
                    emotion_labels=batch.get('emotion', None)
                )
                
                total_reward += rewards.sum().item()
                num_samples += len(responses)
        
        return {
            'mean_reward': total_reward / num_samples if num_samples > 0 else 0.0,
            'total_reward': total_reward,
            'num_samples': num_samples
        }
    
    def _collate_fn(self, batch):
        """Custom collate function for batching"""
        # This should be implemented based on your dataset structure
        # For now, returning a simple structure
        return {
            'context': [item['context'] for item in batch],
            'target': [item['target'] for item in batch],
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'emotion': [item.get('emotion', None) for item in batch]
        }
    
    def _log_metrics(self, train_metrics: Dict, eval_metrics: Dict, is_eval: bool):
        """Log metrics to wandb or console"""
        if is_eval:
            print(f"\nEvaluation Results:")
            print(f"Mean Reward: {eval_metrics.get('mean_reward', 0):.4f}")
            print(f"Total Reward: {eval_metrics.get('total_reward', 0):.4f}")
        else:
            print(f"\nTraining Metrics:")
            print(f"Policy Loss: {train_metrics.get('policy_loss', 0):.4f}")
            print(f"Critic Loss: {train_metrics.get('critic_loss', 0):.4f}")
            print(f"Mean Reward: {train_metrics.get('mean_reward', 0):.4f}")
        
        # Log to wandb if available
        if wandb.run is not None:
            log_dict = {
                'step': self.global_step,
                **train_metrics,
                **eval_metrics
            }
            wandb.log(log_dict)
    
    def _save_checkpoint(self, name: str):
        """Save training checkpoint"""
        checkpoint_path = os.path.join(self.output_dir, f"{name}.pt")
        
        checkpoint = {
            'global_step': self.global_step,
            'best_reward': self.best_reward,
            'policy_model_state_dict': self.policy_model.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'policy_scheduler_state_dict': self.policy_scheduler.state_dict(),
            'critic_scheduler_state_dict': self.critic_scheduler.state_dict(),
            'config': self.config.to_dict()
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        
        self.global_step = checkpoint['global_step']
        self.best_reward = checkpoint['best_reward']
        
        self.policy_model.load_state_dict(checkpoint['policy_model_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.policy_scheduler.load_state_dict(checkpoint['policy_scheduler_state_dict'])
        self.critic_scheduler.load_state_dict(checkpoint['critic_scheduler_state_dict'])
        
        print(f"Checkpoint loaded from {checkpoint_path}") 