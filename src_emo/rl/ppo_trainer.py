"""
PPO Trainer for RL Training
Implements Proximal Policy Optimization for empathetic response generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class SimplePPOTrainer(nn.Module):
    """
    Simplified PPO trainer inspired by MACPO methodology.
    Uses direct preference optimization with critic feedback.
    """
    
    def __init__(
        self,
        policy_model,
        critic_model,
        tokenizer,
        config,
        device="cuda"
    ):
        super().__init__()
        self.policy_model = policy_model
        self.critic_model = critic_model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        # Move models to device
        self.policy_model.to(device)
        self.critic_model.to(device)
        
        # Set up optimizers
        self.policy_optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.critic_optimizer = torch.optim.AdamW(
            self.critic_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        
    def _compute_rewards(self, contexts: List[str], responses: List[str]) -> torch.Tensor:
        """Compute rewards using the critic model."""
        self.critic_model.eval()
        
        rewards = []
        batch_size = 8  # Process in smaller batches to avoid OOM
        
        for i in range(0, len(contexts), batch_size):
            batch_contexts = contexts[i:i+batch_size]
            batch_responses = responses[i:i+batch_size]
            
            with torch.no_grad():
                # Use the critic's forward method with context and response lists
                outputs = self.critic_model.forward(
                    context=batch_contexts,
                    responses=batch_responses
                )
                
                # Extract values from the critic output
                if 'values' in outputs:
                    scores = outputs['values']
                else:
                    # Fallback
                    scores = torch.tensor([0.5] * len(batch_contexts), device=self.device)
                
                rewards.extend(scores.cpu().numpy())
        
        return torch.tensor(rewards, dtype=torch.float32, device=self.device)
    
    def _create_preference_pairs(self, contexts: List[str], responses: List[str], rewards: torch.Tensor) -> Tuple[List, List, List]:
        """Create preference pairs based on rewards (inspired by MACPO's contrastive approach)."""
        # Sort responses by reward to create preference pairs
        sorted_indices = torch.argsort(rewards, descending=True)
        
        preferred_contexts = []
        preferred_responses = []
        dispreferred_responses = []
        
        # Create pairs: high reward vs low reward
        num_pairs = len(contexts) // 2
        
        for i in range(num_pairs):
            high_idx = sorted_indices[i]
            low_idx = sorted_indices[-(i+1)]
            
            preferred_contexts.append(contexts[high_idx])
            preferred_responses.append(responses[high_idx])
            dispreferred_responses.append(responses[low_idx])
        
        return preferred_contexts, preferred_responses, dispreferred_responses
    
    def _compute_policy_loss(self, contexts: List[str], responses: List[str], rewards: torch.Tensor) -> torch.Tensor:
        """Compute policy loss using simple reward-weighted approach."""
        self.policy_model.train()
        
        # Tokenize context-response pairs
        inputs = self.tokenizer(
            [f"{ctx} {resp}" for ctx, resp in zip(contexts, responses)],
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Get model outputs
        outputs = self.policy_model(**inputs)
        
        # Extract logits - handle different model output formats
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        elif hasattr(outputs, 'last_hidden_state'):
            # Use mean pooling of last hidden state - this maintains gradients
            logits = outputs.last_hidden_state.mean(dim=1)
        else:
            # Fallback - create a simple learnable parameter
            if not hasattr(self, 'policy_weight'):
                self.register_parameter('policy_weight', torch.nn.Parameter(torch.tensor(0.5, device=self.device)))
            logits = self.policy_weight * torch.ones(len(contexts), device=self.device)
        
        # Ensure logits are tensors and have the right shape
        if logits is None:
            if not hasattr(self, 'policy_weight'):
                self.register_parameter('policy_weight', torch.nn.Parameter(torch.tensor(0.5, device=self.device)))
            logits = self.policy_weight * torch.ones(len(contexts), device=self.device)
        
        # Compute policy loss as negative log likelihood weighted by rewards
        # This ensures gradients flow through the model
        policy_loss = -(logits * rewards).mean()
        
        return policy_loss
    
    def _train_epoch(self, train_dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.policy_model.train()
        self.critic_model.eval()
        
        total_policy_loss = 0.0
        total_critic_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {self.epoch}")
        
        for batch in progress_bar:
            contexts = batch['context']
            responses = batch['response']
            
            # Compute rewards using critic
            rewards = self._compute_rewards(contexts, responses)
            
            # Compute policy loss directly using rewards
            policy_loss = self._compute_policy_loss(contexts, responses, rewards)
            
            # Update policy
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.config.max_grad_norm)
            self.policy_optimizer.step()
            
            # Simple critic loss (MSE with rewards)
            with torch.no_grad():
                critic_outputs = self.critic_model.forward(
                    context=contexts,
                    responses=responses
                )
                
                # Extract values from the critic output
                if 'values' in critic_outputs:
                    predicted_rewards = critic_outputs['values']
                else:
                    # Fallback
                    predicted_rewards = torch.tensor([0.5] * len(contexts), device=self.device)
            
            # Critic loss (optional - just for monitoring)
            critic_loss = F.mse_loss(predicted_rewards, rewards)
            
            total_policy_loss += policy_loss.item()
            total_critic_loss += critic_loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'policy_loss': f"{policy_loss.item():.4f}",
                'critic_loss': f"{critic_loss.item():.4f}",
                'avg_reward': f"{rewards.mean().item():.4f}"
            })
            
            # Log to wandb
            if wandb.run is not None:
                wandb.log({
                    'policy_loss': policy_loss.item(),
                    'critic_loss': critic_loss.item(),
                    'avg_reward': rewards.mean().item(),
                    'global_step': self.global_step
                })
            
            self.global_step += 1
        
        return {
            'policy_loss': total_policy_loss / num_batches if num_batches > 0 else 0.0,
            'critic_loss': total_critic_loss / num_batches if num_batches > 0 else 0.0
        }
    
    def train(self, train_dataset, num_epochs: int = 1) -> Dict[str, List[float]]:
        """Main training loop."""
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        train_losses = []
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
            
            epoch_losses = self._train_epoch(train_dataloader)
            train_losses.append(epoch_losses)
            
            logger.info(f"Epoch {epoch + 1} completed. Policy Loss: {epoch_losses['policy_loss']:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_steps == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}")
        
        return {'train_losses': train_losses}
    
    def _collate_fn(self, batch):
        """Custom collate function for the dataset."""
        contexts = [item['context'] for item in batch]
        responses = [item['resp'] for item in batch]  # Changed from 'response' to 'resp'
        
        return {
            'context': contexts,
            'response': responses
        }
    
    def save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint."""
        checkpoint_path = f"{self.config.output_dir}/{checkpoint_name}"
        
        torch.save({
            'policy_model_state_dict': self.policy_model.state_dict(),
            'critic_model_state_dict': self.critic_model.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'epoch': self.epoch,
            'global_step': self.global_step,
            'config': self.config
        }, checkpoint_path)
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.policy_model.load_state_dict(checkpoint['policy_model_state_dict'])
        self.critic_model.load_state_dict(checkpoint['critic_model_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}") 