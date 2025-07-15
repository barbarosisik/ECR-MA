"""
Critic Agent for RL Training
Evaluates response quality and provides value estimates for PPO
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Optional, Tuple
import numpy as np


class CriticAgent(nn.Module):
    """Critic agent that evaluates response quality and provides value estimates"""
    
    def __init__(self, config, tokenizer, emotion_list):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.emotion_list = emotion_list
        
        # Set cache directory to avoid internet access issues
        cache_dir = "/data1/s3905993/cache/huggingface"
        
        # Initialize the base model from local cache
        self.base_model = AutoModel.from_pretrained(
            config.critic_model_name,
            cache_dir=cache_dir,
            local_files_only=True
        )
        self.base_tokenizer = AutoTokenizer.from_pretrained(
            config.critic_model_name,
            cache_dir=cache_dir,
            local_files_only=True
        )
        
        # Add padding token if not present
        if self.base_tokenizer.pad_token is None:
            self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(config.critic_hidden_size, config.critic_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.critic_dropout),
            nn.Linear(config.critic_hidden_size // 2, 1)
        )
        
        # Quality assessment head (for detailed evaluation)
        self.quality_head = nn.Sequential(
            nn.Linear(config.critic_hidden_size, config.critic_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.critic_dropout),
            nn.Linear(config.critic_hidden_size // 2, 4)  # bleu, distinct, empathy, recommendation
        )
        
        # Emotion classification head
        self.emotion_head = nn.Sequential(
            nn.Linear(config.critic_hidden_size, config.critic_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.critic_dropout),
            nn.Linear(config.critic_hidden_size // 2, len(emotion_list))
        )
        
        self.to(config.device)
        
    def forward(self, 
                context: List[str],
                responses: List[str],
                return_quality_breakdown: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the critic
        
        Args:
            context: Input conversation context
            responses: Generated responses to evaluate
            return_quality_breakdown: Whether to return detailed quality scores
            
        Returns:
            Dict containing value estimates and optional quality breakdown
        """
        batch_size = len(context)
        
        # Prepare inputs for the base model
        inputs = self._prepare_inputs(context, responses)
        
        # Get base model outputs
        with torch.no_grad():
            outputs = self.base_model(**inputs)
        
        # Get pooled representation (use CLS token or mean pooling)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            # Mean pooling over sequence length
            pooled_output = outputs.last_hidden_state.mean(dim=1)
        
        # Value estimation
        values = self.value_head(pooled_output).squeeze(-1)
        
        result = {'values': values}
        
        if return_quality_breakdown:
            # Quality breakdown
            quality_scores = self.quality_head(pooled_output)
            quality_breakdown = {
                'bleu_score': quality_scores[:, 0],
                'distinct_score': quality_scores[:, 1],
                'empathy_score': quality_scores[:, 2],
                'recommendation_score': quality_scores[:, 3]
            }
            result['quality_breakdown'] = quality_breakdown
            
            # Emotion classification
            emotion_logits = self.emotion_head(pooled_output)
            emotion_probs = F.softmax(emotion_logits, dim=-1)
            result['emotion_probs'] = emotion_probs
        
        return result
    
    def _prepare_inputs(self, context: List[str], responses: List[str]) -> Dict[str, torch.Tensor]:
        """Prepare inputs for the base model"""
        # Combine context and response for evaluation
        combined_texts = []
        for ctx, resp in zip(context, responses):
            combined_text = f"Context: {ctx} Response: {resp}"
            combined_texts.append(combined_text)
        
        # Tokenize
        inputs = self.base_tokenizer(
            combined_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        
        return inputs
    
    def evaluate_responses(self, 
                          context: List[str],
                          responses: List[str],
                          targets: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate responses and return detailed metrics
        
        Args:
            context: Input conversation context
            responses: Generated responses
            targets: Target responses (optional, for comparison)
            
        Returns:
            Dict containing evaluation metrics
        """
        with torch.no_grad():
            outputs = self.forward(context, responses, return_quality_breakdown=True)
        
        # Calculate metrics
        values = outputs['values'].cpu().numpy()
        quality = outputs['quality_breakdown']
        
        metrics = {
            'mean_value': float(np.mean(values)),
            'std_value': float(np.std(values)),
            'mean_bleu_score': float(torch.mean(quality['bleu_score']).item()),
            'mean_distinct_score': float(torch.mean(quality['distinct_score']).item()),
            'mean_empathy_score': float(torch.mean(quality['empathy_score']).item()),
            'mean_recommendation_score': float(torch.mean(quality['recommendation_score']).item())
        }
        
        return metrics
    
    def get_value_estimate(self, context: str, response: str) -> float:
        """Get value estimate for a single context-response pair"""
        with torch.no_grad():
            outputs = self.forward([context], [response])
            return outputs['values'][0].item()
    
    def compute_advantage(self, 
                         rewards: torch.Tensor,
                         values: torch.Tensor,
                         gamma: float = 0.99,
                         gae_lambda: float = 0.95) -> torch.Tensor:
        """
        Compute Generalized Advantage Estimation (GAE)
        
        Args:
            rewards: Reward sequence
            values: Value estimates
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            
        Returns:
            Advantage estimates
        """
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value - values[t]
            advantages[t] = delta + gamma * gae_lambda * last_advantage
            last_advantage = advantages[t]
        
        return advantages
    
    def save_model(self, path: str):
        """Save the critic model"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config.to_dict(),
            'emotion_list': self.emotion_list
        }, path)
    
    @classmethod
    def load_model(cls, path: str, config, tokenizer):
        """Load the critic model"""
        checkpoint = torch.load(path, map_location=config.device)
        model = cls(config, tokenizer, checkpoint['emotion_list'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model 