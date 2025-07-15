"""
RL Configuration for ECR-main Enhancement
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import torch


@dataclass
class RLConfig:
    """Configuration for RL training"""
    
    # PPO Hyperparameters
    ppo_epochs: int = 4
    ppo_clip_epsilon: float = 0.2
    ppo_value_clip_epsilon: float = 0.2
    ppo_entropy_coef: float = 0.01
    ppo_value_coef: float = 0.5
    ppo_max_grad_norm: float = 0.5
    
    # Training Hyperparameters
    rl_learning_rate: float = 1e-5
    rl_batch_size: int = 8
    rl_accumulation_steps: int = 4
    rl_warmup_steps: int = 1000
    rl_max_steps: int = 10000
    
    # Reward Function Weights
    bleu_weight: float = 1.0
    distinct_weight: float = 0.5
    empathy_weight: float = 2.0
    recommendation_weight: float = 1.5
    format_penalty: float = 0.1
    
    # Critic Model
    critic_model_name: str = "roberta-base"
    critic_hidden_size: int = 768
    critic_dropout: float = 0.1
    
    # Multi-Agent Settings
    use_multi_agent: bool = True
    agent_cooperation_weight: float = 0.3
    
    # Evaluation
    eval_frequency: int = 500
    save_frequency: int = 1000
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'ppo_epochs': self.ppo_epochs,
            'ppo_clip_epsilon': self.ppo_clip_epsilon,
            'ppo_value_clip_epsilon': self.ppo_value_clip_epsilon,
            'ppo_entropy_coef': self.ppo_entropy_coef,
            'ppo_value_coef': self.ppo_value_coef,
            'ppo_max_grad_norm': self.ppo_max_grad_norm,
            'rl_learning_rate': self.rl_learning_rate,
            'rl_batch_size': self.rl_batch_size,
            'rl_accumulation_steps': self.rl_accumulation_steps,
            'rl_warmup_steps': self.rl_warmup_steps,
            'rl_max_steps': self.rl_max_steps,
            'bleu_weight': self.bleu_weight,
            'distinct_weight': self.distinct_weight,
            'empathy_weight': self.empathy_weight,
            'recommendation_weight': self.recommendation_weight,
            'format_penalty': self.format_penalty,
            'critic_model_name': self.critic_model_name,
            'critic_hidden_size': self.critic_hidden_size,
            'critic_dropout': self.critic_dropout,
            'use_multi_agent': self.use_multi_agent,
            'agent_cooperation_weight': self.agent_cooperation_weight,
            'eval_frequency': self.eval_frequency,
            'save_frequency': self.save_frequency,
            'device': self.device
        } 