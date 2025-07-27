"""
Reinforcement Learning Module for ECR-main
Enhances empathetic conversational recommender system with RL optimization
"""

from .critic import CriticAgent
from .ppo_trainer import SimplePPOTrainer
from .reward_functions import RewardCalculator
from .rl_config import RLConfig

__all__ = [
    'CriticAgent',
    'SimplePPOTrainer', 
    'RewardCalculator',
    'RLConfig'
] 