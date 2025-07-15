"""
Reward Functions for RL Training
Implements reward calculation based on response quality, empathy, and recommendation accuracy
"""

import re
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from nltk import ngrams
from nltk.translate.bleu_score import sentence_bleu
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F


class RewardCalculator:
    """Calculates rewards for RL training based on multiple metrics"""
    
    def __init__(self, config, tokenizer, emotion_list):
        self.config = config
        self.tokenizer = tokenizer
        self.emotion_list = emotion_list
        self.slot_pattern = re.compile(r'<movie>')
        
        # Initialize emotion classifier for empathy scoring
        self.emotion_classifier = self._init_emotion_classifier()
        
    def _init_emotion_classifier(self):
        """Initialize emotion classifier for empathy scoring"""
        try:
            model_name = "roberta-base"
            cache_dir = "/data1/s3905993/cache/huggingface"
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=True
            )
            model = AutoModel.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=True
            )
            return {"tokenizer": tokenizer, "model": model}
        except Exception as e:
            print(f"Warning: Could not initialize emotion classifier: {e}")
            return None
    
    def calculate_reward(self, 
                        context: List[str],
                        generated_responses: List[str], 
                        target_responses: List[str],
                        emotion_labels: Optional[List[str]] = None) -> torch.Tensor:
        """
        Calculate comprehensive reward for generated responses
        
        Args:
            context: Input conversation context
            generated_responses: Model generated responses
            target_responses: Ground truth responses
            emotion_labels: Target emotion labels for empathy
            
        Returns:
            torch.Tensor: Reward values for each response
        """
        batch_size = len(generated_responses)
        rewards = torch.zeros(batch_size, device=self.config.device)
        
        for i in range(batch_size):
            # Calculate individual reward components
            bleu_reward = self._calculate_bleu_reward(
                generated_responses[i], target_responses[i])
            
            distinct_reward = self._calculate_distinct_reward(
                generated_responses[i])
            
            empathy_reward = self._calculate_empathy_reward(
                context[i], generated_responses[i], emotion_labels[i] if emotion_labels else None)
            
            recommendation_reward = self._calculate_recommendation_reward(
                generated_responses[i])
            
            format_penalty = self._calculate_format_penalty(
                generated_responses[i])
            
            # Combine rewards with weights
            total_reward = (
                self.config.bleu_weight * bleu_reward +
                self.config.distinct_weight * distinct_reward +
                self.config.empathy_weight * empathy_reward +
                self.config.recommendation_weight * recommendation_reward -
                self.config.format_penalty * format_penalty
            )
            
            rewards[i] = total_reward
            
        return rewards
    
    def _calculate_bleu_reward(self, pred: str, target: str) -> float:
        """Calculate BLEU score reward"""
        try:
            pred_tokens = pred.split()
            target_tokens = [target.split()]
            
            # Calculate BLEU-1, BLEU-2, BLEU-3, BLEU-4
            bleu_scores = []
            for n in range(1, 5):
                weights = [0] * 4
                weights[n-1] = 1
                bleu_score = sentence_bleu(target_tokens, pred_tokens, weights)
                bleu_scores.append(bleu_score)
            
            # Average BLEU scores
            avg_bleu = np.mean(bleu_scores)
            return avg_bleu
            
        except Exception as e:
            print(f"Error calculating BLEU reward: {e}")
            return 0.0
    
    def _calculate_distinct_reward(self, response: str) -> float:
        """Calculate distinctiveness reward"""
        try:
            tokens = response.split()
            distinct_ngrams = set()
            
            # Calculate distinct-1, distinct-2, distinct-3, distinct-4
            for n in range(1, 5):
                ngram_set = set(ngrams(tokens, n))
                distinct_ngrams.update(ngram_set)
            
            # Normalize by response length
            if len(tokens) > 0:
                distinct_ratio = len(distinct_ngrams) / len(tokens)
                return min(distinct_ratio, 1.0)  # Cap at 1.0
            return 0.0
            
        except Exception as e:
            print(f"Error calculating distinct reward: {e}")
            return 0.0
    
    def _calculate_empathy_reward(self, context: str, response: str, target_emotion: Optional[str] = None) -> float:
        """Calculate empathy reward based on emotional alignment"""
        try:
            if self.emotion_classifier is None or target_emotion is None:
                return 0.5  # Neutral reward if no emotion classifier
            
            # Simple emotion keyword matching for empathy
            empathy_keywords = {
                'happy': ['happy', 'joy', 'excited', 'great', 'wonderful'],
                'sad': ['sorry', 'sad', 'unfortunate', 'understand', 'feel'],
                'angry': ['understand', 'frustrated', 'upset', 'annoying'],
                'surprised': ['wow', 'amazing', 'incredible', 'surprising'],
                'fearful': ['scary', 'frightening', 'worried', 'concerned'],
                'disgusted': ['disgusting', 'awful', 'terrible', 'bad']
            }
            
            response_lower = response.lower()
            context_lower = context.lower()
            
            # Check if response contains empathy keywords for the target emotion
            if target_emotion in empathy_keywords:
                emotion_words = empathy_keywords[target_emotion]
                empathy_score = sum(1 for word in emotion_words if word in response_lower)
                return min(empathy_score / len(emotion_words), 1.0)
            
            return 0.5  # Neutral reward
            
        except Exception as e:
            print(f"Error calculating empathy reward: {e}")
            return 0.5
    
    def _calculate_recommendation_reward(self, response: str) -> float:
        """Calculate recommendation accuracy reward"""
        try:
            # Count movie recommendations in response
            movie_slots = len(re.findall(self.slot_pattern, response))
            
            # Reward for having at least one recommendation
            if movie_slots > 0:
                return 1.0
            else:
                return 0.0
                
        except Exception as e:
            print(f"Error calculating recommendation reward: {e}")
            return 0.0
    
    def _calculate_format_penalty(self, response: str) -> float:
        """Calculate format penalty for malformed responses"""
        try:
            penalty = 0.0
            
            # Penalty for very short responses
            if len(response.split()) < 3:
                penalty += 0.5
            
            # Penalty for repetitive tokens
            tokens = response.split()
            if len(tokens) > 0:
                unique_tokens = set(tokens)
                repetition_ratio = 1 - (len(unique_tokens) / len(tokens))
                penalty += repetition_ratio
            
            # Penalty for incomplete sentences
            if not response.strip().endswith(('.', '!', '?')):
                penalty += 0.2
            
            return min(penalty, 1.0)  # Cap penalty at 1.0
            
        except Exception as e:
            print(f"Error calculating format penalty: {e}")
            return 0.0
    
    def get_reward_breakdown(self, context: str, response: str, target: str, emotion: Optional[str] = None) -> Dict[str, float]:
        """Get detailed breakdown of reward components for analysis"""
        return {
            'bleu': self._calculate_bleu_reward(response, target),
            'distinct': self._calculate_distinct_reward(response),
            'empathy': self._calculate_empathy_reward(context, response, emotion),
            'recommendation': self._calculate_recommendation_reward(response),
            'format_penalty': self._calculate_format_penalty(response)
        } 