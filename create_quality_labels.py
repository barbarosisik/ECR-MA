#!/usr/bin/env python3
"""
Create quality labels for critic training data
Generate meaningful quality scores based on response characteristics
"""

import json
import re
from typing import List, Dict, Tuple
import numpy as np

def calculate_response_quality(context: List[str], response: str, rec: List, emotion_probs: List[float]) -> Dict[str, float]:
    """
    Calculate quality scores for a response based on various metrics
    
    Args:
        context: Conversation context
        response: Generated response
        rec: Recommended items
        emotion_probs: Emotion probabilities
        
    Returns:
        Dict with quality scores
    """
    
    # 1. Response Length Score (0-1)
    # Prefer responses that are not too short or too long
    response_length = len(response.split())
    if response_length < 3:
        length_score = 0.1  # Too short
    elif response_length < 10:
        length_score = 0.3  # Short but acceptable
    elif response_length < 30:
        length_score = 0.8  # Good length
    elif response_length < 50:
        length_score = 0.9  # Very good length
    else:
        length_score = 0.6  # Too long
    
    # 2. Recommendation Score (0-1)
    # Higher score if recommendations are provided
    if rec and len(rec) > 0:
        rec_score = 0.8
    else:
        rec_score = 0.2
    
    # 3. Emotion Appropriateness Score (0-1)
    # Check if emotion probabilities are reasonable
    if emotion_probs:
        emotion_entropy = -sum(p * np.log(p + 1e-8) for p in emotion_probs if p > 0)
        emotion_score = min(emotion_entropy / 2.0, 1.0)  # Normalize to 0-1
    else:
        emotion_score = 0.5
    
    # 4. Response Specificity Score (0-1)
    # Check for specific words, movie titles, etc.
    specific_words = ['movie', 'film', 'book', 'music', 'song', 'artist', 'actor', 'director', 'genre']
    specific_count = sum(1 for word in specific_words if word.lower() in response.lower())
    specificity_score = min(specific_count * 0.2, 1.0)
    
    # 5. Context Relevance Score (0-1)
    # Check if response relates to context
    context_words = set()
    for turn in context:
        if turn.strip():
            context_words.update(turn.lower().split())
    
    response_words = set(response.lower().split())
    if context_words:
        overlap = len(context_words.intersection(response_words))
        relevance_score = min(overlap / len(context_words), 1.0)
    else:
        relevance_score = 0.5
    
    # 6. BLEU-like Score (0-1)
    # Simple n-gram overlap (not real BLEU, but similar concept)
    response_ngrams = set()
    words = response.lower().split()
    for i in range(len(words) - 1):
        response_ngrams.add((words[i], words[i+1]))
    
    context_ngrams = set()
    for turn in context:
        if turn.strip():
            turn_words = turn.lower().split()
            for i in range(len(turn_words) - 1):
                context_ngrams.add((turn_words[i], turn_words[i+1]))
    
    if context_ngrams:
        ngram_overlap = len(response_ngrams.intersection(context_ngrams))
        bleu_score = min(ngram_overlap / len(context_ngrams), 1.0)
    else:
        bleu_score = 0.5
    
    # 7. Distinctiveness Score (0-1)
    # Check for vocabulary diversity
    unique_words = len(set(response.lower().split()))
    total_words = len(response.split())
    if total_words > 0:
        distinct_score = min(unique_words / total_words, 1.0)
    else:
        distinct_score = 0.0
    
    # 8. Empathy Score (0-1)
    # Check for empathetic language
    empathetic_words = ['sorry', 'understand', 'feel', 'difficult', 'help', 'listen', 'support', 'care']
    empathetic_count = sum(1 for word in empathetic_words if word.lower() in response.lower())
    empathy_score = min(empathetic_count * 0.3, 1.0)
    
    # Calculate overall quality
    overall_score = (
        length_score * 0.15 +
        rec_score * 0.25 +
        emotion_score * 0.1 +
        specificity_score * 0.15 +
        relevance_score * 0.15 +
        bleu_score * 0.1 +
        distinct_score * 0.05 +
        empathy_score * 0.05
    )
    
    return {
        'overall_score': overall_score,
        'bleu_score': bleu_score,
        'distinct_score': distinct_score,
        'empathy_score': empathy_score,
        'recommendation_score': rec_score,
        'length_score': length_score,
        'specificity_score': specificity_score,
        'relevance_score': relevance_score,
        'emotion_score': emotion_score
    }

def process_data_file(input_path: str, output_path: str):
    """Process a data file and add quality labels"""
    
    processed_data = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            try:
                item = json.loads(line.strip())
                
                # Extract fields
                context = item.get('context', [])
                response = item.get('resp', '')
                rec = item.get('rec', [])
                emotion_probs = item.get('emo_probs_lastest', [])
                
                # Calculate quality scores
                quality_scores = calculate_response_quality(context, response, rec, emotion_probs)
                
                # Create new item with quality labels
                new_item = {
                    'context': context,
                    'response': response,
                    'quality_label': quality_scores['overall_score'],
                    'quality_scores': quality_scores,
                    'rec': rec,
                    'emotion_probs': emotion_probs,
                    'original_data': item  # Keep original data for reference
                }
                
                processed_data.append(new_item)
                
                if line_num % 1000 == 0:
                    print(f"Processed {line_num} lines...")
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    # Save processed data
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Processed {len(processed_data)} items")
    print(f"Saved to {output_path}")
    
    # Print some statistics
    overall_scores = [item['quality_label'] for item in processed_data]
    print(f"Quality score statistics:")
    print(f"  Mean: {np.mean(overall_scores):.4f}")
    print(f"  Std: {np.std(overall_scores):.4f}")
    print(f"  Min: {np.min(overall_scores):.4f}")
    print(f"  Max: {np.max(overall_scores):.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create quality labels for critic training data.")
    parser.add_argument('--input', type=str, default='sample_train_data_processed.jsonl', help='Input file')
    parser.add_argument('--output', type=str, default='quality_labels.jsonl', help='Output file')
    args = parser.parse_args()

    process_data_file(args.input, args.output) 