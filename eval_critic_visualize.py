#!/usr/bin/env python3
"""
Critic Model Evaluation and Visualization Script

This script evaluates the trained critic model on test data and generates
visualizations of the predictions for different quality metrics.
"""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

# Add the src_emo directory to the path
import sys
sys.path.append('src_emo')

from rl.critic import CriticAgent

def load_data(file_path: str) -> List[Dict]:
    """Load JSONL data file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def load_critic_model(model_path: str, device: str = 'cpu') -> CriticAgent:
    """Load the trained critic model."""
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create config object from saved config
    class CriticConfig:
        def __init__(self, config_dict):
            self.critic_model_name = config_dict.get('critic_model_name', 'roberta-base')
            self.critic_hidden_size = config_dict.get('critic_hidden_size', 768)
            self.critic_dropout = config_dict.get('critic_dropout', 0.1)
            self.device = device
    
    config = CriticConfig(checkpoint['config'])
    
    # Create dummy tokenizer and emotion list (will be overridden by model loading)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    emotion_list = checkpoint.get('emotion_list', ['neutral'])
    
    # Initialize model
    model = CriticAgent(config, tokenizer, emotion_list)
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model

def extract_data_for_evaluation(data: List[Dict]) -> Tuple[List[str], List[str], torch.Tensor]:
    """Extract context, responses, and labels from data for CriticAgent evaluation."""
    contexts = []
    responses = []
    labels = []
    
    for item in data:
        # Extract context and response
        context_list = item.get('context', [])
        response = item.get('response', '')
        
        # Convert context list to string
        context = ' '.join([turn.strip() for turn in context_list if turn.strip()])
        
        contexts.append(context)
        responses.append(response)
        
        # Extract labels for quality scores
        quality_scores = item.get('quality_scores', {})
        label = [
            quality_scores.get('empathy_score', 0.0),
            quality_scores.get('informativeness_score', 0.0),
            quality_scores.get('recommendation_score', 0.0),
            quality_scores.get('engagement_score', 0.0),
            item.get('overall_score', 0.0),
            quality_scores.get('bleu_score', 0.0)
        ]
        labels.append(label)
    
    return contexts, responses, torch.tensor(labels, dtype=torch.float32)

def evaluate_model(model: CriticAgent, contexts: List[str], responses: List[str], 
                  labels: torch.Tensor, device: str = 'cpu') -> Dict:
    """Evaluate the model and return metrics."""
    model.eval()
    with torch.no_grad():
        labels = labels.to(device)
        
        # Get model predictions
        outputs = model(contexts, responses, return_quality_breakdown=True)
        
        # Extract predictions
        value_predictions = outputs['values'].cpu().numpy()
        quality_predictions = torch.stack([
            outputs['quality_breakdown']['bleu_score'],
            outputs['quality_breakdown']['distinct_score'],
            outputs['quality_breakdown']['empathy_score'],
            outputs['quality_breakdown']['recommendation_score']
        ], dim=1).cpu().numpy()
        
        # Combine predictions (value + quality scores)
        # Note: We'll use value predictions for overall score and quality predictions for individual metrics
        pred_np = np.column_stack([
            quality_predictions[:, 2],  # empathy
            quality_predictions[:, 1],  # distinct (as informativeness proxy)
            quality_predictions[:, 3],  # recommendation
            quality_predictions[:, 1],  # distinct (as engagement proxy)
            value_predictions,          # overall
            quality_predictions[:, 0]   # bleu
        ])
        
        labels_np = labels.cpu().numpy()
        
        metrics = {}
        metric_names = ['empathy', 'informativeness', 'recommendation', 'engagement', 'overall', 'bleu']
        
        for i, name in enumerate(metric_names):
            metrics[name] = {
                'mse': mean_squared_error(labels_np[:, i], pred_np[:, i]),
                'mae': mean_absolute_error(labels_np[:, i], pred_np[:, i]),
                'r2': r2_score(labels_np[:, i], pred_np[:, i]),
                'predictions': pred_np[:, i],
                'ground_truth': labels_np[:, i]
            }
    
    return metrics

def create_visualizations(metrics: Dict, output_dir: str = 'critic_evaluation_plots'):
    """Create comprehensive visualizations of the critic predictions."""
    Path(output_dir).mkdir(exist_ok=True)
    
    metric_names = ['empathy', 'informativeness', 'recommendation', 'engagement', 'overall', 'bleu']
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Overall performance summary
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Critic Model Performance Across All Metrics', fontsize=16, fontweight='bold')
    
    for i, metric in enumerate(metric_names):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        pred = metrics[metric]['predictions']
        gt = metrics[metric]['ground_truth']
        
        # Scatter plot
        ax.scatter(gt, pred, alpha=0.6, s=30)
        
        # Perfect prediction line
        min_val, max_val = min(gt.min(), pred.min()), max(gt.max(), pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
        
        # Regression line
        reg = LinearRegression().fit(gt.reshape(-1, 1), pred)
        ax.plot(gt, reg.predict(gt.reshape(-1, 1)), 'g-', alpha=0.8, 
                label=f'R² = {metrics[metric]["r2"]:.3f}')
        
        ax.set_xlabel(f'Ground Truth {metric.title()}')
        ax.set_ylabel(f'Predicted {metric.title()}')
        ax.set_title(f'{metric.title()} (MSE: {metrics[metric]["mse"]:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/overall_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Metrics comparison
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # MSE comparison
    mse_values = [metrics[m]['mse'] for m in metric_names]
    ax1.bar(metric_names, mse_values, color=sns.color_palette("husl", 6))
    ax1.set_title('Mean Squared Error by Metric')
    ax1.set_ylabel('MSE')
    ax1.tick_params(axis='x', rotation=45)
    
    # MAE comparison
    mae_values = [metrics[m]['mae'] for m in metric_names]
    ax2.bar(metric_names, mae_values, color=sns.color_palette("husl", 6))
    ax2.set_title('Mean Absolute Error by Metric')
    ax2.set_ylabel('MAE')
    ax2.tick_params(axis='x', rotation=45)
    
    # R² comparison
    r2_values = [metrics[m]['r2'] for m in metric_names]
    ax3.bar(metric_names, r2_values, color=sns.color_palette("husl", 6))
    ax3.set_title('R² Score by Metric')
    ax3.set_ylabel('R²')
    ax3.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Distribution plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Distribution of Predictions vs Ground Truth', fontsize=16, fontweight='bold')
    
    for i, metric in enumerate(metric_names):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        pred = metrics[metric]['predictions']
        gt = metrics[metric]['ground_truth']
        
        ax.hist(gt, bins=20, alpha=0.7, label='Ground Truth', density=True)
        ax.hist(pred, bins=20, alpha=0.7, label='Predictions', density=True)
        ax.set_xlabel(f'{metric.title()} Score')
        ax.set_ylabel('Density')
        ax.set_title(f'{metric.title()} Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create correlation matrix
    all_predictions = np.column_stack([metrics[m]['predictions'] for m in metric_names])
    all_ground_truth = np.column_stack([metrics[m]['ground_truth'] for m in metric_names])
    
    # Combine predictions and ground truth
    combined_data = np.hstack([all_predictions, all_ground_truth])
    combined_names = [f'Pred_{m}' for m in metric_names] + [f'GT_{m}' for m in metric_names]
    
    corr_matrix = np.corrcoef(combined_data.T)
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                xticklabels=combined_names, yticklabels=combined_names, ax=ax)
    ax.set_title('Correlation Matrix: Predictions vs Ground Truth')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Error analysis
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Error Analysis: Prediction Errors by Metric', fontsize=16, fontweight='bold')
    
    for i, metric in enumerate(metric_names):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        pred = metrics[metric]['predictions']
        gt = metrics[metric]['ground_truth']
        errors = pred - gt
        
        ax.hist(errors, bins=20, alpha=0.7, color='red')
        ax.axvline(0, color='black', linestyle='--', alpha=0.8)
        ax.set_xlabel(f'Prediction Error ({metric.title()})')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{metric.title()} Error Distribution')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_metrics_summary(metrics: Dict):
    """Print a summary of the evaluation metrics."""
    print("\n" + "="*80)
    print("CRITIC MODEL EVALUATION SUMMARY")
    print("="*80)
    
    metric_names = ['empathy', 'informativeness', 'recommendation', 'engagement', 'overall', 'bleu']
    
    print(f"{'Metric':<15} {'MSE':<10} {'MAE':<10} {'R²':<10}")
    print("-" * 50)
    
    for metric in metric_names:
        mse = metrics[metric]['mse']
        mae = metrics[metric]['mae']
        r2 = metrics[metric]['r2']
        print(f"{metric:<15} {mse:<10.4f} {mae:<10.4f} {r2:<10.4f}")
    
    # Overall averages
    avg_mse = np.mean([metrics[m]['mse'] for m in metric_names])
    avg_mae = np.mean([metrics[m]['mae'] for m in metric_names])
    avg_r2 = np.mean([metrics[m]['r2'] for m in metric_names])
    
    print("-" * 50)
    print(f"{'AVERAGE':<15} {avg_mse:<10.4f} {avg_mae:<10.4f} {avg_r2:<10.4f}")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description='Evaluate and visualize critic model performance')
    parser.add_argument('--model_path', type=str, default='critic_pretrained_rich/critic_pretrained_final.pth',
                       help='Path to the trained critic model')
    parser.add_argument('--data_path', type=str, default='llama2_scored_rich.jsonl',
                       help='Path to the test data file')
    parser.add_argument('--output_dir', type=str, default='critic_evaluation_plots',
                       help='Directory to save visualization plots')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use for evaluation (cpu/cuda)')
    
    args = parser.parse_args()
    
    print("Loading data...")
    data = load_data(args.data_path)
    print(f"Loaded {len(data)} samples")
    
    print("Extracting features...")
    contexts, responses, labels = extract_data_for_evaluation(data)
    print(f"Contexts: {len(contexts)}, Responses: {len(responses)}, Labels shape: {labels.shape}")
    
    print("Loading critic model...")
    model = load_critic_model(args.model_path, args.device)
    print("Model loaded successfully")
    
    print("Evaluating model...")
    metrics = evaluate_model(model, contexts, responses, labels, args.device)
    
    print_metrics_summary(metrics)
    
    print("Creating visualizations...")
    create_visualizations(metrics, args.output_dir)
    print(f"Visualizations saved to {args.output_dir}/")
    
    # Save metrics to JSON for later analysis
    metrics_save = {}
    for metric_name, metric_data in metrics.items():
        metrics_save[metric_name] = {
            'mse': float(metric_data['mse']),
            'mae': float(metric_data['mae']),
            'r2': float(metric_data['r2'])
        }
    
    with open(f'{args.output_dir}/evaluation_metrics.json', 'w') as f:
        json.dump(metrics_save, f, indent=2)
    
    print("Evaluation complete!")

if __name__ == "__main__":
    main() 