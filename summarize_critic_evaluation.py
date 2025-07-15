#!/usr/bin/env python3
"""
Critic Evaluation Summary Script

This script provides a human-readable summary of the critic model evaluation results.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

def load_evaluation_metrics(metrics_path: str = 'critic_evaluation_plots/evaluation_metrics.json'):
    """Load evaluation metrics from JSON file."""
    with open(metrics_path, 'r') as f:
        return json.load(f)

def print_detailed_summary(metrics: dict):
    """Print a detailed summary of the evaluation results."""
    print("\n" + "="*80)
    print("DETAILED CRITIC MODEL EVALUATION SUMMARY")
    print("="*80)
    
    metric_names = ['empathy', 'informativeness', 'recommendation', 'engagement', 'overall', 'bleu']
    
    print(f"{'Metric':<15} {'MSE':<12} {'MAE':<12} {'R²':<10} {'Performance':<15}")
    print("-" * 75)
    
    for metric in metric_names:
        mse = metrics[metric]['mse']
        mae = metrics[metric]['mae']
        r2 = metrics[metric]['r2']
        
        # Performance assessment
        if r2 > 0.7:
            performance = "Excellent"
        elif r2 > 0.5:
            performance = "Good"
        elif r2 > 0.3:
            performance = "Fair"
        elif r2 > 0.1:
            performance = "Poor"
        else:
            performance = "Very Poor"
        
        print(f"{metric:<15} {mse:<12.6f} {mae:<12.6f} {r2:<10.4f} {performance:<15}")
    
    # Overall assessment
    avg_r2 = sum(metrics[m]['r2'] for m in metric_names) / len(metric_names)
    avg_mse = sum(metrics[m]['mse'] for m in metric_names) / len(metric_names)
    avg_mae = sum(metrics[m]['mae'] for m in metric_names) / len(metric_names)
    
    print("-" * 75)
    print(f"{'OVERALL':<15} {avg_mse:<12.6f} {avg_mae:<12.6f} {avg_r2:<10.4f}")
    
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    print("1. R² Scores: All R² values are 0.0, indicating that the model predictions")
    print("   are not correlated with the ground truth labels. This suggests:")
    print("   - The model may not have learned meaningful patterns from the training data")
    print("   - The synthetic labels may not be representative of real quality metrics")
    print("   - The model architecture may need adjustment")
    print("   - More training data or different training approach may be needed")
    
    print("\n2. MSE and MAE: The low MSE and MAE values suggest the model is predicting")
    print("   values close to zero, which may indicate:")
    print("   - The model is predicting conservative, low-variance scores")
    print("   - The ground truth labels may have low variance")
    print("   - The model may be underfitting the data")
    
    print("\n3. Recommendations:")
    print("   - Consider using real human annotations instead of synthetic scores")
    print("   - Increase the diversity and size of the training dataset")
    print("   - Experiment with different model architectures")
    print("   - Try different loss functions or training strategies")
    print("   - Validate the quality of the synthetic labels")

def display_plots(plots_dir: str = 'critic_evaluation_plots'):
    """Display the generated plots."""
    plots_path = Path(plots_dir)
    
    if not plots_path.exists():
        print(f"Plots directory {plots_dir} not found!")
        return
    
    plot_files = [
        'overall_performance.png',
        'metrics_comparison.png',
        'distributions.png',
        'correlation_heatmap.png',
        'error_analysis.png'
    ]
    
    print(f"\nGenerated plots in {plots_dir}:")
    for plot_file in plot_files:
        plot_path = plots_path / plot_file
        if plot_path.exists():
            print(f"  ✓ {plot_file}")
        else:
            print(f"  ✗ {plot_file} (missing)")

def main():
    print("CRITIC MODEL EVALUATION SUMMARY")
    print("="*50)
    
    # Load metrics
    try:
        metrics = load_evaluation_metrics()
        print_detailed_summary(metrics)
        display_plots()
    except FileNotFoundError:
        print("Evaluation metrics file not found. Please run the evaluation script first.")
    except Exception as e:
        print(f"Error loading evaluation results: {e}")

if __name__ == "__main__":
    main() 