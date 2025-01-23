"""Visualization utilities for prediction experiments"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import json

def plot_prediction_accuracy(results: Dict, save_path: str = None):
    """Plot prediction accuracy across different LLMs and contexts"""
    # Extract metrics
    data = []
    for variation, var_results in results.items():
        metrics = var_results['metrics']
        settings = var_results['settings']
        context_level = sum(1 for v in settings.values() if v)
        
        data.append({
            'Variation': variation,
            'Context Level': context_level,
            'MSE': metrics['mse'] if metrics['mse'] is not None else float('nan'),
            'MAE': metrics['mae'] if metrics['mae'] is not None else float('nan'),
            'Correlation': metrics['correlation'] if metrics['correlation'] is not None else float('nan')
        })
    
    df = pd.DataFrame(data)
    
    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # MSE plot
    sns.barplot(data=df, x='Context Level', y='MSE', ax=axes[0])
    axes[0].set_title('Mean Squared Error vs Context')
    
    # MAE plot
    sns.barplot(data=df, x='Context Level', y='MAE', ax=axes[1])
    axes[1].set_title('Mean Absolute Error vs Context')
    
    # Correlation plot
    sns.barplot(data=df, x='Context Level', y='Correlation', ax=axes[2])
    axes[2].set_title('Prediction Correlation vs Context')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig

def plot_predictions_vs_true(results: Dict, true_values: List[float], 
                           sequence_ids: List[int] = None, save_path: str = None):
    """Plot predicted vs true values for selected sequences"""
    if sequence_ids is None:
        sequence_ids = list(range(min(5, len(true_values))))  # Default to first 5
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot true values
    true_subset = [true_values[i] for i in sequence_ids]
    ax.plot(sequence_ids, true_subset, 'k-', label='True Values', linewidth=2)
    
    # Plot predictions for each variation
    for variation, var_results in results.items():
        predictions = var_results['predictions']
        pred_subset = [predictions[i] for i in sequence_ids]
        ax.plot(sequence_ids, pred_subset, '--', label=f'{variation}', alpha=0.7)
    
    ax.set_xlabel('Sequence ID')
    ax.set_ylabel('Value')
    ax.set_title('Predictions vs True Values')
    ax.legend()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig

def plot_explanation_heatmap(results: Dict, save_path: str = None):
    """Create heatmap of explanation patterns"""
    # Extract key terms from explanations
    key_terms = ['trend', 'pattern', 'increase', 'decrease', 'stable', 'noise']
    
    data = []
    for variation, var_results in results.items():
        term_counts = {term: 0 for term in key_terms}
        
        for explanation in var_results['explanations']:
            if explanation:  # Skip None values
                for term in key_terms:
                    if term in explanation.lower():
                        term_counts[term] += 1
        
        data.append({
            'Variation': variation,
            **term_counts
        })
    
    df = pd.DataFrame(data)
    df.set_index('Variation', inplace=True)
    
    # Create heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, cmap='YlOrRd', fmt='d')
    plt.title('Explanation Term Frequency')
    
    if save_path:
        plt.savefig(save_path)
    
    return plt.gcf()

def create_report(results_path: str, output_dir: str):
    """Create comprehensive visualization report"""
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizations
    plot_prediction_accuracy(results, output_dir / 'accuracy.png')
    plot_explanation_heatmap(results, output_dir / 'explanations.png')
    
    # Create summary stats
    summary = {
        'variation_metrics': {},
        'top_terms': {},
        'consistency_metrics': {}
    }
    
    for variation, var_results in results.items():
        summary['variation_metrics'][variation] = var_results['metrics']
    
    # Save summary
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=4)

if __name__ == '__main__':
    # Example usage
    results_file = 'path/to/prediction_results.json'
    output_dir = 'path/to/output'
    create_report(results_file, output_dir)