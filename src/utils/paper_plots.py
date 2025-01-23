import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

def create_learning_comparison_plot(results_df, save_path=None):
    """Create comparison plot for different architectures and learning rates."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Training loss evolution
    sns.lineplot(data=results_df, x='epoch', y='loss', 
                hue='architecture', style='learning_rate',
                ax=ax1)
    ax1.set_title('Training Loss Evolution')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    
    # Validation accuracy across noise
    sns.boxplot(data=results_df, x='noise_level', y='accuracy',
               hue='architecture', ax=ax2)
    ax2.set_title('Validation Accuracy vs Noise')
    ax2.set_xlabel('Noise Level')
    ax2.set_ylabel('Accuracy')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig

def create_noise_resilience_plot(results_df, save_path=None):
    """Create noise resilience comparison plot."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    metrics = ['mse', 'mae', 'r2', 'runtime']
    
    for ax, metric in zip(axes.flat, metrics):
        sns.lineplot(data=results_df, x='noise_level', y=metric,
                    hue='architecture', style='learning_rate',
                    ax=ax)
        ax.set_title(f'{metric.upper()} vs Noise Level')
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig

def generate_paper_figures(results_dir):
    """Generate all figures for the paper."""
    # Load results
    results_df = pd.read_csv(Path(results_dir) / 'results.csv')
    
    # Create output directory
    output_dir = Path('paper_figures')
    output_dir.mkdir(exist_ok=True)
    
    # Generate figures
    create_learning_comparison_plot(
        results_df,
        save_path=output_dir / 'learning_comparison.png'
    )
    
    create_noise_resilience_plot(
        results_df,
        save_path=output_dir / 'noise_resilience.png'
    )

if __name__ == '__main__':
    # Example usage
    results_dir = 'path/to/results'
    generate_paper_figures(results_dir)
