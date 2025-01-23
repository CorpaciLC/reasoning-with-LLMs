# utils/plotting.py
import matplotlib.pyplot as plt
import seaborn as sns

def plot_model_comparison(results_df):
    """Plot comparison of model results."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Check actual column names
    print("Available columns:", results_df.columns.tolist())
    
    # Use correct column names from results DataFrame
    sns.barplot(
        x='model',  # Changed from 'Label'
        y='final_val_loss',  # Changed from 'Final Loss'
        data=results_df, 
        ax=ax1
    )
    ax1.set_title('Final Validation Loss by Model')
    ax1.set_xlabel('Model Type')
    ax1.set_ylabel('Final Validation Loss')
    
    # Runtime comparison
    sns.barplot(
        x='model',
        y='runtime',
        data=results_df,
        ax=ax2
    )
    ax2.set_title('Runtime by Model')
    ax2.set_xlabel('Model Type')
    ax2.set_ylabel('Runtime (seconds)')
    
    plt.tight_layout()
    return fig


def plot_training_curves(histories, labels, metric='loss'):
    """Plot training curves for multiple models."""
    plt.figure(figsize=(10, 6))
    
    for history, label in zip(histories, labels):
        plt.plot(history.history[metric], label=f"{label} train")
        if f"val_{metric}" in history.history:
            plt.plot(history.history[f"val_{metric}"], 
                    label=f"{label} val",
                    linestyle='--')
    
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.title(f'Training and Validation {metric.capitalize()}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return plt.gcf()