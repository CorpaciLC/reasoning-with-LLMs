# Quick Start Guide

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd reasoning-with-LLMs
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running Your First Experiment

1. Create a minimal configuration:
```python
from src.experiments_v2 import ExperimentConfig, ExperimentRunner

# Minimal configuration for quick testing
config = ExperimentConfig(
    noise_levels=[0],
    sequence_lengths=[24],
    num_samples=100,
    architectures=['lstm'],
    learning_rates=[0.01],
    num_neurons=[32],
    batch_sizes=[10],
    epochs=5,
    n_folds=2
)

# Run experiment
runner = ExperimentRunner(config)
results = runner.run()
```

2. View results:
```python
# Print summary
print(results.describe())

# Access specific metrics
print("\nAverage MSE per architecture:")
print(results.groupby('architecture')['mse_mean'].mean())
```

## Common Tasks

### Comparing Architectures
```python
config = ExperimentConfig(
    architectures=['ltc', 'lstm', 'gru'],
    noise_levels=[0, 0.1],
    sequence_lengths=[24],
    epochs=10
)
```

### Testing Noise Robustness
```python
config = ExperimentConfig(
    architectures=['ltc'],
    noise_levels=[0, 0.1, 0.2, 0.5],
    sequence_lengths=[24],
    epochs=25
)
```

### Long Sequence Learning
```python
config = ExperimentConfig(
    architectures=['ltc', 'lstm'],
    noise_levels=[0],
    sequence_lengths=[48, 96, 192],
    epochs=50
)
```

## Visualizing Results

```python
from src.utils.plotting import plot_training_curves, plot_model_comparison

# Plot training curves
plot_training_curves(runner.histories, runner.labels)

# Compare models
plot_model_comparison(results)
```

## Next Steps

1. Check the full [documentation](experiments.md) for detailed information
2. Explore different architectures and configurations
3. Add custom metrics or architectures
4. Contribute to the framework

## Getting Help

- Check the troubleshooting section in the main documentation
- File issues on GitHub
- Contact the maintainers