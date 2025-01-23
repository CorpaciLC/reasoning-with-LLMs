# Experimental Framework Documentation

## Overview
The experimental framework provides a structured way to run neural network experiments with various architectures and configurations, including cross-validation and comprehensive metrics collection.

## Key Components

### ExperimentConfig
Configuration class that defines all experimental parameters:

```python
config = ExperimentConfig(
    noise_levels=[0, 0.1, 0.5],           # Different noise levels to test
    sequence_lengths=[24, 48, 96],         # Sequence lengths for time series
    num_samples=1000,                      # Number of samples to generate
    architectures=['ltc', 'rnn', 'lstm'],  # Neural architectures to test
    learning_rates=[0.001, 0.01, 0.1],     # Learning rates to try
    num_neurons=[32, 64],                  # Network sizes
    batch_sizes=[10, 32],                  # Batch sizes
    epochs=25,                             # Training epochs
    n_folds=5                             # Number of cross-validation folds
)
```

### ExperimentRunner
Main class for executing experiments:

```python
# Basic usage
runner = ExperimentRunner(config)
results_df = runner.run()
```

## Supported Architectures

1. **LTC (Liquid Time-Constant Networks)**
   - Requires wiring configuration
   - Supports different connectivity patterns
   ```python
   wiring = wirings.FullyConnected(neurons=64, output_size=1)
   ```

2. **RNN Variants**
   - SimpleRNN
   - LSTM
   - GRU
   - Supports bidirectional configurations

## Directory Structure
```
results/
├── YYYY-MM-DD_HH-MM-SS/      # Timestamp-based experiment directory
│   ├── config.json           # Experiment configuration
│   ├── results.csv           # Metrics for all runs
│   ├── histories.pkl         # Training histories
│   ├── labels.pkl            # Run labels
│   └── checkpoints/          # Model checkpoints
        └── model_name/       # Per-model checkpoints
```

## Results Format
The `results.csv` file contains:
- Architecture details (type, neurons, etc.)
- Training parameters (learning rate, batch size)
- Data parameters (noise, sequence length)
- Metrics (MSE, MAE, R²) with mean and std across folds

## Example Usage

### Basic Experiment
```python
from src.experiments_v2 import ExperimentConfig, ExperimentRunner

# Create configuration
config = ExperimentConfig(
    architectures=['ltc', 'lstm'],
    noise_levels=[0, 0.1],
    sequence_lengths=[24],
    epochs=10
)

# Run experiment
runner = ExperimentRunner(config)
results = runner.run()
```

### Custom Configuration
```python
# Extended configuration for thorough comparison
config = ExperimentConfig(
    architectures=['ltc', 'lstm', 'gru'],
    noise_levels=[0, 0.1, 0.2, 0.5],
    sequence_lengths=[24, 48],
    learning_rates=[0.001, 0.01],
    num_neurons=[32, 64, 128],
    batch_sizes=[16, 32],
    epochs=50,
    n_folds=5,
    save_checkpoints=True
)
```

## Metrics Collection
The framework collects:
- Training metrics (loss, validation loss)
- Performance metrics (MSE, MAE, R²)
- Cross-validation statistics
- Runtime information

## Visualization
Built-in visualization functions:
- `plot_training_curves()`: Training progress
- `plot_model_comparison()`: Architecture comparison

## Best Practices

1. **Configuration Management**
   - Save configurations for reproducibility
   - Use meaningful experiment names
   ```python
   config.save('configs/experiment_v1.json')
   loaded_config = ExperimentConfig.load('configs/experiment_v1.json')
   ```

2. **Resource Management**
   - Monitor memory usage with large models
   - Use appropriate batch sizes
   - Enable checkpointing for long runs

3. **Results Analysis**
   - Use cross-validation for robust comparisons
   - Consider statistical significance
   - Plot learning curves for insight

## Extending the Framework

### Adding New Architectures
1. Create new model class in `models/`
2. Update ModelFactory
3. Add architecture-specific parameters

Example:
```python
# Adding a new architecture
class NewArchitecture(BaseModel):
    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(config, **kwargs)
    
    def _build(self):
        # Implementation
        pass

# Update factory
ModelFactory.create:
    if model_type == 'new_arch':
        return NewArchitecture(config, **kwargs)
```

### Custom Metrics
Add new metrics in `utils/metrics.py`:
```python
def custom_metric(y_true, y_pred):
    # Implementation
    pass

class ModelEvaluator:
    def calculate_metrics(self):
        metrics = super().calculate_metrics()
        metrics['custom'] = custom_metric(self.y_test, self.y_pred)
        return metrics
```

## Troubleshooting

1. **Memory Issues**
   - Reduce batch size
   - Decrease sequence length
   - Limit concurrent models

2. **Convergence Problems**
   - Check learning rates
   - Inspect validation curves
   - Verify data preprocessing

3. **Import Errors**
   - Verify project structure
   - Check package installation
   - Review PYTHONPATH

## Contributing
1. Follow code style (PEP 8)
2. Add tests for new features
3. Update documentation
4. Create pull request