import sys
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models import ModelFactory, ModelConfig
from ncps import wirings
from src.utils.metrics import ModelEvaluator
from src.utils.plotting import plot_training_curves, plot_model_comparison
from src.utils.nn_info import NetworkAnalyzer
from src.data.generators import generate_multiple_sinusoidal_data
from src.data.preprocessing import split_data

# Setup logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    """Enhanced configuration with more options and better documentation"""
    # Data parameters
    noise_levels: List[float] = field(default_factory=lambda: [0, 0.1, 0.5])
    sequence_lengths: List[int] = field(default_factory=lambda: [24, 48, 96])
    num_samples: int = 1000
    
    # Model parameters
    architectures: List[str] = field(default_factory=lambda: ['ltc', 'rnn', 'lstm'])
    learning_rates: List[float] = field(default_factory=lambda: [0.001, 0.01, 0.1])
    num_neurons: List[int] = field(default_factory=lambda: [32, 64])
    batch_sizes: List[int] = field(default_factory=lambda: [10, 32])
    
    # Training parameters
    epochs: int = 25
    n_folds: int = 5  # for cross-validation
    early_stopping_patience: int = 5
    
    # Additional parameters
    random_seed: int = 42
    save_checkpoints: bool = True
    checkpoint_frequency: int = 5

    def save(self, path: str):
        """Save configuration to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

class ExperimentRunner:
    """Class to manage experiment execution and results collection"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = []
        self.histories = []
        self.labels = []
        self.current_experiment_dir = None
        
        # Set random seeds for reproducibility
        tf.random.set_seed(config.random_seed)
        np.random.seed(config.random_seed)

    def setup_experiment_directory(self) -> Path:
        """Create and setup experiment directory with timestamp"""
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        exp_dir = Path('results') / timestamp
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        self.config.save(exp_dir / 'config.json')
        
        self.current_experiment_dir = exp_dir
        return exp_dir

    def create_callbacks(self, model_name: str) -> List[tf.keras.callbacks.Callback]:
        """Create training callbacks including early stopping and checkpoints"""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True
            )
        ]
        
        if self.config.save_checkpoints:
            checkpoint_dir = self.current_experiment_dir / 'checkpoints' / model_name
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    str(checkpoint_dir / 'model_{epoch:02d}.h5'),
                    save_freq=self.config.checkpoint_frequency * self.config.batch_sizes[0]
                )
            )
        
        return callbacks

    def cross_validate_model(self, X: np.ndarray, y: np.ndarray, 
                           model_creator: callable, model_params: dict) -> Dict:
        """Perform k-fold cross validation for a model"""
        kf = KFold(n_splits=self.config.n_folds, shuffle=True, 
                  random_state=self.config.random_seed)
        
        fold_metrics = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            logger.info(f"Training fold {fold + 1}/{self.config.n_folds}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = model_creator(**model_params)
            callbacks = self.create_callbacks(f"{model_params['model_type']}_fold_{fold}")
            
            history = model.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.config.epochs,
                batch_size=model_params['batch_size'],
                callbacks=callbacks,
                verbose=1
            )
            
            evaluator = ModelEvaluator(model.model, X_val, y_val)
            metrics = evaluator.calculate_metrics()
            fold_metrics.append(metrics)
            
            self.histories.append(history)
            self.labels.append(f"{model_params['model_type']}_fold_{fold}")
        
        # Aggregate metrics across folds
        avg_metrics = {}
        for metric in fold_metrics[0].keys():
            values = [m[metric] for m in fold_metrics]
            avg_metrics[metric] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }
        
        return avg_metrics

    def run(self):
        """Execute the complete experiment suite"""
        exp_dir = self.setup_experiment_directory()
        logger.info(f"Starting experiments, saving results to {exp_dir}")
        
        for noise in self.config.noise_levels:
            for seq_len in self.config.sequence_lengths:
                logger.info(f"Generating data with noise={noise}, sequence_length={seq_len}")
                X, y = generate_multiple_sinusoidal_data(
                    self.config.num_samples, 
                    seq_len, 
                    noise
                )
                
                for architecture in self.config.architectures:
                    for lr in self.config.learning_rates:
                        for neurons in self.config.num_neurons:
                            for batch_size in self.config.batch_sizes:
                                logger.info(f"Training {architecture} model with "
                                          f"lr={lr}, neurons={neurons}, batch_size={batch_size}")
                                
                                # Base model params
                                model_params = {
                                    'model_type': architecture,
                                    'config': ModelConfig(
                                        input_shape=(None, 2),
                                        learning_rate=lr,
                                        hidden_units=neurons
                                    ),
                                    'batch_size': batch_size
                                }

                                # Add specific parameters for different architectures
                                if architecture == 'ltc':
                                    model_params['wiring'] = wirings.FullyConnected(neurons, 1)
                                elif architecture in ['rnn', 'lstm']:
                                    model_params['mid_layer'] = None  # Will be handled by factory
                                
                                metrics = self.cross_validate_model(
                                    X, y, 
                                    ModelFactory.create, 
                                    model_params
                                )
                                
                                # Record results
                                self.results.append({
                                    'architecture': architecture,
                                    'neurons': neurons,
                                    'noise': noise,
                                    'sequence_length': seq_len,
                                    'learning_rate': lr,
                                    'batch_size': batch_size,
                                    **{f"{k}_mean": v['mean'] for k, v in metrics.items()},
                                    **{f"{k}_std": v['std'] for k, v in metrics.items()}
                                })
        
        self.save_results(exp_dir)
        return pd.DataFrame(self.results)

    def save_results(self, exp_dir: Path):
        """Save all experimental results and artifacts"""
        # Save results DataFrame
        pd.DataFrame(self.results).to_csv(exp_dir / 'results.csv', index=False)
        
        # Save histories and labels
        with open(exp_dir / 'histories.pkl', 'wb') as f:
            pickle.dump(self.histories, f)
        with open(exp_dir / 'labels.pkl', 'wb') as f:
            pickle.dump(self.labels, f)
        
        # Generate and save plots
        plot_training_curves(self.histories, self.labels)
        plot_model_comparison(pd.DataFrame(self.results))
        
        logger.info(f"Results saved to {exp_dir}")

def main():
    try:
        # Example usage
        config = ExperimentConfig()
        runner = ExperimentRunner(config)
        results_df = runner.run()
        print("Experiments completed successfully!")
    except Exception as e:
        logger.error(f"Error during experiment execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()