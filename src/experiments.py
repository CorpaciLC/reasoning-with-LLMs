# experiment.py
import sys
sys.path.append('..')

from models import ModelFactory, ModelConfig
from models.ltc import LTCModel
from models.rnn import RNNModel

from typing import Dict, List
from dataclasses import dataclass, field
import pandas as pd
import pickle

from models import ModelFactory, ModelConfig
from utils.metrics import ModelEvaluator
from utils.plotting import plot_training_curves, plot_model_comparison
from utils.nn_info import NetworkAnalyzer
from data.generators import generate_multiple_sinusoidal_data
from data.preprocessing import split_data
from ncps import wirings
from tensorflow.keras import layers
import os
from datetime import datetime

@dataclass
class ExperimentConfig:
    noise_levels: List[float] = field(default_factory=lambda: [0, 0.1, 0.5])
    sequence_lengths: List[int] = field(default_factory=lambda: [24, 48, 96])
    num_samples: int = 1000
    learning_rates: List[float] = field(default_factory=lambda: [0.01])
    num_neurons: List[int] = field(default_factory=lambda: [4, 64])
    batch_sizes: List[int] = field(default_factory=lambda: [10])
    epochs: int = 25

def run_experiments(config: ExperimentConfig):
    results = []
    histories = []
    labels = []

    for noise in config.noise_levels:
        for seq_len in config.sequence_lengths:
            # Generate data for this experiment
            X, y = generate_multiple_sinusoidal_data(
                config.num_samples, 
                seq_len, 
                noise
            )
            # Split into train, val, test
            X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)
            
            for lr in config.learning_rates:
                for neurons in config.num_neurons:
                    for batch_size in config.batch_sizes:
                        print(f"\n=== Experiment: noise={noise}, seq_len={seq_len}, "
                              f"lr={lr}, neurons={neurons}, batch_size={batch_size} ===")

                        # Configure a common ModelConfig
                        model_config = ModelConfig(
                            input_shape=(None, 2),
                            learning_rate=lr,
                            hidden_units=neurons
                        )

                        # ---------------------------
                        # 1) LTC Model
                        # ---------------------------
                        wiring = wirings.FullyConnected(neurons, 1)
                        ltc_model = ModelFactory.create(
                            model_type='ltc', 
                            config=model_config,
                            wiring=wiring  # pass wiring so LTCModel can use it
                        )

                        # Train LTC (direct .fit approach)
                        history_ltc = ltc_model.model.fit(
                            X_train, y_train,
                            validation_data=(X_val, y_val),
                            epochs=config.epochs,
                            batch_size=batch_size,
                            verbose=0  # or 1 if you want progress
                        )
                        histories.append(history_ltc)
                        labels.append(f'LTC_n{neurons}_lr{lr}_noise{noise}')

                        # Evaluate LTC
                        evaluator_ltc = ModelEvaluator(ltc_model.model, X_test, y_test)
                        metrics_ltc = evaluator_ltc.calculate_metrics()

                        results.append({
                            'model': 'LTC',
                            'neurons': neurons,
                            'noise': noise,
                            'seq_len': seq_len,
                            'lr': lr,
                            'batch_size': batch_size,
                            **metrics_ltc
                        })

                        # ---------------------------
                        # 2) RNN Model
                        # ---------------------------
                        # Example: pass a custom middle layer if that's how your factory is set up
                        rnn_model = ModelFactory.create(
                            model_type='rnn',
                            config=model_config,
                            mid_layer=layers.LSTM(neurons, return_sequences=True)
                        )

                        # If the RNN class has a custom train_and_evaluate, use that:
                        # (Otherwise, just do rnn_model.model.fit(...) like LTC.)
                        history_rnn, runtime_rnn = rnn_model.train_and_evaluate(
                            X_train, y_train,
                            X_val, y_val,
                            epochs=config.epochs,
                            batch_size=batch_size
                        )
                        histories.append(history_rnn)
                        labels.append(f'RNN_n{neurons}_lr{lr}_noise{noise}')

                        # Suppose we keep final_loss as well:
                        final_val_loss = history_rnn.history['val_loss'][-1]
                        results.append({
                        'model': 'LTC',
                        'neurons': neurons,
                        'noise': noise,
                        'sequence_length': seq_len,
                        'learning_rate': lr,
                        'batch_size': batch_size,
                        #'runtime': runtime_ltc,
                        'final_val_loss': final_val_loss  # Consistent naming
                    })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Plot training curves and comparisons
    plot_training_curves(histories, labels)
    plot_model_comparison(results_df)

    return results_df, histories, labels

if __name__ == "__main__":
    config = ExperimentConfig()
    results_df, histories, labels = run_experiments(config)

    # Create results directory with today's date
    today_str = datetime.now().strftime('%Y-%m-%d')
    results_dir = os.path.join('results', today_str)
    os.makedirs(results_dir, exist_ok=True)

    # Save results to CSV
    results_df.to_csv(os.path.join(results_dir, 'experiment_results.csv'), index=False)

    # Save histories and labels for later inspection
    with open(os.path.join(results_dir, 'histories.pkl'), 'wb') as f:
        pickle.dump(histories, f)
    with open(os.path.join(results_dir, 'labels.pkl'), 'wb') as f:
        pickle.dump(labels, f)

    # Save parameters to a .txt file
    with open(os.path.join(results_dir, 'parameters.txt'), 'w') as f:
        f.write(f"Noise levels: {config.noise_levels}\n")
        f.write(f"Sequence lengths: {config.sequence_lengths}\n")
        f.write(f"Number of samples: {config.num_samples}\n")
        f.write(f"Learning rates: {config.learning_rates}\n")
        f.write(f"Number of neurons: {config.num_neurons}\n")
        f.write(f"Batch sizes: {config.batch_sizes}\n")
        f.write(f"Epochs: {config.epochs}\n")
        f.write("Models: LTC, RNN\n")
        f.write("Data type: Multiple sinusoidal data\n")

    # Optionally save plots
    # plt.savefig(os.path.join(results_dir, 'training_curves.png'))
    # plt.savefig(os.path.join(results_dir, 'model_comparison.png'))

    print("Experiment complete!")
