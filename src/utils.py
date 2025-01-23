
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ncps import wirings
from ncps.tf import LTC

from typing import Tuple, Optional
from tensorflow.keras import layers, models, optimizers

import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
import pickle

def generate_multiple_sinusoidal_data(num_samples: int, length: int, 
                                      noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate multiple samples of sine+cosine inputs and sine-based outputs 
    (with some noise added to the outputs).
    """
    data_x_list = []
    data_y_list = []
    
    for _ in range(num_samples):
        # Base signals
        sin_wave = np.sin(np.linspace(0, 3 * np.pi, length))
        cos_wave = np.cos(np.linspace(0, 3 * np.pi, length))
        
        # Shape: (1, length, 2)
        input_data = np.stack([sin_wave, cos_wave], axis=1)
        input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
        
        # Target signal
        output_data = np.sin(np.linspace(0, 6 * np.pi, length))
        output_data = output_data.reshape(1, length, 1).astype(np.float32)
        
        # Random noise for target
        noise = noise_level * np.random.normal(size=(1, length, 1)).astype(np.float32)
        output_data_noisy = output_data + noise
        
        data_x_list.append(input_data)
        data_y_list.append(output_data_noisy)
    
    # Stack up to shape: (num_samples, length, 2) and (num_samples, length, 1)
    data_x = np.vstack(data_x_list)
    data_y = np.vstack(data_y_list)
    return data_x, data_y


def generate_sinusoidal_sequences(num_samples: int, length: int, noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sequences where each new sequence starts at a different phase point.
    Returns data suitable for next-step prediction.
    """
    data_x_list = []
    data_y_list = []
    
    for i in range(num_samples):
        # Random phase shift for each sequence
        phase_shift = np.random.uniform(0, 2 * np.pi)
        t = np.linspace(0, 4 * np.pi, length + 1)  # +1 to have target value
        
        # Input features: sin(t + phase) and cos(t + phase)
        sin_wave = np.sin(t[:-1] + phase_shift)
        cos_wave = np.cos(t[:-1] + phase_shift)
        
        # Target: next value in sequence
        target = np.sin(t[1:] + phase_shift)
        
        # Add noise to target
        if noise_level > 0:
            target += np.random.normal(0, noise_level, size=target.shape)
        
        # Shape: (length, 2) for input, (length, 1) for output
        input_data = np.stack([sin_wave, cos_wave], axis=1)
        target_data = target.reshape(-1, 1)
        
        data_x_list.append(input_data)
        data_y_list.append(target_data)
    
    return np.array(data_x_list), np.array(data_y_list)

def generate_chaotic_sequences(num_samples: int, length: int, noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sequences using the Lorenz system, which provides more complex patterns.
    Good for testing prediction capabilities on chaotic systems.
    """
    def lorenz(x, y, z, s=10, r=28, b=2.667):
        x_dot = s*(y - x)
        y_dot = r*x - y - x*z
        z_dot = x*y - b*z
        return x_dot, y_dot, z_dot
    
    dt = 0.01
    data_x_list = []
    data_y_list = []
    
    for _ in range(num_samples):
        # Random starting points
        x, y, z = np.random.uniform(-15, 15, 3)
        points = []
        
        # Generate sequence
        for _ in range(length + 1):  # +1 to have target value
            dx, dy, dz = lorenz(x, y, z)
            x += dx * dt
            y += dy * dt
            z += dz * dt
            points.append([x, y])
        
        points = np.array(points)
        
        # Input: current point
        input_data = points[:-1]
        # Target: next point's x-coordinate
        target_data = points[1:, 0:1]  # Only x-coordinate
        
        if noise_level > 0:
            target_data += np.random.normal(0, noise_level, size=target_data.shape)
            
        data_x_list.append(input_data)
        data_y_list.append(target_data)
    
    return np.array(data_x_list), np.array(data_y_list)

def generate_mixed_frequency_sequences(num_samples: int, length: int, noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sequences with multiple frequency components.
    Tests ability to capture both short-term and long-term patterns.
    """
    data_x_list = []
    data_y_list = []
    
    for _ in range(num_samples):
        t = np.linspace(0, 6 * np.pi, length + 1)  # +1 to have target value
        
        # Random amplitudes and frequencies
        a1, a2 = np.random.uniform(0.3, 1.0, 2)
        f1, f2 = np.random.uniform(0.5, 2.0, 2)
        
        # Generate signal with two frequency components
        signal = (a1 * np.sin(f1 * t) + 
                 a2 * np.sin(f2 * t))
        
        # Input features: current value and derivative
        input_data = np.stack([
            signal[:-1],  # Current value
            np.gradient(signal)[:-1]  # Rate of change
        ], axis=1)
        
        # Target: next value
        target_data = signal[1:].reshape(-1, 1)
        
        if noise_level > 0:
            target_data += np.random.normal(0, noise_level, size=target_data.shape)
            
        data_x_list.append(input_data)
        data_y_list.append(target_data)
    
    return np.array(data_x_list), np.array(data_y_list)

def split_data(data_x: np.ndarray, data_y: np.ndarray, test_size: float = 0.2, val_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_temp, y_train, y_temp = train_test_split(data_x, data_y, test_size=(test_size + val_size))
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(test_size / (test_size + val_size)))
    return X_train, y_train, X_val, y_val, X_test, y_test


#%% models

def compile_model(model: models.Sequential, learning_rate: float, loss: str, gradient_clip: float) -> models.Sequential:
    optimizer = optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=gradient_clip
    )
    model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])
    return model

def create_ltc_model(wiring: object, activation: str = 'linear', loss: str = 'mean_squared_error', input_shape: Tuple[Optional[int], int] = (None, 2), learning_rate: float = 0.01, dropout_rate: float = 0.2, gradient_clip: float = 1.0) -> models.Sequential:
    try:
        model = models.Sequential([
            layers.InputLayer(input_shape=input_shape),
            LTC(wiring, return_sequences=True),
            layers.Dense(1, activation=activation)
        ])
        return compile_model(model, learning_rate, loss, gradient_clip)
    except Exception as e:
        print(f"Error creating LTC model: {e}")
        return None

def create_rnn_model(mid_layer: layers.Layer, activation: str = 'linear', loss: str = 'mean_squared_error', input_shape: Tuple[Optional[int], int] = (None, 2), learning_rate: float = 0.01, dropout_rate: float = 0.2, gradient_clip: float = 1.0) -> models.Sequential:
    try:
        model = models.Sequential([
            layers.InputLayer(input_shape=input_shape),
            mid_layer,
            layers.Dense(1, activation=activation)
        ])
        return compile_model(model, learning_rate, loss, gradient_clip)
    except Exception as e:
        print(f"Error creating RNN model: {e}")
        return None


def train_and_evaluate_model_with_validation(model: models.Sequential, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, epochs: int = 400, batch_size: int = 1, verbose: int = 1) -> Tuple[models.Sequential, float]:
    start_time = time.time()  # Start time tracking
    history = model.fit(
        x=train_x,
        y=train_y,
        validation_data=(val_x, val_y),
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose
    )
    end_time = time.time()  # End time tracking
    runtime = end_time - start_time

    return history, runtime

def save_intermediate_results(result: dict):
    """Save results after each model training to prevent data loss"""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"results_{timestamp}.json"

    # Create a clean version of the result dict (remove non-serializable items)
    clean_result = {
        k: v for k, v in result.items()
        if k not in ['model_info', 'training_history']
    }

    # Save the main results
    with open(filename, 'a') as f:
        json.dump(clean_result, f)
        f.write('\n')

    # Save the full training history separately
    history_file = f"history_{timestamp}.pkl"
    with open(history_file, 'wb') as f:
        pickle.dump({
            'model_info': result['model_info'],
            'training_history': result['training_history']
        }, f)


#%% plotting
def plot_curves(histories: list, labels: list, metric: str = 'loss'):
    plt.figure(figsize=(10, 6))

    unique_models = set(label.split('_')[0] for label in labels)
    colors = plt.cm.get_cmap('coolwarm', len(unique_models))
    model_color_map = {model: colors(i) for i, model in enumerate(unique_models)}

    # Plot the loss curves
    for history, label in zip(histories, labels):
        model_name = label.split('_')[0]
        plt.plot(history.history[metric], label=label, color=model_color_map[model_name], alpha=0.4)

    plt.xlabel('Epochs')
    plt.ylabel(f'{metric.capitalize()})')
    plt.title(f'{metric.capitalize()} Curves')
    plt.legend()
    plt.show()


def plot_final_losses_vs_params(results_df: pd.DataFrame):
    plt.figure(figsize=(10, 6))

    # Plot final loss vs different parameters
    sns.barplot(x='Label', y='Final Loss', data=results_df)
    plt.xticks(rotation=90)
    plt.title('Final Losses for Different Models and Parameters')
    plt.show()

    # Plot runtime vs different parameters
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Label', y='Runtime (seconds)', data=results_df)
    plt.xticks(rotation=90)
    plt.title('Runtime for Different Models and Parameters')
    plt.show()


#### Extract NN info
def extract_nn_info(model: models.Sequential, model_name: str, detailed: bool = False) -> Tuple[dict, str]:
    
    """Extract and summarize information about a neural network model."""
    info = {
        "model_name": model_name,
        "total_layers": len(model.layers),
        "total_parameters": model.count_params(),
        "trainable_parameters": sum([layer.count_params() for layer in model.layers if len(layer.trainable_weights) > 0]),
        "non_trainable_parameters": sum([layer.count_params() for layer in model.layers if len(layer.non_trainable_weights) > 0]),
        "layers": [],
        "connectivity": {
            "input_shape": str(model.input_shape),
            "output_shape": str(model.output_shape),
        },
        "memory_estimate_mb": sum([
            sum([w.numpy().nbytes for w in layer.weights])
            for layer in model.layers
        ]) / (1024 * 1024),
    }

    # Layer Details
    for idx, layer in enumerate(model.layers):
        layer_info = {
            "index": idx,
            "name": layer.name,
            "type": type(layer).__name__,
            "output_shape": str(layer.output_shape),
            "activation": layer.activation.__name__ if hasattr(layer, 'activation') else None,
            "parameters": layer.count_params(),
            "trainable": layer.trainable,
        }

        if detailed:
            weights = layer.get_weights()
            if weights:
                layer_info["weights"] = []
                for i, w in enumerate(weights):
                    weight_info = {
                        "shape": w.shape,
                        "size": w.size,
                        "min": float(w.min()),
                        "max": float(w.max()),
                        "mean": float(w.mean()),
                        "std": float(w.std()),
                        "sparsity": float((w == 0).sum() / w.size),
                    }
                    layer_info["weights"].append(weight_info)

        info["layers"].append(layer_info)

    # Generate Textual Summary
    summary = f"""
    Model Name: {model_name}
    Total Layers: {info['total_layers']}
    Total Parameters: {info['total_parameters']:,} ({info['trainable_parameters']:,} trainable)
    Input Shape: {info['connectivity']['input_shape']}
    Output Shape: {info['connectivity']['output_shape']}
    Memory Estimate: {info['memory_estimate_mb']:.2f} MB
    """
    if detailed:
        for layer in info["layers"]:
            summary += f"""
            • {layer['type']} (Layer {layer['index']}):
            - Name: {layer['name']}
            - Shape: {layer['output_shape']}
            - Parameters: {layer['parameters']:,}
            - Activation: {layer['activation']}
            """

    # # Add Reasoning Prompts
    # summary += """
    # **Reasoning Prompts**
    # 1. How do the activation functions affect the model's learning behavior?
    # 2. What role do the number of layers and parameters play in generalization and overfitting?
    # 3. Are there specific risks associated with the optimizer or loss function settings?
    # 4. What insights can be drawn from the memory estimate and parameter count?
    # """

    return info, summary



#### Extact NN info - extras
def extract_nn_info_gemini(model, model_name):
    print(f"\nModel: {model_name}")

    # Structure
    print("\n**Structure**")
    print("Number of Layers:", len(model.layers))
    for i, layer in enumerate(model.layers):
        print(f"\nLayer {i+1}:")  # Added layer number for easier reference
        print(f"  Layer Name: {layer.name}")
        print(f"  Type: {type(layer).__name__}")
        print(f"  Output Shape: {layer.output_shape}")
        print(f"  Trainable: {layer.trainable}")  # Added trainable status

        # More details for specific layer types
        if isinstance(layer, Dense):
            print(f"  Units: {layer.units}")
            print(f"  Use Bias: {layer.use_bias}")
            print(f"  Kernel Regularizer: {layer.kernel_regularizer}")
            print(f"  Bias Regularizer: {layer.bias_regularizer}")
        elif isinstance(layer, LSTM):
            print(f"  Units: {layer.units}")
            print(f"  Return Sequences: {layer.return_sequences}")
            print(f"  Recurrent Activation: {layer.recurrent_activation.__name__}")
            print(f"  Dropout: {layer.dropout}")
            print(f"  Recurrent Dropout: {layer.recurrent_dropout}")

    # Weights and Connections (unchanged, but could be expanded)
    print("\nWeights and Connections:")
    for layer in model.layers:
        weights = layer.get_weights()
        if weights:
            print(f"Layer {layer.name} Weights:")
            for i, w in enumerate(weights):
                print(f"  Weight {i}: Shape {w.shape}")

    # Function
    print("\n**Function**")
    activation_functions = [layer.activation.__name__ for layer in model.layers if hasattr(layer, 'activation')]
    print("Activation Functions:", activation_functions)

    # Optimizer (added)
    print("\nOptimizer:")
    print(f"  Name: {type(model.optimizer).__name__}")
    # You can add more optimizer details here if needed

    # Loss Function (added)
    print("\nLoss Function:")
    print(f"  Name: {model.loss}")

    # Metrics (added)
    print("\nMetrics:")
    print(f"  Metrics: {model.metrics_names}")





def extract_nn_info_copilot(model, model_name):
    info = {
        "model_name": model_name,
        "total_layers": len(model.layers),
        "total_parameters": model.count_params(),
        "trainable_parameters": sum([layer.count_params() for layer in model.layers if len(layer.trainable_weights) > 0]),
        "non_trainable_parameters": sum([layer.count_params() for layer in model.layers if len(layer.non_trainable_weights) > 0]),
        "layers": [],
        "connectivity": {
            "input_shape": str(model.input_shape),
            "output_shape": str(model.output_shape),
        },
        "memory_estimate_mb": sum([
            sum([w.numpy().nbytes for w in layer.weights])
            for layer in model.layers
        ]) / (1024 * 1024),
    }

    # Detailed layer analysis
    for idx, layer in enumerate(model.layers):
        layer_info = {
            "index": idx,
            "name": layer.name,
            "type": type(layer).__name__,
            "output_shape": str(layer.output_shape),
            "activation": layer.activation.__name__ if hasattr(layer, 'activation') else None,
            "parameters": layer.count_params(),
            "trainable": layer.trainable,
        }

        # Get weights information
        weights = layer.get_weights()
        if weights:
            layer_info["weights"] = []
            for i, w in enumerate(weights):
                weight_info = {
                    "shape": w.shape,
                    "size": w.size,
                    "min": float(w.min()),
                    "max": float(w.max()),
                    "mean": float(w.mean()),
                    "std": float(w.std()),
                    "sparsity": float((w == 0).sum() / w.size),
                }
                layer_info["weights"].append(weight_info)

        # Special handling for different layer types
        if isinstance(layer, Dense):
            layer_info["units"] = layer.units
            layer_info["use_bias"] = layer.use_bias

        elif isinstance(layer, LSTM):
            layer_info["units"] = layer.units
            layer_info["return_sequences"] = layer.return_sequences
            layer_info["stateful"] = layer.stateful

        elif "LTC" in str(type(layer)):  # Handle LTC layers
            if hasattr(layer, "wiring"):
                layer_info["neurons"] = layer.wiring.units
                layer_info["sensory_neurons"] = layer.wiring.sensory_neurons
                layer_info["inter_neurons"] = layer.wiring.inter_neurons
                layer_info["motor_neurons"] = layer.wiring.motor_neurons
                layer_info["sparsity_level"] = 1 - (layer.wiring.adjacency_matrix.sum() /
                                                  (layer.wiring.adjacency_matrix.shape[0] *
                                                   layer.wiring.adjacency_matrix.shape[1]))

        info["layers"].append(layer_info)

    # Generate a text summary for LLM prompts
    summary_text = f"""
        Model: {model_name}
        Total Parameters: {info['total_parameters']:,} ({info['trainable_parameters']:,} trainable)
        Architecture Summary:
        - Input Shape: {info['connectivity']['input_shape']}
        - Total Layers: {info['total_layers']}
        - Memory Estimate: {info['memory_estimate_mb']:.2f} MB
        - Output Shape: {info['connectivity']['output_shape']}
        Layer Details:
        """
    for layer in info["layers"]:
        summary_text += f"""
        • {layer['type']} (Layer {layer['index']}):
        - Name: {layer['name']}
        - Shape: {layer['output_shape']}
        - Parameters: {layer['parameters']:,}
        - Activation: {layer['activation']}
        """
        if "weights" in layer:
            summary_text += f"  - Weight Stats: Mean={layer['weights'][0]['mean']:.4f}, Std={layer['weights'][0]['std']:.4f}, Sparsity={layer['weights'][0]['sparsity']:.2%}\n"

    info["llm_summary"] = summary_text

    return info


