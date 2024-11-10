# #%%
# !python -m pip install --upgrade pip
# !pip install seaborn ncps
# !pip install "numpy<2.0" 
# !pip install tensorflow==2.10.0
# !pip install networkx


#%%
import time
import os

import numpy as np
import pandas as pd
import tensorflow.keras as keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from ncps import wirings  
from ncps.tf import LTC
import seaborn as sns
from random import randrange

from sklearn.model_selection import train_test_split

results_file = 'C:\\Users\\lcorpaci\\Documents\\Thesis\\reasoning-with-LLMs\\src\\results.txt'

#%%
# data generation
def generate_sinusoidal_data(length):
    data_x = np.stack([
        np.sin(np.linspace(0, 3 * np.pi, length)),
        np.cos(np.linspace(0, 3 * np.pi, length))
    ], axis=1)
    data_x = np.expand_dims(data_x, axis=0).astype(np.float32)
    data_y = np.sin(np.linspace(0, 6 * np.pi, length)).reshape([1, length, 1]).astype(np.float32)
    return data_x, data_y

def generate_noisy_data(length, noise_level=0.1):
    noise = noise_level * np.random.normal(size=(1, length, 1)).astype(np.float32)
    data_x = np.stack([
        np.sin(np.linspace(0, 3 * np.pi, length) + noise),
        np.cos(np.linspace(0, 3 * np.pi, length)) + noise
    ], axis=1)
    data_x = np.expand_dims(data_x, axis=0).astype(np.float32)
    data_y = (np.sin(np.linspace(0, 6 * np.pi, length)).reshape([1, length, 1]).astype(np.float32))
    return data_x, data_y



# Updated data generation function
def generate_multiple_sinusoidal_data(num_samples, length, noise_level=0.1):
    data_x = []
    data_y = []
    
    for _ in range(num_samples):
        # Generate base sine and cosine waves
        sine_wave = np.sin(np.linspace(0, 3 * np.pi, length))
        cosine_wave = np.cos(np.linspace(0, 3 * np.pi, length))

        # Stack sine and cosine together
        input_data = np.stack([sine_wave, cosine_wave], axis=1)
        
        # Expand dims to match LSTM input
        input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
        
        # Generate target output and add random noise
        output_data = np.sin(np.linspace(0, 6 * np.pi, length)).reshape([1, length, 1]).astype(np.float32)
        noise = noise_level * np.random.normal(size=(1, length, 1)).astype(np.float32)
        output_data_noisy = output_data + noise

        data_x.append(input_data)
        data_y.append(output_data_noisy)
    
    # Stack all samples together to create the final dataset
    data_x = np.vstack(data_x)
    data_y = np.vstack(data_y)
    return data_x, data_y


# Split the generated data into train, validation, and test sets
def split_data(data_x, data_y, train_ratio=0.7, validation_ratio=0.15, test_ratio=0.15):
    # Initial train-test split
    X_train, X_temp, y_train, y_temp = train_test_split(data_x, data_y, test_size=(1 - train_ratio), random_state=42)

    # Split the remaining data into validation and test
    validation_size = test_ratio / (validation_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=validation_size, random_state=42)

    return X_train, y_train, X_val, y_val, X_test, y_test


#%% model creation
def create_ltc_model(wiring, input_shape=(None, 2)):
    model = keras.models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        LTC(wiring, return_sequences=True),
    ])
    model.compile(optimizer=keras.optimizers.Adam(0.01), loss='mean_squared_error')
    return model

def create_rnn_model(mid_layer, input_shape=(None, 2)):
    model = keras.models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        mid_layer,
        layers.Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.Adam(0.01), loss='mean_squared_error')
    return model

def train_and_evaluate_model_with_runtime(model, data_x, data_y, epochs=400, batch_size=1, verbose=1):
    start_time = time.time()  # Start time tracking
    history = model.fit(x=data_x, y=data_y, batch_size=batch_size, epochs=epochs, verbose=verbose)
    end_time = time.time()  # End time tracking
    runtime = end_time - start_time
    return history, runtime

def train_and_evaluate_model_with_validation(model, train_x, train_y, val_x, val_y, epochs=400, batch_size=1, verbose=1):
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


def run_pipeline_with_data_splits(num_samples, sequence_lengths, wiring_options, mid_layer_options, model_types, learning_rates, batch_sizes, noise_levels, epochs=100):
    histories = []
    labels = []
    runtimes = []  # To track runtime for each configuration
    test_results = []  # To track test results for each configuration

    for length in sequence_lengths:
        for noise_level in noise_levels:
            # Generate multiple data samples
            data_x, data_y = generate_multiple_sinusoidal_data(num_samples, length, noise_level)

            # Split data into train, validation, and test sets
            X_train, y_train, X_val, y_val, X_test, y_test = split_data(data_x, data_y)

            for model_name, model_creator in model_types.items():
                for learning_rate in learning_rates:
                    for batch_size in batch_sizes:                        
                        if model_name.startswith("LTC"):
                            for wiring_name, wiring in wiring_options.items():
                                model = model_creator(wiring)
                                model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='mean_squared_error')
                                history, runtime = train_and_evaluate_model_with_validation(model, X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size)
                                histories.append(history)
                                label = f'{model_name}_{wiring_name}_lr{learning_rate}_batch{batch_size}_noise{noise_level}_length{length}'
                                labels.append(label)
                                runtimes.append((label, runtime))
                                
                                # Evaluate on test data
                                test_loss = model.evaluate(X_test, y_test, verbose=0)
                                test_results.append((label, test_loss))

                                print(label, test_loss, history.history["val_loss"][-1])
                                with open(results_file, 'a') as f:                                    
                                    # Write the results
                                    val_loss = history.history['val_loss'][-1] if 'val_loss' in history.history else 'N/A'
                                    epoch_losses = ','.join(map(str, history.history['loss']))
                                    f.write(f'{label},{test_loss},{val_loss},{epoch_losses}\n')
                                
                        elif model_name.startswith("RNN"):
                            for mid_layer_name, mid_layer in mid_layer_options.items():
                                model = model_creator(mid_layer)
                                model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='mean_squared_error')
                                history, runtime = train_and_evaluate_model_with_validation(model, X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size)
                                histories.append(history)
                                label = f'{model_name}_{mid_layer_name}_lr{learning_rate}_batch{batch_size}_noise{noise_level}_length{length}'
                                labels.append(label)
                                runtimes.append((label, runtime))
                                
                                # Evaluate on test data
                                test_loss = model.evaluate(X_test, y_test, verbose=0)
                                test_results.append((label, test_loss))

                                print(label, test_loss, history.history["val_loss"][-1])
                                with open(results_file, 'a') as f:                                    
                                    # Write the results
                                    val_loss = history.history['val_loss'][-1] if 'val_loss' in history.history else 'N/A'
                                    epoch_losses = ','.join(map(str, history.history['loss']))
                                    f.write(f'{label},{test_loss},{val_loss},{epoch_losses}\n')
                        else:
                            pass
                       

    return histories, labels, runtimes, test_results

#%% plotting
def plot_curves(histories, labels, metric='loss'):
    plt.figure(figsize=(10, 6))
    
    unique_models = set(label.split('_')[0] for label in labels)
    colors = plt.cm.get_cmap('coolwarm', len(unique_models))
    model_color_map = {model: colors(i) for i, model in enumerate(unique_models)}
    
    # Plot the loss curves
    for history, label in zip(histories, labels):
        model_name = label.split('_')[0] 
        plt.plot(history.history[metric], label=label, color=model_color_map[model_name], alpha=0.4)
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curves')
    plt.legend()
    plt.show()


def plot_final_losses_vs_params(results_df):
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


#%% Experiments

noise_levels = [0.1, 0.2, 0.5] # replace with [0, 0.1, 0.2, 0.5]
sequence_lengths = [24]#, 48, 96]
num_samples = 1000


## Models
learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
num_neurons = [4, 8, 16, 32, 64]

batch_sizes = [10] #[1, 8]
epochs = 25

LTC_wiring_options = {}
RNN_mid_layer_options = {}

for n_neurons in num_neurons:
    LTC_wiring_options[f'AutoNCP_{n_neurons}'] = wirings.AutoNCP(n_neurons, 1)
    LTC_wiring_options[f'RandomSparsity75_{n_neurons}'] = wirings.Random(n_neurons, 1, sparsity_level=0.75)
    LTC_wiring_options[f'FullyConnected_{n_neurons}'] = wirings.FullyConnected(n_neurons, 1)

    RNN_mid_layer_options[f'LSTM_{n_neurons}'] = layers.LSTM(n_neurons, return_sequences=True)
    RNN_mid_layer_options[f'bi-RNN_{n_neurons}'] = layers.Bidirectional(layers.SimpleRNN(n_neurons, return_sequences=True))
    RNN_mid_layer_options[f'bi-LSTM_{n_neurons}'] = layers.Bidirectional(layers.LSTM(n_neurons, return_sequences=True))

model_types = {
    'LTC': create_ltc_model,
    'RNN': create_rnn_model
}

# Run the extended pipeline
histories, labels, runtimes, test_results = run_pipeline_with_data_splits(
    num_samples=num_samples,
    sequence_lengths=sequence_lengths,
    wiring_options=LTC_wiring_options,
    mid_layer_options=RNN_mid_layer_options,
    model_types=model_types,
    learning_rates=learning_rates,
    batch_sizes=batch_sizes,
    noise_levels=noise_levels,
    epochs=epochs
)

plot_curves(histories, labels, 'loss')


#%%

import pickle

timestamp = time.time()

with open(f'histories_{timestamp}.pkl', 'wb') as f:
    pickle.dump(histories, f)

with open(f'labels_{timestamp}.pkl', 'wb') as f:
    pickle.dump(labels, f)

with open(f'runtimes_{timestamp}.pkl', 'wb') as f:
    pickle.dump(runtimes, f)

with open(f'test_results_{timestamp}.pkl', 'wb') as f:
    pickle.dump(test_results, f)

# %%
