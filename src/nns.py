import time

import numpy as np
import pandas as pd
import tensorflow.keras as keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from ncps import wirings  
from ncps.tf import LTC
import seaborn as sns


#%%
# data generation
def generate_sine_cosine_data(length):
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
        np.sin(np.linspace(0, 3 * np.pi, length)),
        np.cos(np.linspace(0, 3 * np.pi, length))
    ], axis=1)
    data_x = np.expand_dims(data_x, axis=0).astype(np.float32)
    data_y = (np.sin(np.linspace(0, 6 * np.pi, length)).reshape([1, length, 1]).astype(np.float32)
              + noise)
    return data_x, data_y


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

def run_pipeline_extended(data_generators, wiring_options, mid_layer_options, sequence_lengths, model_types, learning_rates, batch_sizes, noise_levels, epochs=100):
    histories = []
    labels = []
    runtimes = []  # To track runtime for each configuration
    
    for length in sequence_lengths:
        for noise_level in noise_levels:
            for data_generator in data_generators:
                data_x, data_y = data_generator(length)
                
                for model_name, model_creator in model_types.items():
                    for learning_rate in learning_rates:
                        for batch_size in batch_sizes:
                            print(f'Running {data_generator.__name__} with {model_name}, lr={learning_rate}, batch_size={batch_size}, noise={noise_level}, length={length}')
                            if model_name.startswith("LTC"):
                                for wiring_name, wiring in wiring_options.items():
                                    model = model_creator(wiring)
                                    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='mean_squared_error')
                                    history, runtime = train_and_evaluate_model_with_runtime(model, data_x, data_y, epochs=epochs, batch_size=batch_size)
                                    histories.append(history)
                                    labels.append(f'{model_name}_{data_generator.__name__}_{wiring_name}_lr{learning_rate}_batch{batch_size}_noise{noise_level}_length{length}')
                                    runtimes.append(runtime)
                            else:
                                for mid_layer_name, mid_layer in mid_layer_options.items():
                                    model = model_creator(mid_layer)
                                    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='mean_squared_error')
                                    history, runtime = train_and_evaluate_model_with_runtime(model, data_x, data_y, epochs=epochs, batch_size=batch_size)
                                    histories.append(history)
                                    labels.append(f'{model_name}_{data_generator.__name__}_{mid_layer_name}_lr{learning_rate}_batch{batch_size}_noise{noise_level}_length{length}')
                                    runtimes.append(runtime)

    return histories, labels, runtimes


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

#---------------------------------------------------------------------
## Config
#---------------------------------------------------------------------

# Learning rates to test
learning_rates = [0.01, 0.001, 0.0001]

# Batch sizes to test
batch_sizes = [1]#, 8, 16]

# Noise levels to test (for noisy data generation)
noise_levels = [0.1, 0.2, 0.5]

# Sequence lengths to test
sequence_lengths = [24]#, 48, 96]

# number neurons
num_neurons = 8

# wiring options for LTC
LTC_wiring_options = {
    'FullyConnected': wirings.FullyConnected(num_neurons, 1),
    'RandomSparsity75': wirings.Random(num_neurons, 1, sparsity_level=0.75),
    'AutoNCP': wirings.AutoNCP(num_neurons, 1)
}
# mid layer options for RNN
RNN_mid_layer_options = {
    'LSTM': layers.LSTM(num_neurons, return_sequences=True),
    # 'GRU': layers.GRU(num_neurons, return_sequences=True),
    'bi-RNN': layers.Bidirectional(layers.SimpleRNN(num_neurons, return_sequences=True)),
    'bi-LSTM': layers.Bidirectional(layers.LSTM(num_neurons, return_sequences=True)),
        
}
#  data generators
data_generators = [generate_sine_cosine_data, generate_noisy_data]

# model types
model_types = {
    'LTC': create_ltc_model,
    'RNN': create_rnn_model
}

#---------------------------------------------------------------------
## Config
#---------------------------------------------------------------------

# Run the extended pipeline
histories, labels, runtimes = run_pipeline_extended(
    data_generators, 
    LTC_wiring_options, 
    RNN_mid_layer_options, 
    sequence_lengths, 
    model_types, 
    learning_rates, 
    batch_sizes, 
    noise_levels, 
    epochs=10
)



#---------------------------------------------------------------------
## Plots & Logs
#---------------------------------------------------------------------

# Plot training curves
plot_curves(histories, labels, 'loss')

# Save the results to a DataFrame
results_df = pd.DataFrame({'Label': labels, 'Final Loss': [history.history['loss'][-1] for history in histories], 'Runtime (seconds)': runtimes})

# save to file
timestamp = time.time()
results_df.to_csv(f'outputs/results_{timestamp}.csv', index=False)

#%%