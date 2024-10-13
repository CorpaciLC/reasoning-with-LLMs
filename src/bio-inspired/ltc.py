#%% Imports
# -------------------------------
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from ncps import wirings  # Assuming the ncps package
from ncps.tf import LTC
import seaborn as sns

# data generation
# -------------------------------

def generate_sine_cosine_data(length):
    data_x = np.stack([
        np.sin(np.linspace(0, 3 * np.pi, length)),
        np.cos(np.linspace(0, 3 * np.pi, length))
    ], axis=1)
    data_x = np.expand_dims(data_x, axis=0).astype(np.float32)
    data_y = np.sin(np.linspace(0, 6 * np.pi, length)).reshape([1, length, 1]).astype(np.float32)
    return data_x, data_y

def generate_noisy_data(length, noise_level=0.1):
    #todo: add noise to input and remove from output
    noise = noise_level * np.random.normal(size=(1, length, 1)).astype(np.float32)
    data_x = np.stack([
        np.sin(np.linspace(0, 3 * np.pi, length)),
        np.cos(np.linspace(0, 3 * np.pi, length))
    ], axis=1)
    data_x = np.expand_dims(data_x, axis=0).astype(np.float32)
    data_y = (np.sin(np.linspace(0, 6 * np.pi, length)).reshape([1, length, 1]).astype(np.float32)
              + noise)
    return data_x, data_y

# model creation
# -------------------------------

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

# training and evaluation pipeline
# -------------------------------

def train_and_evaluate_model(model, data_x, data_y, epochs=400, batch_size=1, verbose=1):
    history = model.fit(x=data_x, y=data_y, batch_size=batch_size, epochs=epochs, verbose=verbose)
    return history

# plotting
# -------------------------------

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

# plug-and-play pipeline
# -------------------------------

def run_pipeline(data_generators, wiring_options, mid_layer_options,sequence_lengths, model_types, epochs=100):
    histories = []
    labels = []

    for length in sequence_lengths:
        for data_generator in data_generators:
            data_x, data_y = data_generator(length)

            for model_name, model_creator in model_types.items():
                print(f'Running {data_generator.__name__} with {model_name} models of length {length}')
                if model_name.startswith("LTC"):
                    for wiring_name, wiring in wiring_options.items():
                        model = model_creator(wiring)
                        history = train_and_evaluate_model(model, data_x, data_y, epochs=epochs)
                        histories.append(history)
                        labels.append(f'{model_name}_{data_generator.__name__}_{wiring_name}_length{length}')
                else:
                    for mid_layer_name, mid_layer in mid_layer_options.items():
                        model = model_creator(mid_layer)
                        history = train_and_evaluate_model(model, data_x, data_y, epochs=epochs)
                        histories.append(history)
                        labels.append(f'{model_name}_{data_generator.__name__}_{mid_layer_name}_length{length}')

            sns.set()
            plt.figure(figsize=(6, 4))
            plt.plot(data_x[0, :, 0], label="Input feature 1")
            plt.plot(data_x[0, :, 1], label="Input feature 1")
            plt.plot(data_y[0, :, 0], label="Target output")
            plt.ylim((-1, 1))
            plt.title("Training data")
            plt.legend(loc="upper right")
            plt.show()
    return histories, labels
    plot_training_histories(histories, labels)

# wiring options for LTC
LTC_wiring_options = {
    'FullyConnected': wirings.FullyConnected(8, 1),
    'RandomSparsity75': wirings.Random(8, 1, sparsity_level=0.75),
    'AutoNCP': wirings.AutoNCP(8, 1)
}
# mid layer options for RNN
RNN_mid_layer_options = {
    'LSTM': layers.LSTM(8, return_sequences=True),
    # 'GRU': layers.GRU(8, return_sequences=True),
    'bi-RNN': layers.Bidirectional(layers.SimpleRNN(8, return_sequences=True)),
    'bi-LSTM': layers.Bidirectional(layers.LSTM(8, return_sequences=True)),
        
}
#  data generators
data_generators = [generate_sine_cosine_data, generate_noisy_data]

# model types
model_types = {
    'LTC': create_ltc_model,
    'RNN': create_rnn_model
}

# sequence lengths to test
sequence_lengths = [24, 48]#, 96]



# run the pipeline
histories, labels = run_pipeline(data_generators, LTC_wiring_options, RNN_mid_layer_options, sequence_lengths, model_types, epochs=10) 

plot_curves(histories, labels, 'loss')

# %%
