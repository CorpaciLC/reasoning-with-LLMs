#%%
!python -m pip install --upgrade pip
!pip install seaborn ncps
!pip install "numpy<2.0" 
!pip install tensorflow==2.10.0
!pip install networkx


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

            X_train, y_train, X_val, y_val, X_test, y_test = split_data(data_x, data_y)

            for model_name, model_creator in model_types.items():
                for learning_rate in learning_rates:
                    for batch_size in batch_sizes:
                        print(f'Running with {model_name}, lr={learning_rate}, batch_size={batch_size}, noise={noise_level}, length={length}')
                        
                        start_time = time.time()
                        
                        if model_name.startswith("LTC"):
                            for wiring_name, wiring in wiring_options.items():
                                model = model_creator(wiring)
                                model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='mean_squared_error')
                                history = train_and_evaluate_model_with_validation(model, X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size)
                                histories.append(history)
                                labels.append(f'{model_name}_{wiring_name}_lr{learning_rate}_batch{batch_size}_noise{noise_level}_length{length}')
                                
                                # Evaluate on test data
                                test_loss = model.evaluate(X_test, y_test, verbose=0)
                                test_results.append((labels[-1], test_loss))
                                
                        elif model_name.startswith("RNN"):
                            for mid_layer_name, mid_layer in mid_layer_options.items():
                                model = model_creator(mid_layer)
                                model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='mean_squared_error')
                                history = train_and_evaluate_model_with_validation(model, X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size)
                                histories.append(history)
                                labels.append(f'{model_name}_{mid_layer_name}_lr{learning_rate}_batch{batch_size}_noise{noise_level}_length{length}')
                                
                                # Evaluate on test data
                                test_loss = model.evaluate(X_test, y_test, verbose=0)
                                test_results.append((labels[-1], test_loss))
                        
                        else:
                            # Handle other model types if necessary
                            pass
                        
                        end_time = time.time()
                        runtimes.append((labels[-1], end_time - start_time))

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
    

#%%

class Wiring_with_names(wirings.Wiring):
    def __init__(self, units):
        super(Wiring_with_names, self).__init__(units)

    def get_graph(self, include_sensory_neurons=True, sensory_neurons=None, motor_neurons=None):
      """
      Returns a networkx.DiGraph object of the wiring diagram
      :param include_sensory_neurons: Whether to include the sensory neurons as nodes in the graph
      """
      if not self.is_built():
          raise ValueError(
              "Wiring is not built yet.\n"
              "This is probably because the input shape is not known yet.\n"
              "Consider calling the model.build(...) method using the shape of the inputs."
          )
      # Only import networkx package if we really need it
      import networkx as nx

      DG = nx.DiGraph()
      for i in range(self.units):
          neuron_type = self.get_type_of_neuron(i)
          if motor_neurons is None:
            DG.add_node("neuron_{:d}".format(i), neuron_type=neuron_type)
          else:
            DG.add_node(motor_neurons[i], neuron_type="motor")

      for i in range(self.input_dim):
        if sensory_neurons is None:
          DG.add_node("sensory_{:d}".format(i), neuron_type="sensory")
        else:
          DG.add_node(sensory_neurons[i], neuron_type="sensory")

      erev = self.adjacency_matrix
      sensory_erev = self.sensory_adjacency_matrix

      for src in range(self.input_dim):
          for dest in range(self.units):
              if self.sensory_adjacency_matrix[src, dest] != 0:
                  polarity = (
                      "excitatory" if sensory_erev[src, dest] >= 0.0 else "inhibitory"
                  )
                  sensory_name = "sensory_{:d}".format(src) if sensory_neurons is None else sensory_neurons[src]
                  motor_name = "neuron_{:d}".format(dest) if motor_neurons is None else motor_neurons[dest]
                  DG.add_edge(
                      "sensory_{:d}".format(src) if sensory_neurons is None else sensory_neurons[src],
                      "neuron_{:d}".format(dest) if motor_neurons is None else motor_neurons[dest],
                      polarity=polarity,
                  )

      for src in range(self.units):
          for dest in range(self.units):
              if self.adjacency_matrix[src, dest] != 0:
                  polarity = "excitatory" if erev[src, dest] >= 0.0 else "inhibitory"
                  motor_name_src = "neuron_{:d}".format(src) if motor_neurons is None else motor_neurons[src]
                  motor_name_dest = "neuron_{:d}".format(dest) if motor_neurons is None else motor_neurons[dest]
                  DG.add_edge(
                      "neuron_{:d}".format(src) if motor_neurons is None else motor_neurons[src],
                      "neuron_{:d}".format(dest) if motor_neurons is None else motor_neurons[dest],
                      polarity=polarity,
                  )
      return DG

    def draw_graph(
        self,
        layout="shell",
        neuron_colors=None,
        synapse_colors=None,
        draw_labels=False,
        sensory_neurons=None,
        motor_neurons=None
    ):
        """Draws a matplotlib graph of the wiring structure
        Examples::
            >>> import matplotlib.pyplot as plt
            >>> plt.figure(figsize=(6, 4))
            >>> legend_handles = wiring.draw_graph(draw_labels=True)
            >>> plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
            >>> plt.tight_layout()
            >>> plt.show()
        :param layout:
        :param neuron_colors:
        :param synapse_colors:
        :param draw_labels:
        :return:
        """

        # May switch to Cytoscape once support in Google Colab is available
        # https://stackoverflow.com/questions/62421021/how-do-i-install-cytoscape-on-google-colab
        import networkx as nx
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt

        if isinstance(synapse_colors, str):
            synapse_colors = {
                "excitatory": synapse_colors,
                "inhibitory": synapse_colors,
            }
        elif synapse_colors is None:
            synapse_colors = {"excitatory": "tab:green", "inhibitory": "tab:red"}

        default_colors = {
            "inter": "tab:blue",
            "motor": "tab:orange",
            "sensory": "tab:olive",
        }
        if neuron_colors is None:
            neuron_colors = {}
        # Merge default with user provided color dict
        for k, v in default_colors.items():
            if not k in neuron_colors.keys():
                neuron_colors[k] = v

        legend_patches = []
        for k, v in neuron_colors.items():
            label = "{}{} neurons".format(k[0].upper(), k[1:])
            color = v
            legend_patches.append(mpatches.Patch(color=color, label=label))

        G = self.get_graph(sensory_neurons = sensory_neurons, motor_neurons = motor_neurons)
        layouts = {
            "kamada": nx.kamada_kawai_layout,
            "circular": nx.circular_layout,
            "random": nx.random_layout,
            "shell": nx.shell_layout,
            "spring": nx.spring_layout,
            "spectral": nx.spectral_layout,
            "spiral": nx.spiral_layout,
        }
        if not layout in layouts.keys():
            raise ValueError(
                "Unknown layer '{}', use one of '{}'".format(
                    layout, str(layouts.keys())
                )
            )
        pos = layouts[layout](G)

        # Draw neurons
        for i in range(self.units):
            if motor_neurons is None:
                node_name = "neuron_{:d}".format(i)
            else:
                node_name = motor_neurons[i]
            neuron_type = G.nodes[node_name]["neuron_type"]
            neuron_color = "tab:blue"
            if neuron_type in neuron_colors.keys():
                neuron_color = neuron_colors[neuron_type]
            nx.draw_networkx_nodes(G, pos, [node_name], node_color=neuron_color)

        # Draw sensory neurons
        for i in range(self.input_dim):
            if sensory_neurons is None:
                node_name = "sensory_{:d}".format(i)
            else: 
                node_name = sensory_neurons[i]
            neuron_color = "blue"
            if "sensory" in neuron_colors.keys():
                neuron_color = neuron_colors["sensory"]
            nx.draw_networkx_nodes(G, pos, [node_name], node_color=neuron_color)

        # Optional: draw labels
        if draw_labels:
            nx.draw_networkx_labels(G, pos)

        # Draw edges
        for node1, node2, data in G.edges(data=True):
            polarity = data["polarity"]
            edge_color = synapse_colors[polarity]
            nx.draw_networkx_edges(G, pos, [(node1, node2)], edge_color=edge_color)

        return legend_patches





class Forward(Wiring_with_names):
    def __init__(self, units=1, output_dim=1):
        super(Forward, self).__init__(units)
        self.set_output_dim(output_dim)
        self.sparsity_level = 0 # Need no sparsity, all the connections are defined in the build function

    def build(self, input_shape):
        super().build(input_shape)
        self.sensory_adjacency_matrix[0,0] = 1 #idx0: source neuron index from sensory neurons, idx1: destination neuron index from units (shape: [input_dim, units])
        self.adjacency_matrix[0,0] = 1 #idx0: source neuron index from units, idx1: destination neuron index from units (shape: [units, units])




class Forward_without_loop(Wiring_with_names):
    def __init__(self, units=1, output_dim=1):
        super(Forward_without_loop, self).__init__(units)
        self.set_output_dim(output_dim)
        self.sparsity_level = 0 # Need no sparsity, all the connections are defined in the build function

    def build(self, input_shape):
        super().build(input_shape)
        self.sensory_adjacency_matrix[0,0] = 1 #idx0: source neuron index from sensory neurons, idx1: destination neuron index from units (shape: [input_dim, units])
        #self.adjacency_matrix[0,0] = 1 #idx0: source neuron index from units, idx1: destination neuron index from units (shape: [units, units])


def generate_forward_activation_on(n=100):
  N = 48 # Length of the time-series
  x_train_forward_activation_on = []
  y_train_forward_activation_on = []

  for i in range(n): 

    # Initialize the time-series data
    data_x1, data_y1 = [0] * N,  [0] * N

    # Set the signal of the input data
    # Generate signals with different start time and activation length 
    i_start1 = 2 + randrange(20)
    input_active_length1 = 2 + randrange(4)
    data_x1[i_start1:(i_start1 + input_active_length1)] = [1] * input_active_length1 # Activate signal from i_start input_active_length long

    x_train_forward_activation_on.append(data_x1)

    # Target output is a delayed activation 
    output_delay1 = 1 # Delay between the input activation and the output activation
    data_y1[(i_start1+output_delay1):] = [1] * (N-(i_start1+output_delay1)) # Activate signal from i_start + output_delay

    y_train_forward_activation_on.append(data_y1)
  
  x_train_forward_activation_on = np.stack(x_train_forward_activation_on, axis=0)
  y_train_forward_activation_on = np.stack(y_train_forward_activation_on, axis=0)

  # Print shapes
  print("x_train_forward_activation_on.shape: ", str(x_train_forward_activation_on.shape))
  print("y_train_forward_activation_on.shape: ", str(y_train_forward_activation_on.shape))

  return x_train_forward_activation_on, y_train_forward_activation_on

def generate_forward_activation_with_deactivation(n=100):
  N = 48 # Length of the time-series
  x_train_forward_activation_with_deactivation = []
  y_train_forward_activation_with_deactivation = []

  for i in range(n): 

    # Initialize the time-series data
    data_x, data_y = [0] * N,  [0] * N

    # Set the signal of the input data
    # Generate signals with different start time and activation length 
    i_start = 2 + randrange(20)
    input_active_length = 2 + randrange(4)
    data_x[i_start:(i_start + input_active_length)] = [1] * input_active_length # Activate signal from i_start input_active_length long

    x_train_forward_activation_with_deactivation.append(data_x)

    # Target output is a delayed activation 
    output_delay = 1 # Delay between the input activation and the output activation
    output_active_length = input_active_length +2

    data_y[(i_start+output_delay):(i_start+output_delay+output_active_length)] = [1] * output_active_length # Activate signal for output_active_length long

    y_train_forward_activation_with_deactivation.append(data_y)
  
  x_train_forward_activation_with_deactivation = np.stack(x_train_forward_activation_with_deactivation, axis=0)
  y_train_forward_activation_with_deactivation = np.stack(y_train_forward_activation_with_deactivation, axis=0)

  # Print shapes
  print("x_train_forward_activation_with_deactivation.shape: ", str(x_train_forward_activation_with_deactivation.shape))
  print("y_train_forward_activation_with_deactivation.shape: ", str(y_train_forward_activation_with_deactivation.shape))

  return x_train_forward_activation_with_deactivation, y_train_forward_activation_with_deactivation



#%% Experiments

#---------------------------------------------------------------------
## Config
#---------------------------------------------------------------------

## Data

noise_levels = [0, 0.1, 0.2, 0.5] # replace with [0, 0.1, 0.2, 0.5]
sequence_lengths = [24]#, 48, 96]
num_samples = 1000


## Models
learning_rates = [0.01, 0.001, 0.0001]
num_neurons = 2

batch_sizes = [8] #[1, 8]
epochs = 50

# forward_architecture = Forward(1,1) 
# forward_without_loop_architecture = Forward_without_loop(1,1) 

LTC_wiring_options = {
    'ForwardArchitecture': Forward(1,1) , 
    'ForwardsWithoutLoop':Forward_without_loop(1,1),
    'FullyConnected': wirings.FullyConnected(num_neurons, 1),
    'RandomSparsity75': wirings.Random(num_neurons, 1, sparsity_level=0.75),
    # 'AutoNCP': wirings.AutoNCP(num_neurons, 1) # todo: check what the error is
}
RNN_mid_layer_options = {
    'LSTM': layers.LSTM(num_neurons, return_sequences=True),
    # 'GRU': layers.GRU(num_neurons, return_sequences=True),
    'bi-RNN': layers.Bidirectional(layers.SimpleRNN(num_neurons, return_sequences=True)),
    'bi-LSTM': layers.Bidirectional(layers.LSTM(num_neurons, return_sequences=True)),
        
}

model_types = {
    'LTC': create_ltc_model,
    'RNN': create_rnn_model
}

#---------------------------------------------------------------------
## Config
#---------------------------------------------------------------------

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


#---------------------------------------------------------------------
## Plots & Logs
#---------------------------------------------------------------------

# Plot training curves
plot_curves(histories, labels, 'loss')

# # Save the results to a DataFrame
# results_df = pd.DataFrame({'Label': labels, 'Final Loss': [history.history['loss'][-1] for history in histories], 'Runtime (seconds)': runtimes})

# # save to file
# timestamp = time.time()
# results_df.to_csv(f'results_{timestamp}.csv', index=False)

#%%

# save histories, labels, runtimes, test_results to files
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
