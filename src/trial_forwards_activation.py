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

from sklearn.model_selection import train_test_split
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


#%%
# Train/ validation/ test data

x_train_forward_activation_on, y_train_forward_activation_on = generate_forward_activation_on(100)
x_valid_forward_activation_on, y_valid_forward_activation_on = generate_forward_activation_on(20)
x_test_forward_activation_on, y_test_forward_activation_on = generate_forward_activation_on(10)


x_train_forward_activation_with_deactivation, y_train_forward_activation_with_deactivation = generate_forward_activation_with_deactivation(100)
x_valid_forward_activation_with_deactivation, y_valid_forward_activation_with_deactivation = generate_forward_activation_with_deactivation(20)
x_test_forward_activation_with_deactivation, y_test_forward_activation_with_deactivation = generate_forward_activation_with_deactivation(10)

#%% Forward architecture
forward_architecture = Forward(1,1) 

ncp_model = keras.models.Sequential(
    [
        keras.layers.InputLayer(input_shape=(None, 1)),
        LTC(forward_architecture, return_sequences=True, input_mapping=None, output_mapping=None),
    ]
)
ncp_model.compile(
    optimizer=keras.optimizers.Adam(0.01), loss='mean_squared_error'
)


hist_forward_ncp = ncp_model.fit(x=x_train_forward_activation_on, y=y_train_forward_activation_on, batch_size=32, epochs=100, validation_data=(x_valid_forward_activation_on, y_valid_forward_activation_on))


# Plotting the history data
sns.set()
plt.figure(figsize=(6, 4))
plt.plot(hist_forward_ncp.history["loss"], label="Training loss")
plt.plot(hist_forward_ncp.history["val_loss"], label="Validation loss")

plt.legend(loc="upper right")
plt.xlabel("Training steps")
plt.ylabel("Loss")
plt.show()


#%% Forward without loop architecture

forward_without_loop_architecture = Forward_without_loop(1,1) 

ncp_model2 = keras.models.Sequential(
    [
        keras.layers.InputLayer(input_shape=(None, 1)),
        LTC(forward_without_loop_architecture, return_sequences=True, input_mapping=None, output_mapping=None),
    ]
)
ncp_model2.compile(
    optimizer=keras.optimizers.Adam(0.01), loss='mean_squared_error'
)

hist_forward_ncp2 = ncp_model2.fit(x=x_train_forward_activation_on, y=y_train_forward_activation_on, batch_size=32, epochs=200, validation_data=(x_valid_forward_activation_on, y_valid_forward_activation_on))

# Plotting the history data
sns.set()
plt.figure(figsize=(6, 4))
plt.plot(hist_forward_ncp2.history["loss"], label="Training loss")
plt.plot(hist_forward_ncp2.history["val_loss"], label="Validation loss")

plt.legend(loc="upper right")
plt.xlabel("Training steps")
plt.ylabel("Loss")
plt.show()

y_test_forward_activation_on_predicted2 = ncp_model2.predict(x_test_forward_activation_on)

# Plotting the test data
plt.figure(figsize=(8, 4))
fig, axs = plt.subplots(3,1)
axs[0].plot(x_test_forward_activation_on[0, :], 'tab:red')
axs[0].set_title('PLM')
axs[1].plot(y_test_forward_activation_on[0, :], 'tab:blue')
axs[1].set_title('AVB true')
axs[2].plot(np.squeeze(y_test_forward_activation_on_predicted2[0, :]), 'tab:green')
axs[2].set_title('AVB predicted')


plt.tight_layout()
plt.show()

#%% Forward activation with deactivation

forward_architecture = Forward(1,1) 

ncp_model_with_deactivation = keras.models.Sequential(
    [
        keras.layers.InputLayer(input_shape=(None, 1)),
        LTC(forward_architecture, return_sequences=True, input_mapping=None, output_mapping=None),
    ]
)
ncp_model_with_deactivation.compile(
    optimizer=keras.optimizers.Adam(0.01), loss='mean_squared_error'
)
sns.set_style("white")
plt.figure(figsize=(6, 4))
legend_handles = forward_architecture.draw_graph(draw_labels=True)
plt.legend(handles=legend_handles, bbox_to_anchor=(1.1, 1.1))
sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.show()

hist_forward_ncp_with_deactivation = ncp_model_with_deactivation.fit(x=x_train_forward_activation_with_deactivation, y=y_train_forward_activation_with_deactivation, batch_size=32, epochs=100, validation_data=(x_valid_forward_activation_with_deactivation, y_valid_forward_activation_with_deactivation))


# Plotting the history data
sns.set()
plt.figure(figsize=(6, 4))
plt.plot(hist_forward_ncp_with_deactivation.history["loss"], label="Training loss")
plt.plot(hist_forward_ncp_with_deactivation.history["val_loss"], label="Validation loss")

plt.legend(loc="upper right")
plt.xlabel("Training steps")
plt.ylabel("Loss")
plt.show()


y_test_forward_activation_with_deactivation_predicted = ncp_model_with_deactivation.predict(x_test_forward_activation_with_deactivation)

# Plotting the training data
plt.figure(figsize=(8, 4))
fig, axs = plt.subplots(3,1)
axs[0].plot(x_test_forward_activation_with_deactivation[0, :], 'tab:red')
axs[0].set_title('PLM')
axs[1].plot(y_test_forward_activation_with_deactivation[0, :], 'tab:blue')
axs[1].set_title('AVB true')
axs[2].plot(np.squeeze(y_test_forward_activation_with_deactivation_predicted[0, :]), 'tab:green')
axs[2].set_title('AVB predicted')

plt.tight_layout()
plt.show()


#%%