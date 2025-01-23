# rnn.py

from .base import BaseModel, ModelConfig
from tensorflow.keras import layers, models

class RNNModel(BaseModel):
    def _build(self) -> models.Model:
        """
        Build an RNN model (LSTM here) for time-series regression.
        """
        model = models.Sequential([
            layers.InputLayer(input_shape=self.config.input_shape),
            # Optionally pick only one normalization
            # layers.LayerNormalization(),
            layers.LSTM(self.config.hidden_units, return_sequences=True),
            layers.Dropout(self.config.dropout_rate),
            layers.BatchNormalization(),
            layers.Dense(self.config.hidden_units, activation=self.config.activation),
            # For regression, often linear output is used:
            layers.Dense(1, activation='linear')
        ])
        return model
