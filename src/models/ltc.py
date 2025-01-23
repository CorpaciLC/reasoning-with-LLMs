# ltc.py

from .base import BaseModel, ModelConfig
from ncps.tf import LTC
from tensorflow.keras import layers, models

class LTCModel(BaseModel):
    def __init__(self, config: ModelConfig, wiring, **kwargs):
        self.wiring = wiring
        super().__init__(config, **kwargs)

    def _build(self) -> models.Sequential:
        return models.Sequential([
            layers.InputLayer(input_shape=self.config.input_shape),
            LTC(
                self.wiring,                   # pass the wiring
                # activation=self.config.activation, 
                return_sequences=True
            ),
            layers.Dropout(self.config.dropout_rate),
            layers.BatchNormalization(),
            layers.Dense(1)
        ])
