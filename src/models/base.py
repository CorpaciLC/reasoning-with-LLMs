# base.py

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import History

@dataclass
class ModelConfig:
    """
    Configuration for building and compiling a Keras model.
    - input_shape: (timesteps, features)
    - learning_rate: ...
    - dropout_rate: ...
    - gradient_clip: ...
    - loss: ...
    - activation: ...
    - hidden_units: ...
    """
    input_shape: Tuple[Optional[int], int] = (None, 2)
    learning_rate: float = 0.01
    dropout_rate: float = 0.2
    gradient_clip: float = 1.0
    loss: str = 'mean_squared_error'
    activation: str = 'tanh'
    hidden_units: int = 32

class BaseModel(ABC):
    """
    Abstract BaseModel for Keras-based neural nets. 
    Subclasses should implement `_build()` returning a tf.keras.Model.
    """
    def __init__(self, config: ModelConfig, **kwargs):
        self.config = config
        # Child classes should define how they store any extra arguments (e.g., wiring).
        self.model = self._build()
        self._compile()
    
    @abstractmethod
    def _build(self) -> models.Model:
        """
        Build and return a Keras Model instance (Sequential or functional).
        """
        pass
    
    def _compile(self):
        """
        Compile the Keras model with the provided config parameters.
        """
        optimizer = optimizers.Adam(
            learning_rate=self.config.learning_rate,
            clipnorm=self.config.gradient_clip
        )
        self.model.compile(
            optimizer=optimizer,
            loss=self.config.loss,
            metrics=['mae']
        )
    def train_and_evaluate(self, X_train, y_train, X_val, y_val, epochs=25, batch_size=32) -> Tuple[History, float]:
        """Train model and return history with runtime."""
        start_time = time.time()
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        runtime = time.time() - start_time
        return history, runtime