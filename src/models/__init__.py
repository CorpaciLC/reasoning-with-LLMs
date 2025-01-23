# models/__init__.py
from .base import ModelConfig
from .ltc import LTCModel 
from .rnn import RNNModel
from tensorflow.keras import layers

class ModelFactory:
    @staticmethod
    def create(model_type: str, config: ModelConfig, **kwargs):
        """Create a model instance based on the specified type and configuration.
        
        Args:
            model_type (str): Type of model to create ('ltc', 'rnn', 'lstm', 'gru')
            config (ModelConfig): Model configuration
            **kwargs: Additional arguments passed to the model constructor
        
        Returns:
            BaseModel: An instance of the specified model type
        """
        if model_type == 'ltc':
            if 'wiring' not in kwargs:
                raise ValueError("LTC models require a 'wiring' parameter")
            return LTCModel(config, **kwargs)
        
        elif model_type in ['rnn', 'lstm', 'gru']:
            # Create appropriate layer based on type
            if model_type == 'lstm':
                mid_layer = layers.LSTM(config.hidden_units, return_sequences=True)
            elif model_type == 'gru':
                mid_layer = layers.GRU(config.hidden_units, return_sequences=True)
            else:  # basic RNN
                mid_layer = layers.SimpleRNN(config.hidden_units, return_sequences=True)
            
            kwargs['mid_layer'] = mid_layer
            return RNNModel(config, **kwargs)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")

__all__ = ['ModelConfig', 'LTCModel', 'RNNModel', 'ModelFactory']