# nn_info.py
from typing import Dict, Any

class NetworkAnalyzer:
    def __init__(self, model, model_name: str):
        self.model = model
        self.model_name = model_name
        
    def extract_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "total_layers": len(self.model.layers),
            "parameters": {
                "total": self.model.count_params(),
                "trainable": sum([layer.count_params() for layer in self.model.layers 
                                if len(layer.trainable_weights) > 0]),
                "non_trainable": sum([layer.count_params() for layer in self.model.layers 
                                    if len(layer.non_trainable_weights) > 0])
            },
            "layers": self._analyze_layers(),
            "memory_estimate_mb": self._calculate_memory()
        }
    
    def _analyze_layers(self):
        return [{
            "name": layer.name,
            "type": type(layer).__name__,
            "shape": str(layer.output_shape),
            "parameters": layer.count_params()
        } for layer in self.model.layers]
        
    def _calculate_memory(self):
        return sum([sum([w.numpy().nbytes for w in layer.weights])
                   for layer in self.model.layers]) / (1024 * 1024)