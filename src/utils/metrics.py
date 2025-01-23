# utils/metrics.py
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict

class ModelEvaluator:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = model.predict(X_test)

    def _reshape_for_metrics(self, y):
        """Reshape 3D array to 2D for sklearn metrics."""
        if y.ndim == 3:
            return y.reshape(y.shape[0] * y.shape[1], -1)
        return y

    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate regression metrics after reshaping if needed."""
        y_test_reshaped = self._reshape_for_metrics(self.y_test)
        y_pred_reshaped = self._reshape_for_metrics(self.y_pred)

        return {
            'mse': mean_squared_error(y_test_reshaped, y_pred_reshaped),
            'mae': mean_absolute_error(y_test_reshaped, y_pred_reshaped),
            'r2': r2_score(y_test_reshaped, y_pred_reshaped)
        }