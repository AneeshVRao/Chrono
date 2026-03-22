"""
Ensemble model implementation mapping to BaseModel.
"""

from typing import Literal

import pandas as pd
import numpy as np

from src.models.base_model import BaseModel


class EnsembleModel(BaseModel):
    def __init__(self, models: list[BaseModel], voting: Literal["hard", "soft"] = "soft", params: dict | None = None) -> None:
        super().__init__(name="Ensemble", params=params)
        
        self.models = models
        self.voting = voting
        self.is_fitted = False

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "EnsembleModel":
        """Fit all underlying models."""
        for model in self.models:
            model.fit(X, y)
            
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict.")
            
        if self.voting == "hard":
            # Majority voting
            predictions = np.asarray([model.predict(X) for model in self.models])
            
            # Map elements to sum up votes. Assuming binary classes {0, 1}
            # Or {-1, 1}
            # For robustness, we check the most common class.
            from scipy.stats import mode
            mode_result = mode(predictions, axis=0, keepdims=True)
            return mode_result.mode[0]
            
        elif self.voting == "soft":
            # Average probabilities
            probas = self.predict_proba(X)
            # Threshold at 0.5. Assuming classes are {0, 1} mapped from target_direction
            # If thresholding is different (e.g. classes are -1, 1), we map appropriately.
            # Assuming our target is {0, 1} built from (target_direction > 0).astype(int)
            return (probas > 0.5).astype(int)
            
        else:
            raise ValueError(f"Unknown voting method {self.voting}")

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict_proba.")
            
        # Returns probability of class 1 averaged across all models
        probas = np.asarray([model.predict_proba(X) for model in self.models])
        return np.mean(probas, axis=0)
