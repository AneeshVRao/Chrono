"""
Base interface for all ML models.
"""

from abc import ABC, abstractmethod

import pandas as pd
import numpy as np


class BaseModel(ABC):
    """
    Abstract base class for all machine learning models in the platform.
    """

    def __init__(self, name: str, params: dict | None = None) -> None:
        self.name = name
        self.params = params or {}
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "BaseModel":
        """
        Train the machine learning model.
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        Returns array of {1, -1} (or {1, 0}).
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        Returns array of probabilities for the positive class.
        """
        pass
