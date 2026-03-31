"""
LightGBM model wrapper conforming to BaseModel interface.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from src.models.base_model import BaseModel


class LightGBMModel(BaseModel):
    """LightGBM gradient boosting classifier."""

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(name="LightGBM", params=params)

        model_kwargs = self.params.get("model_kwargs", {})
        self.model = LGBMClassifier(
            n_estimators=model_kwargs.get("n_estimators", 100),
            max_depth=model_kwargs.get("max_depth", 5),
            learning_rate=model_kwargs.get("learning_rate", 0.05),
            num_leaves=model_kwargs.get("num_leaves", 31),
            min_child_samples=model_kwargs.get("min_child_samples", 20),
            random_state=model_kwargs.get("random_state", 42),
            verbose=-1,
        )
        self.is_fitted = False

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "LightGBMModel":
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        proba = self.model.predict_proba(X)
        classes = list(self.model.classes_)
        if 1 in classes:
            return proba[:, classes.index(1)]
        return proba[:, -1]
