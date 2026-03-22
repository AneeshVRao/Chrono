"""
XGBoost model implementation mapping to BaseModel.
"""

import pandas as pd
import numpy as np

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

from src.models.base_model import BaseModel


class XGBoostModel(BaseModel):
    """
    XGBoost Classifier wrapper.
    Requires xgboost module installed.
    """

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(name="XGBoost", params=params)
        
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed. Please install it to use XGBoostModel.")
            
        # Default typical quantitative params allowing tree pruning and anti-overfitting
        default_kwargs = {
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.05,
            "eval_metric": "logloss",
            "random_state": 42
        }
        
        model_kwargs = self.params.get("model_kwargs", default_kwargs)
        self.model = XGBClassifier(**model_kwargs)

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "XGBoostModel":
        # XGBoost requires targets strictly starting from 0, e.g {0, 1}
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict_proba.")
            
        probas = self.model.predict_proba(X)
        
        # XGBClassifier generally returns probabilities shaped (N, 2) when trained on binary
        classes = list(self.model.classes_)
        if 1 in classes:
            idx = classes.index(1)
            return probas[:, idx]
            
        return probas[:, -1]
