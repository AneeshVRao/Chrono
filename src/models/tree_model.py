"""
Tree-based model implementation mapping to BaseModel.
(Random Forest placeholder)
"""

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

from src.models.base_model import BaseModel


class RandomForestModel(BaseModel):
    def __init__(self, params: dict | None = None) -> None:
        super().__init__(name="RandomForest", params=params)
        
        model_kwargs = self.params.get("model_kwargs", {"n_estimators": 100, "random_state": 42})
        self.model = RandomForestClassifier(**model_kwargs)

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "RandomForestModel":
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
        classes = list(self.model.classes_)
        if 1 in classes:
            idx = classes.index(1)
            return self.model.predict_proba(X)[:, idx]
        return self.model.predict_proba(X)[:, -1]
