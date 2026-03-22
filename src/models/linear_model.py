"""
Logistic Regression model implementation mapping to BaseModel.
"""

from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

from src.models.base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    def __init__(self, params: dict | None = None) -> None:
        super().__init__(name="LogisticRegression", params=params)
        
        # Extract scikit-learn kwargs
        model_kwargs = self.params.get("model_kwargs", {"max_iter": 1000, "random_state": 42})
        self.model = LogisticRegression(**model_kwargs)

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "LogisticRegressionModel":
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
        # Returns probability of class 1
        classes = list(self.model.classes_)
        if 1 in classes:
            idx = classes.index(1)
            return self.model.predict_proba(X)[:, idx]
        return self.model.predict_proba(X)[:, -1]
