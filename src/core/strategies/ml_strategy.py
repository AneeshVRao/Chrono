"""
Machine Learning Strategy implementation.
"""

import numpy as np
import pandas as pd

from src.core.strategies.base import BaseStrategy


class MLStrategy(BaseStrategy):
    """
    Strategy driven by pre-computed machine learning predictions.
    prediction > 0 -> 1 (long)
    prediction <= 0 -> 0 (or -1 if shorting is enabled)
    """

    def __init__(self, model_name: str, params: dict | None = None) -> None:
        name = params.get("name", f"ML_Strategy({model_name})") if params else f"ML_Strategy({model_name})"
        super().__init__(name=name, params=params)
        
        self.model_name = model_name
        self.prediction_col = f"pred_{model_name}"
        self.allow_shorts = self.params.get("allow_shorts", False)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate signals by reading the pre-computed prediction column.
        """
        if df.empty or self.prediction_col not in df.columns:
            return pd.Series(0, index=df.index, dtype=int)

        preds = df[self.prediction_col].values
        
        # map predictions to positions
        signals_array = np.zeros_like(preds, dtype=int)
        
        # For our target, >0 (i.e. 1) is up, <=0 (i.e. 0) is down
        if self.allow_shorts:
            signals_array[preds > 0] = 1
            signals_array[preds <= 0] = -1
        else:
            signals_array[preds > 0] = 1
            signals_array[preds <= 0] = 0

        signals = pd.Series(signals_array, index=df.index, dtype=int)
        return signals
