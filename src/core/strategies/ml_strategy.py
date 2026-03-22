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
        self.proba_col = f"proba_{model_name}"
        self.allow_shorts = self.params.get("allow_shorts", False)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate signals by reading the pre-computed prediction column.
        """
        if df.empty:
            return pd.Series(0.0, index=df.index, dtype=float)

        if self.proba_col in df.columns:
            probas = df[self.proba_col].values
            # Continuous proportional sizing
            base_signals = 2.0 * (probas - 0.5)
            base_signals = np.clip(base_signals, -1.0, 1.0)
            if not self.allow_shorts:
                base_signals = np.clip(base_signals, 0.0, 1.0)
        elif self.prediction_col in df.columns:
            preds = df[self.prediction_col].values
            if self.allow_shorts:
                base_signals = np.where(preds > 0, 1.0, -1.0)
            else:
                base_signals = np.where(preds > 0, 1.0, 0.0)
        else:
            return pd.Series(0.0, index=df.index, dtype=float)

        signals_array = base_signals.copy()

        # Apply Market Regime Biases if available
        if "market_regime" in df.columns:
            regimes = df["market_regime"].values
            
            # 1. High Volatility (1) -> Reduce exposure (flat)
            signals_array[regimes == 1] = 0.0
            
            # 2. Trending
            # Trending UP (2) -> Momentum bias: Block short trades
            signals_array[(regimes == 2) & (base_signals < 0)] = 0.0
            # Trending DOWN (3) -> Momentum bias: Block long trades
            signals_array[(regimes == 3) & (base_signals > 0)] = 0.0
            
            # 3. Mean Reverting (0) -> Revert bias
            if "log_ret_1d" in df.columns:
                ret_1d = df["log_ret_1d"].values
                # Cancel long signal if yesterday was already positive
                signals_array[(regimes == 0) & (base_signals > 0) & (ret_1d > 0)] = 0.0
                # Cancel short signal if yesterday was already negative
                if self.allow_shorts:
                    signals_array[(regimes == 0) & (base_signals < 0) & (ret_1d < 0)] = 0.0

        return pd.Series(signals_array, index=df.index, dtype=float)
