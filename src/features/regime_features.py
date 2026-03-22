"""
Market regime detection features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RegimeFeatures:
    """
    Computes market regime features:
    - High Volatility
    - Trending (Up/Down)
    - Mean-Reverting (Ranging)
    """

    def __init__(self, params: dict | None = None) -> None:
        self.params = params or {}
        self.vol_window = self.params.get("vol_window", 21)
        self.trend_window = self.params.get("trend_window", 50)

    def add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add regime classifications to the dataframe."""
        logger.info("Computing market regime features...")
        
        # 1. Rolling Volatility (annualized)
        if "log_ret_1d" not in df.columns:
            df["log_ret_1d"] = np.log(df["close"] / df["close"].shift(1))
            
        vol = df["log_ret_1d"].rolling(window=self.vol_window, min_periods=self.vol_window).std() * np.sqrt(252)
        df["regime_volatility"] = vol
        
        # High Volatility Regime: Volatility is > 1.5x its historical 252-day moving average
        vol_mean_252 = vol.rolling(252, min_periods=63).mean()
        df["regime_is_high_vol"] = (vol > (vol_mean_252 * 1.5)).astype(int)

        # 2. Trend Strength (Slope of SMA)
        sma = df["close"].rolling(self.trend_window, min_periods=self.trend_window).mean()
        # Slope over 5 days
        sma_slope = (sma - sma.shift(5)) / sma.shift(5)
        df["regime_trend_slope"] = sma_slope
        
        # Trending Regime: absolute slope > 1% over 5 days
        df["regime_is_trending_up"] = (sma_slope > 0.01).astype(int)
        df["regime_is_trending_down"] = (sma_slope < -0.01).astype(int)
        
        # 3. Drawdown
        roll_max = df["close"].rolling(252, min_periods=63).max()
        df["regime_drawdown"] = (df["close"] / roll_max) - 1.0

        # 4. Master Regime Classification
        # 1 = High Vol
        # 2 = Trending Up
        # 3 = Trending Down
        # 0 = Mean Reverting / Ranging (Default)
        
        regime = pd.Series(0, index=df.index, dtype=int)
        regime[df["regime_is_trending_up"] == 1] = 2
        regime[df["regime_is_trending_down"] == 1] = 3
        
        # Overwrite with high vol if exists (High Vol takes precedence to reduce exposure)
        regime[df["regime_is_high_vol"] == 1] = 1
        
        df["market_regime"] = regime
        
        logger.info(f"Regimes detected: {df['market_regime'].value_counts().to_dict()}")
        return df
