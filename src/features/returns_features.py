"""
Returns and volatility features — all computed from past data only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReturnsFeatures:
    """Log returns, rolling returns, realized volatility, and rolling statistics."""

    def __init__(self, params: dict[str, Any]) -> None:
        ret = params.get("returns", {})
        vol = params.get("volatility", {})
        stats = params.get("rolling_stats", {})

        self.log_return_periods: list[int] = ret.get("log_return_periods", [1, 5, 10, 20])
        self.rolling_return_windows: list[int] = ret.get("rolling_return_windows", [5, 10, 20])
        self.lagged_returns: list[int] = ret.get("lagged_returns", [1, 2, 3, 5])
        self.vol_windows: list[int] = vol.get("rolling_windows", [5, 10, 20])
        self.ewm_span: int = vol.get("ewm_span", 20)
        self.stat_windows: list[int] = stats.get("windows", [5, 10, 20])
        self.stat_metrics: list[str] = stats.get("metrics", ["mean", "std", "skew"])

    def add_log_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Log returns over N periods: ln(P_t / P_{t-N})."""
        for n in self.log_return_periods:
            df[f"log_ret_{n}d"] = np.log(df["close"] / df["close"].shift(n))
        
        # Lagged 1-day returns
        if "log_ret_1d" not in df.columns:
            df["log_ret_1d"] = np.log(df["close"] / df["close"].shift(1))
            
        for lag in self.lagged_returns:
            df[f"log_ret_1d_lag_{lag}"] = df["log_ret_1d"].shift(lag)
            
        return df

    def add_rolling_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simple rolling cumulative return over window."""
        for w in self.rolling_return_windows:
            df[f"rolling_ret_{w}d"] = df["close"].pct_change(periods=w)
        return df

    def add_realized_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Realized volatility = rolling std of log returns, annualized."""
        # Ensure 1-day log return exists
        if "log_ret_1d" not in df.columns:
            df["log_ret_1d"] = np.log(df["close"] / df["close"].shift(1))

        for w in self.vol_windows:
            df[f"realized_vol_{w}d"] = (
                df["log_ret_1d"].rolling(window=w, min_periods=w).std() * np.sqrt(252)
            )

        # Exponentially weighted volatility
        df["ewm_vol"] = (
            df["log_ret_1d"].ewm(span=self.ewm_span, min_periods=self.ewm_span).std() * np.sqrt(252)
        )
        
        # Volatility ratio (short vs long)
        if "realized_vol_5d" in df.columns and "realized_vol_20d" in df.columns:
            df["vol_ratio_5_20"] = df["realized_vol_5d"] / (df["realized_vol_20d"] + 1e-10)
            
        return df

    def add_rolling_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling mean, std, skewness of returns."""
        if "log_ret_1d" not in df.columns:
            df["log_ret_1d"] = np.log(df["close"] / df["close"].shift(1))

        for w in self.stat_windows:
            rolling = df["log_ret_1d"].rolling(window=w, min_periods=w)
            if "mean" in self.stat_metrics:
                df[f"ret_mean_{w}d"] = rolling.mean()
            if "std" in self.stat_metrics:
                df[f"ret_std_{w}d"] = rolling.std()
            if "skew" in self.stat_metrics:
                df[f"ret_skew_{w}d"] = rolling.skew()

        return df

    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume-derived features: relative volume, OBV approximation."""
        # Volume change
        df["volume_change_pct"] = df["volume"].pct_change()

        # Relative volume vs 20-day average
        df["relative_volume"] = df["volume"] / (
            df["volume"].rolling(20, min_periods=10).mean() + 1e-10
        )

        # On-Balance Volume (OBV)
        direction = np.sign(df["close"].diff())
        df["obv"] = (direction * df["volume"]).cumsum()
        # Normalize OBV to rolling z-score for comparability
        obv_mean = df["obv"].rolling(63, min_periods=21).mean()
        obv_std = df["obv"].rolling(63, min_periods=21).std()
        df["obv_zscore"] = (df["obv"] - obv_mean) / (obv_std + 1e-10)

        return df

    def add_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all return and volatility features."""
        logger.info("Computing returns & volatility features...")
        df = self.add_log_returns(df)
        df = self.add_rolling_returns(df)
        df = self.add_realized_volatility(df)
        df = self.add_rolling_statistics(df)
        df = self.add_volume_features(df)
        logger.info(f"Returns/vol features added: {len(df.columns)} total columns")
        return df
