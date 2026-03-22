"""
Technical indicators — computed using ONLY past data (no lookahead bias).
All indicators use pandas-ta for vectorized computation where available,
with manual implementations as fallback.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TechnicalIndicators:
    """Compute technical indicators from OHLCV data. All lookback-only."""

    def __init__(self, params: dict[str, Any]) -> None:
        ti = params.get("technical_indicators", {})
        self.sma_windows: list[int] = ti.get("sma_windows", [10, 20, 50, 200])
        self.ema_windows: list[int] = ti.get("ema_windows", [12, 26, 50])
        self.rsi_period: int = ti.get("rsi_period", 14)
        self.macd_fast: int = ti.get("macd", {}).get("fast", 12)
        self.macd_slow: int = ti.get("macd", {}).get("slow", 26)
        self.macd_signal: int = ti.get("macd", {}).get("signal", 9)
        self.bb_window: int = ti.get("bollinger", {}).get("window", 20)
        self.bb_std: float = ti.get("bollinger", {}).get("num_std", 2.0)
        self.atr_period: int = ti.get("atr_period", 14)

    def add_sma(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simple Moving Averages."""
        for w in self.sma_windows:
            df[f"sma_{w}"] = df["close"].rolling(window=w, min_periods=w).mean()
        return df

    def add_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Exponential Moving Averages."""
        for w in self.ema_windows:
            df[f"ema_{w}"] = df["close"].ewm(span=w, adjust=False, min_periods=w).mean()
        return df

    def add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Relative Strength Index (Wilder's smoothing)."""
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1 / self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        df["rsi"] = 100.0 - (100.0 / (1.0 + rs))
        return df

    def add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """MACD line, signal line, histogram."""
        ema_fast = df["close"].ewm(span=self.macd_fast, adjust=False, min_periods=self.macd_fast).mean()
        ema_slow = df["close"].ewm(span=self.macd_slow, adjust=False, min_periods=self.macd_slow).mean()

        df["macd_line"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd_line"].ewm(
            span=self.macd_signal, adjust=False, min_periods=self.macd_signal
        ).mean()
        df["macd_histogram"] = df["macd_line"] - df["macd_signal"]
        return df

    def add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bollinger Bands: middle, upper, lower, bandwidth, %B."""
        rolling = df["close"].rolling(window=self.bb_window, min_periods=self.bb_window)
        df["bb_middle"] = rolling.mean()
        bb_std = rolling.std()

        df["bb_upper"] = df["bb_middle"] + self.bb_std * bb_std
        df["bb_lower"] = df["bb_middle"] - self.bb_std * bb_std
        df["bb_bandwidth"] = (df["bb_upper"] - df["bb_lower"]) / (df["bb_middle"] + 1e-10)
        df["bb_pct_b"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)
        return df

    def add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Average True Range — volatility measure."""
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.ewm(alpha=1 / self.atr_period, min_periods=self.atr_period, adjust=False).mean()
        df["atr_pct"] = df["atr"] / (df["close"] + 1e-10)  # normalized ATR
        return df

    def add_zscores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price Z-score: (price - rolling_mean) / rolling_std"""
        for w in self.sma_windows:
            rolling = df["close"].rolling(window=w, min_periods=w)
            mean = rolling.mean()
            std = rolling.std()
            df[f"zscore_{w}"] = (df["close"] - mean) / (std + 1e-10)
        return df

    def add_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all technical indicators."""
        logger.info("Computing technical indicators...")
        df = self.add_sma(df)
        df = self.add_ema(df)
        df = self.add_rsi(df)
        df = self.add_macd(df)
        df = self.add_bollinger_bands(df)
        df = self.add_atr(df)
        df = self.add_zscores(df)

        # Cross-feature: price relative to SMAs
        for w in self.sma_windows:
            col = f"sma_{w}"
            if col in df.columns:
                df[f"close_to_{col}"] = df["close"] / (df[col] + 1e-10) - 1.0

        logger.info(f"Technical indicators added: {len(df.columns)} total columns")
        return df
