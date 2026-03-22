"""
Mean reversion strategy — buys oversold conditions, sells overbought.

Signal logic (all lookback-only):
    - Uses RSI and price deviation from SMA (both pre-computed in features)
    - Buy when RSI < oversold AND price < SMA - threshold
    - Sell when RSI > overbought OR price > SMA + threshold
    - Hold otherwise
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.core.strategies.base import BaseStrategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion: buy dips, sell rips based on RSI + SMA deviation."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        defaults = {
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "sma_window": 20,
            "sma_entry_threshold": -0.03,
            "sma_exit_threshold": 0.02,
        }
        merged = {**defaults, **(params or {})}
        super().__init__(name="Mean Reversion", params=merged)

        self.rsi_oversold = merged["rsi_oversold"]
        self.rsi_overbought = merged["rsi_overbought"]
        self.sma_window = merged["sma_window"]
        self.sma_entry = merged["sma_entry_threshold"]
        self.sma_exit = merged["sma_exit_threshold"]

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate mean-reversion signals from RSI and SMA deviation."""
        # Get RSI (pre-computed by feature pipeline)
        if "rsi" not in df.columns:
            raise ValueError("RSI column not found — run feature pipeline first")

        rsi = df["rsi"]

        # Get SMA deviation (pre-computed)
        sma_dev_col = f"close_to_sma_{self.sma_window}"
        if sma_dev_col in df.columns:
            sma_deviation = df[sma_dev_col]
        else:
            sma = df["close"].rolling(self.sma_window, min_periods=self.sma_window).mean()
            sma_deviation = df["close"] / (sma + 1e-10) - 1.0

        signals = pd.Series(0, index=df.index, dtype=int)

        position = 0
        for i in range(len(df)):
            r = rsi.iloc[i]
            dev = sma_deviation.iloc[i]

            if pd.isna(r) or pd.isna(dev):
                # Exit any position on NaN data — don't carry stale positions
                position = 0
                signals.iloc[i] = 0
                continue

            if position == 0:
                # Entry: oversold + below SMA
                if r < self.rsi_oversold and dev < self.sma_entry:
                    position = 1  # long (expect reversion upward)
                # Entry: overbought + above SMA (short the overextension)
                elif r > self.rsi_overbought and dev > -self.sma_entry:
                    position = -1  # short (expect reversion downward)
            elif position == 1:
                # Long exit: RSI recovered or price above SMA
                if r > self.rsi_overbought or dev > self.sma_exit:
                    position = 0
            elif position == -1:
                # Short exit: RSI dropped or price below SMA
                if r < self.rsi_oversold or dev < -self.sma_exit:
                    position = 0

            signals.iloc[i] = position

        n_long = (signals == 1).sum()
        n_short = (signals == -1).sum()
        n_flat = (signals == 0).sum()
        logger.info(
            f"Mean Reversion signals: {n_long} long, {n_short} short, {n_flat} flat "
            f"(RSI bounds={self.rsi_oversold}/{self.rsi_overbought})"
        )
        return signals
