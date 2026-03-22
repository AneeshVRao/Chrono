"""
Momentum strategy — goes long when recent returns are positive.

Signal logic (all lookback-only):
    - Compute N-day momentum (cumulative return over past N days)
    - Long if momentum > entry_threshold
    - Flat if momentum < exit_threshold
    - Hold otherwise
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.core.strategies.base import BaseStrategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MomentumStrategy(BaseStrategy):
    """Time-series momentum: ride trends, exit on reversals."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        defaults = {
            "lookback_period": 21,
            "entry_threshold": 0.02,
            "exit_threshold": -0.01,
        }
        merged = {**defaults, **(params or {})}
        super().__init__(name="Momentum", params=merged)

        self.lookback = merged["lookback_period"]
        self.entry_thresh = merged["entry_threshold"]
        self.exit_thresh = merged["exit_threshold"]

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate momentum signals using only past N-day returns."""
        # Use rolling return column if available, otherwise compute
        col = f"rolling_ret_{self.lookback}d"
        if col in df.columns:
            momentum = df[col]
        else:
            momentum = df["close"].pct_change(periods=self.lookback)

        signals = pd.Series(0, index=df.index, dtype=int)

        # Vectorized signal generation
        position = 0
        for i in range(len(df)):
            mom = momentum.iloc[i]

            if pd.isna(mom):
                # Exit any position on NaN data -- don't carry stale positions
                position = 0
                signals.iloc[i] = 0
                continue

            if position == 0:
                # Entry condition
                if mom > self.entry_thresh:
                    position = 1
                elif mom < -self.entry_thresh:
                    position = -1
            elif position == 1:
                # Long exit
                if mom < self.exit_thresh:
                    position = 0
            elif position == -1:
                # Short exit
                if mom > -self.exit_thresh:
                    position = 0

            signals.iloc[i] = position

        n_long = (signals == 1).sum()
        n_short = (signals == -1).sum()
        n_flat = (signals == 0).sum()
        logger.info(
            f"Momentum signals: {n_long} long, {n_short} short, {n_flat} flat "
            f"(lookback={self.lookback}d)"
        )
        return signals
