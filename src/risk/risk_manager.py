"""
Risk Management Engine.
Handles volatility targeting, maximum drawdown protection, and stop losses.
"""

from typing import Any

import numpy as np
import pandas as pd


class RiskManager:
    """
    Applies risk controls to transform raw strategy signals into risk-adjusted positions.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}
        
        # Volatility Targeting
        self.target_vol = self.params.get("target_vol", 0.15)  # 15% annualized target
        self.use_vol_target = self.params.get("use_vol_target", True)
        
        # Max Drawdown Guard
        self.max_dd_limit = self.params.get("max_dd_limit", -0.15)  # Reduce exposure if DD > 15%
        self.use_dd_guard = self.params.get("use_dd_guard", True)

        # Dynamic Trailing Stop Loss
        self.use_trailing_stop = self.params.get("use_trailing_stop", True)
        self.atr_multiplier = self.params.get("trailing_stop_atr_multiplier", 3.0)

    def apply_rules(
        self, 
        df: pd.DataFrame, 
        positions: pd.Series, 
        asset_returns: pd.Series
    ) -> pd.Series:
        """
        Apply risk rules sequentially to raw positions.
        `df` is the ticker prices/features dataframe.
        `positions` is the initial fractional positions series.
        `asset_returns` is the daily return of the asset.
        """
        adj_positions = positions.copy()
        
        # 1. Volatility Targeting
        if self.use_vol_target and "log_ret_1d" in df.columns:
            # Calculate realized ~21d volatility annualized
            realized_vol = df["log_ret_1d"].rolling(21, min_periods=21).std() * np.sqrt(252)
            # Fill NaNs with a reasonable default (e.g. 0.20)
            realized_vol = realized_vol.fillna(0.20)
            
            # Prevent extreme scaling by capping realized_vol lower bound
            realized_vol = np.maximum(realized_vol, 0.05)
            
            # Vol target scalar: Target Vol / Realized Vol
            vol_scalar = self.target_vol / realized_vol
            
            # We don't leverage > 1.5x via Vol Targeting usually 
            vol_scalar = np.clip(vol_scalar, 0.0, 1.5)
            
            adj_positions = adj_positions * vol_scalar

        # Track PnL internally to calculate Drawdowns and Stop loss limits
        # We simulate the exact returns using the adjusted positions
        # since actual positions happen next bar, we shift them.
        running_returns = adj_positions.shift(1).fillna(0) * asset_returns
        cum_returns = (1 + running_returns).cumprod()
        roll_max = cum_returns.cummax()
        drawdowns = cum_returns / roll_max - 1.0

        # 2. Max Drawdown Guard
        if self.use_dd_guard:
            # If current drawdown is worse than max_dd_limit, reduce exposure by 50%
            dd_violation = drawdowns < self.max_dd_limit
            # Exposure scales dynamically: 0.5x during deep DD.
            # We use shifted DD to avoid lookahead (acting on yesterday's closing DD)
            adj_positions.loc[dd_violation.shift(1).fillna(False)] *= 0.5

        # 3. Dynamic Position-Level Trailing Stop Loss
        if self.use_trailing_stop and "close" in df.columns:
            close_price = df["close"]
            
            # Estimate daily vol as ATR proxy
            if "log_ret_1d" in df.columns:
                daily_vol = df["log_ret_1d"].rolling(21, min_periods=5).std().bfill().fillna(0.02)
                atr_proxy = close_price * daily_vol
            else:
                atr_proxy = close_price * 0.02
                
            # Identify contiguous non-zero position blocks
            is_active = adj_positions != 0
            trade_blocks = (is_active != is_active.shift(1)).cumsum()
            
            stop_triggered = pd.Series(False, index=df.index)
            
            # Running max price for LONG positions
            is_long = adj_positions > 0
            if is_long.any():
                long_close = close_price.where(is_long)
                running_max = long_close.groupby(trade_blocks).cummax()
                long_trailing_stop = running_max - (self.atr_multiplier * atr_proxy)
                long_stop_triggered = (close_price < long_trailing_stop) & is_long
                stop_triggered = stop_triggered | long_stop_triggered
                
            # Running min price for SHORT positions
            is_short = adj_positions < 0
            if is_short.any():
                short_close = close_price.where(is_short)
                running_min = short_close.groupby(trade_blocks).cummin()
                short_trailing_stop = running_min + (self.atr_multiplier * atr_proxy)
                short_stop_triggered = (close_price > short_trailing_stop) & is_short
                stop_triggered = stop_triggered | short_stop_triggered
            
            # Propagate stop status to the end of the current active trade block
            stop_active = stop_triggered.groupby(trade_blocks).cumsum() > 0
            
            # Flat position on the day AFTER stop is hit
            stop_mask = stop_active.shift(1) == True
            adj_positions = adj_positions.where(~stop_mask, 0.0)
            
        # Ensure we stay within max leverage [-1, 1] generically
        adj_positions = np.clip(adj_positions, -1.0, 1.0)
        
        return adj_positions

