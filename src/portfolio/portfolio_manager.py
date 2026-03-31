"""
Portfolio Manager module.
Handles capital allocation across an array of independent assets/signals.
"""

from typing import Any

import pandas as pd
import numpy as np


class PortfolioManager:
    """
    Groups asset positions and computes final portfolio-level allocations.
    Supports: equal_weight, risk_parity
    """

    def __init__(self, allocation_type: str = "equal_weight", returns_data: dict[str, pd.Series] | None = None) -> None:
        self.allocation_type = allocation_type
        self.returns_data = returns_data or {}
        
    def allocate(self, positions_dict: dict[str, pd.Series]) -> pd.DataFrame:
        """
        Takes raw risk-managed positions [-1, 1] per ticker and applies portfolio-level capital allocation.
        Returns a DataFrame of allocated fractional weights.
        """
        df_positions = pd.DataFrame(positions_dict).fillna(0.0)
        
        if self.allocation_type == "equal_weight":
            n_assets = len(positions_dict)
            if n_assets == 0:
                return df_positions
            return df_positions / n_assets
            
        elif self.allocation_type == "risk_parity":
            return self._risk_parity(df_positions)
        
        raise NotImplementedError(f"Allocation method {self.allocation_type} not supported.")
    
    def _risk_parity(self, df_positions: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse-volatility weighting: each asset's weight is proportional
        to 1/vol, so high-vol assets get smaller allocations.
        Uses rolling 60-day volatility from returns_data if available.
        """
        if not self.returns_data:
            # Fallback to equal weight if no returns data provided
            n = df_positions.shape[1]
            return df_positions / max(n, 1)
        
        # Build volatility dataframe
        vol_dict = {}
        for ticker, ret_series in self.returns_data.items():
            if ticker in df_positions.columns:
                vol = ret_series.rolling(60, min_periods=20).std() * np.sqrt(252)
                vol_dict[ticker] = vol
        
        if not vol_dict:
            n = df_positions.shape[1]
            return df_positions / max(n, 1)
            
        vol_df = pd.DataFrame(vol_dict).reindex(df_positions.index).ffill().fillna(0.2)
        
        # Inverse volatility weights
        inv_vol = 1.0 / (vol_df + 1e-10)
        weight_sum = inv_vol.sum(axis=1)
        weights = inv_vol.div(weight_sum, axis=0)
        
        # Apply weights to positions
        allocated = df_positions * weights
        return allocated

