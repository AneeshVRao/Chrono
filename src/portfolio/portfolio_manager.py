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
    """

    def __init__(self, allocation_type: str = "equal_weight") -> None:
        self.allocation_type = allocation_type
        
    def allocate(self, positions_dict: dict[str, pd.Series]) -> pd.DataFrame:
        """
        Takes raw risk-managed positions [-1, 1] per ticker and applies portfolio-level capital allocation.
        Returns a DataFrame of allocated fractional weights.
        """
        # Combine into a single dataframe of raw position intents, filling unmapped dates securely
        df_positions = pd.DataFrame(positions_dict).fillna(0.0)
        
        # Baseline equal weight logic
        if self.allocation_type == "equal_weight":
            n_assets = len(positions_dict)
            if n_assets == 0:
                return df_positions
                
            allocated_weights = df_positions / n_assets
            return allocated_weights
        
        # Future extension e.g. Mean Variance Optimization / Risk-Parity
        raise NotImplementedError(f"Allocation method {self.allocation_type} not supported.")
