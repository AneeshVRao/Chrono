"""
Beta Neutralization module.
Calculates dynamic rolling beta vs SPY and applies portfolio-level hedges.
"""

from typing import Any
import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

class BetaNeutralizer:
    def __init__(self, spy_returns: pd.Series, window: int = 60) -> None:
        """
        spy_returns: pd.Series of SPY daily returns aligned with the trading days.
        window: rolling regression window for beta calculation (must be backward-looking).
        """
        self.spy_returns = spy_returns
        self.window = window

    def compute_asset_beta(self, asset_returns: pd.Series) -> pd.Series:
        """
        Compute rolling 60-day beta (asset_returns ~ SPY_returns).
        No lookahead: rolling covariance and variance are purely backward-looking.
        """
        # Align series to avoid dimension issues
        df = pd.DataFrame({"asset": asset_returns, "spy": self.spy_returns}).dropna()
        
        # covariance matrix rolling
        cov = df["asset"].rolling(self.window).cov(df["spy"])
        var = df["spy"].rolling(self.window).var()
        
        # beta = cov(asset, spy) / var(spy)
        beta = cov / (var + 1e-10)
        
        # fillna with 1.0 (market beta) or ffill
        return beta.reindex(asset_returns.index).fillna(1.0)
        
    def alpha_decomposition(self, asset_returns: pd.Series, beta: pd.Series) -> pd.DataFrame:
        """
        Decomposes total return into alpha and beta components.
        total_return = alpha + beta * market_return
        return DataFrame with total_return, alpha, beta_component
        """
        aligned_spy = self.spy_returns.reindex(asset_returns.index).fillna(0)
        beta_component = beta * aligned_spy
        alpha = asset_returns - beta_component
        
        return pd.DataFrame({
            "total_return": asset_returns,
            "alpha": alpha,
            "beta_component": beta_component,
            "beta": beta
        })

    def apply_hedge(self, portfolio_positions: pd.DataFrame, asset_betas: pd.DataFrame) -> pd.Series:
        """
        Given the positions (weights or capital exposure) of each asset in the portfolio,
        and their respective dynamic betas, calculate the SPY synthetic hedge position.
        hedge_position = -beta * portfolio_exposure
        portfolio_positions: DataFrame indexed by date, columns=tickers
        asset_betas: DataFrame indexed by date, columns=tickers
        Returns a Series of SPY positions (weights).
        """
        # Align DataFrames
        pos, betas = portfolio_positions.align(asset_betas, join='inner')
        # Total portfolio beta = sum_i(weight_i * beta_i)
        portfolio_beta = (pos * betas).sum(axis=1)
        # Hedge is short the portfolio beta
        hedge_position = -portfolio_beta
        return hedge_position
