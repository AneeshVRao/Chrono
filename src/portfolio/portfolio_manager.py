"""
Portfolio Manager module.
Handles capital allocation across an array of independent assets/signals.

Supports:
  - equal_weight   : 1/N allocation
  - risk_parity    : inverse-volatility weighting
  - cvar            : Conditional Value-at-Risk optimised allocation (scipy)
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import numpy as np
from scipy.optimize import minimize

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#   CVaR Optimizer (Historical Simulation)
# ═══════════════════════════════════════════════════════════════════════════════

class CVaROptimizer:
    """
    Minimise the 95 % Conditional Value-at-Risk (CVaR / Expected Shortfall)
    of a multi-asset portfolio using historical simulation.

    Methodology
    -----------
    1.  Build a rolling returns matrix R ∈ ℝ^{T×N} from daily asset returns
        over a trailing `lookback` window.
    2.  For candidate weights w ∈ ℝ^N, compute portfolio returns  r_p = R @ w.
    3.  VaR_α  = α-quantile of r_p   (the worst-case daily loss threshold).
    4.  CVaR_α = mean of all r_p ≤ VaR_α  (average loss in the tail).
    5.  Use scipy.optimize.minimize (SLSQP) to find w* that minimises CVaR_α
        subject to:
          •  Σ w_i = 1                (fully invested)
          •  w_min ≤ w_i ≤ w_max     (no extreme leverage / shorting)

    Parameters
    ----------
    alpha : float
        Confidence level for CVaR.  95 % → alpha = 0.95.
    lookback : int
        Number of trailing calendar-day returns to use per optimisation step.
    min_weight : float
        Lower bound on individual asset weight  (≥ 0 prevents shorting).
    max_weight : float
        Upper bound on individual asset weight  (≤ 1 prevents excessive concentration).
    rebalance_freq : int
        Re-optimise weights every `rebalance_freq` trading days to reduce look-ahead
        and reflect the cost of continuous re-balancing.
    min_history : int
        Minimum number of valid return observations required before CVaR is computed;
        falls back to equal-weight below this threshold.
    """

    def __init__(
        self,
        alpha: float = 0.95,
        lookback: int = 60,
        min_weight: float = 0.0,
        max_weight: float = 0.40,
        rebalance_freq: int = 20,
        min_history: int = 30,
    ) -> None:
        self.alpha = alpha
        self.lookback = lookback
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.rebalance_freq = rebalance_freq
        self.min_history = min_history

    # ── core objective ──────────────────────────────────────────────────────
    @staticmethod
    def _portfolio_cvar(
        weights: np.ndarray,
        returns_matrix: np.ndarray,
        alpha: float,
    ) -> float:
        """
        Compute CVaR (Expected Shortfall) at confidence level `alpha`
        for the portfolio defined by `weights`.

        Returns a *positive* number representing the expected tail loss
        (we sign-flip so that the optimiser minimises tail risk).
        """
        port_returns = returns_matrix @ weights            # (T,)
        var_threshold = np.percentile(port_returns, (1 - alpha) * 100)
        tail = port_returns[port_returns <= var_threshold]

        if len(tail) == 0:
            return 0.0

        cvar = -tail.mean()  # positive = larger tail loss = worse
        return cvar

    # ── single-shot optimisation ────────────────────────────────────────────
    def optimise(self, returns_matrix: np.ndarray) -> np.ndarray:
        """
        Solve for the minimum-CVaR portfolio given a (T, N) returns matrix.

        Returns
        -------
        weights : ndarray of shape (N,)
        """
        n_assets = returns_matrix.shape[1]

        # Seed with equal-weight
        w0 = np.ones(n_assets) / n_assets

        # Bounds per asset
        bounds = [(self.min_weight, self.max_weight)] * n_assets

        # Full investment constraint
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        result = minimize(
            self._portfolio_cvar,
            w0,
            args=(returns_matrix, self.alpha),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-10, "disp": False},
        )

        if result.success:
            return result.x
        else:
            logger.warning(
                f"CVaR optimisation did not converge: {result.message}. "
                f"Falling back to equal-weight."
            )
            return w0

    # ── full time-series weight generation ──────────────────────────────────
    def compute_weights(
        self,
        returns_df: pd.DataFrame,
        index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        Walk through `index` producing optimised CVaR weights.

        Parameters
        ----------
        returns_df : DataFrame  (T × N)
            Daily returns per asset, indexed by date.
        index : DatetimeIndex
            Target dates to compute weights for (must be a subset of returns_df's index).

        Returns
        -------
        DataFrame  (T × N)  of asset weights aligned to `index`.
        """
        tickers = returns_df.columns.tolist()
        n_assets = len(tickers)
        equal_w = np.ones(n_assets) / n_assets

        weights_list: list[np.ndarray] = []
        dates_used: list[pd.Timestamp] = []

        last_w = equal_w.copy()
        bars_since_rebalance = self.rebalance_freq  # force optimisation on first bar

        for dt in index:
            # Locate trailing window of returns available *up to and including* dt
            loc = returns_df.index.get_loc(dt) if dt in returns_df.index else None
            if loc is None:
                weights_list.append(last_w)
                dates_used.append(dt)
                continue

            # Resolve slices if loc is a slice (shouldn't be for unique index)
            if isinstance(loc, slice):
                loc = loc.stop - 1

            start = max(0, loc - self.lookback + 1)
            window = returns_df.iloc[start : loc + 1].values  # (≤lookback, N)

            bars_since_rebalance += 1

            if window.shape[0] >= self.min_history and bars_since_rebalance >= self.rebalance_freq:
                # Drop rows with any NaN
                valid = ~np.isnan(window).any(axis=1)
                clean = window[valid]

                if clean.shape[0] >= self.min_history:
                    last_w = self.optimise(clean)
                    bars_since_rebalance = 0

            weights_list.append(last_w)
            dates_used.append(dt)

        weights_out = pd.DataFrame(weights_list, index=dates_used, columns=tickers)
        return weights_out.reindex(index).ffill().fillna(1.0 / n_assets)


# ═══════════════════════════════════════════════════════════════════════════════
#   Portfolio Manager
# ═══════════════════════════════════════════════════════════════════════════════

class PortfolioManager:
    """
    Groups asset positions and computes final portfolio-level allocations.
    Supports: equal_weight, risk_parity, cvar
    """

    def __init__(
        self,
        allocation_type: str = "equal_weight",
        returns_data: dict[str, pd.Series] | None = None,
        cvar_params: dict[str, Any] | None = None,
    ) -> None:
        self.allocation_type = allocation_type
        self.returns_data = returns_data or {}
        self.cvar_params = cvar_params or {}

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

        elif self.allocation_type == "cvar":
            return self._cvar_allocation(df_positions)

        raise NotImplementedError(f"Allocation method {self.allocation_type} not supported.")

    # ── Risk Parity (inverse-volatility) ────────────────────────────────────
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

    # ── CVaR Allocation ─────────────────────────────────────────────────────
    def _cvar_allocation(self, df_positions: pd.DataFrame) -> pd.DataFrame:
        """
        Minimise 95 % CVaR across the portfolio.

        Steps
        -----
        1. Build a daily-returns matrix from `self.returns_data`.
        2. Feed it into CVaROptimizer which returns time-varying weights.
        3. Multiply raw positions by CVaR weights for the final allocation.
        """
        if not self.returns_data:
            logger.warning("CVaR allocation requested but no returns data — falling back to equal weight.")
            n = df_positions.shape[1]
            return df_positions / max(n, 1)

        # Assemble returns matrix (only assets present in positions)
        tickers_in_scope = [t for t in df_positions.columns if t in self.returns_data]

        if len(tickers_in_scope) == 0:
            n = df_positions.shape[1]
            return df_positions / max(n, 1)

        returns_df = pd.DataFrame(
            {t: self.returns_data[t] for t in tickers_in_scope}
        ).sort_index().ffill().fillna(0.0)

        # Build the optimiser from config
        optimizer = CVaROptimizer(
            alpha=self.cvar_params.get("alpha", 0.95),
            lookback=self.cvar_params.get("lookback", 60),
            min_weight=self.cvar_params.get("min_weight", 0.0),
            max_weight=self.cvar_params.get("max_weight", 0.40),
            rebalance_freq=self.cvar_params.get("rebalance_freq", 20),
            min_history=self.cvar_params.get("min_history", 30),
        )

        weights_df = optimizer.compute_weights(returns_df, df_positions.index)

        # Ensure columns align (add zero-weight for any ticker missing returns)
        for col in df_positions.columns:
            if col not in weights_df.columns:
                weights_df[col] = 1.0 / df_positions.shape[1]

        weights_df = weights_df.reindex(columns=df_positions.columns).fillna(
            1.0 / df_positions.shape[1]
        )

        # Apply weights to positions
        allocated = df_positions * weights_df
        return allocated
