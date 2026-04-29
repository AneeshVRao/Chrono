"""
Realistic Execution Model — ADV caps, bid-ask spread, and market impact slippage.

All computations use only past data (no lookahead bias).
Designed to plug into BacktestEngine without modifying signal generation or
walk-forward validation.

Models:
    1. ADV Participation Cap   — limits order size to max_adv_participation of
                                  rolling 20-day Average Dollar Volume.
    2. Bid-Ask Spread          — spread = k * volatility * price
                                  (asset-specific via realized vol).
    3. Market Impact Slippage  — slippage = spread/2 + impact_cost
                                  impact_cost = (order_size/ADV)^2 * volatility
                                  (non-linear in order size).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExecutionStats:
    """Collects execution model diagnostics for post-run reporting."""
    total_position_changes: int = 0
    orders_capped_by_adv: int = 0
    total_slippage_bps: float = 0.0
    slippage_per_trade: list[float] = field(default_factory=list)

    @property
    def pct_orders_capped(self) -> float:
        if self.total_position_changes == 0:
            return 0.0
        return self.orders_capped_by_adv / self.total_position_changes

    @property
    def avg_slippage_bps(self) -> float:
        if not self.slippage_per_trade:
            return 0.0
        return float(np.mean(self.slippage_per_trade))

    def summary(self) -> str:
        return (
            f"  Execution Model Stats:\n"
            f"    Total position changes:   {self.total_position_changes}\n"
            f"    Orders capped by ADV:     {self.orders_capped_by_adv} "
            f"({self.pct_orders_capped:.1%})\n"
            f"    Avg slippage per trade:   {self.avg_slippage_bps:.2f} bps\n"
            f"    Total slippage cost:      {self.total_slippage_bps:.2f} bps"
        )


class ExecutionModel:
    """
    Computes realistic execution costs per bar, applied only on position changes.

    Parameters (from config dict):
        enabled              : bool   — master toggle (default True)
        adv_window           : int    — rolling window for ADV (default 20)
        max_adv_participation: float  — max fraction of ADV per order (default 0.05)
        spread_k             : float  — spread constant (default 0.15)
        vol_window           : int    — rolling window for realized vol (default 20)
        impact_exponent      : float  — exponent for market impact (default 2.0)
        impact_coefficient   : float  — scalar on (order/ADV)^exp * vol (default 1.0)
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        p = params or {}
        self.enabled = p.get("enabled", True)
        self.adv_window = p.get("adv_window", 20)
        self.max_adv_participation = p.get("max_adv_participation", 0.05)
        self.spread_k = p.get("spread_k", 0.15)
        self.vol_window = p.get("vol_window", 20)
        self.impact_exponent = p.get("impact_exponent", 2.0)
        self.impact_coefficient = p.get("impact_coefficient", 1.0)

    # ------------------------------------------------------------------
    # Pre-computation: build ADV and vol series from price/volume data
    # ------------------------------------------------------------------

    def precompute(
        self, df: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Compute rolling ADV, realized volatility, and estimated spread
        from the DataFrame (must have 'close' and 'volume' columns).

        Returns:
            adv       — rolling 20-day Average Dollar Volume
            vol       — rolling realized volatility (annualized)
            spread    — estimated bid-ask spread in price units
        """
        close = df["close"]
        volume = df["volume"]

        # Rolling ADV (dollar volume) — strictly backward-looking
        dollar_volume = close * volume
        adv = dollar_volume.rolling(self.adv_window, min_periods=1).mean()

        # Realized volatility (annualized) — backward-looking
        log_ret = np.log(close / close.shift(1))
        vol = log_ret.rolling(self.vol_window, min_periods=1).std() * np.sqrt(252)
        vol = vol.fillna(0.20)  # default 20% annualized
        vol = np.maximum(vol, 0.01)  # floor

        # Estimated spread = k * vol * price
        spread = self.spread_k * vol * close

        return adv, vol, spread

    # ------------------------------------------------------------------
    # Core: apply execution costs to a position series
    # ------------------------------------------------------------------

    def apply(
        self,
        df: pd.DataFrame,
        positions: pd.Series,
        initial_capital: float = 100_000.0,
    ) -> tuple[pd.Series, pd.Series, ExecutionStats]:
        """
        Compute execution costs and optionally cap positions.

        This modifies positions (via ADV cap) and returns:
            adj_positions  — positions after ADV capping
            exec_costs     — per-bar execution cost as a fraction of portfolio
                             (only non-zero when position changes)
            stats          — ExecutionStats diagnostics

        No lookahead: ADV and vol are computed from past bars only.
        """
        stats = ExecutionStats()

        if not self.enabled:
            return positions.copy(), pd.Series(0.0, index=positions.index), stats

        # Pre-compute market microstructure series
        adv, vol, spread = self.precompute(df)
        close = df["close"]

        # Detect position changes
        position_changes = positions.diff().fillna(0)
        abs_changes = position_changes.abs()

        # Identify bars where a trade occurs
        trade_mask = abs_changes > 1e-10

        adj_positions = positions.copy()

        # ── ADV Participation Cap (sequential — position dependencies) ──
        order_dollars = abs_changes * initial_capital
        adv_cap_dollars = self.max_adv_participation * adv
        needs_capping = trade_mask & (adv > 0) & (order_dollars > adv_cap_dollars)

        if needs_capping.any():
            cap_indices = np.where(needs_capping.values)[0]
            for i in cap_indices:
                stats.orders_capped_by_adv += 1
                scale_factor = adv_cap_dollars.iloc[i] / order_dollars.iloc[i]
                new_change = position_changes.iloc[i] * scale_factor
                if i > 0:
                    adj_positions.iloc[i] = adj_positions.iloc[i - 1] + new_change
                else:
                    adj_positions.iloc[i] = new_change
                # Update abs_changes for downstream cost calc
                abs_changes.iloc[i] = abs(new_change)

        # ── Vectorized cost computation ─────────────────────────────────
        # Recompute order dollars after potential capping adjustments
        order_dollars_final = abs_changes * initial_capital

        # Half-spread cost (fraction of price)
        half_spread_frac = (spread / 2.0) / (close + 1e-10)

        # Market impact slippage (non-linear in participation rate)
        participation_rate = order_dollars_final / (adv + 1e-10)
        impact_cost_frac = (
            self.impact_coefficient
            * (participation_rate ** self.impact_exponent)
            * vol
        )

        # Total slippage per bar (as fraction of notional)
        total_slippage_frac = half_spread_frac + impact_cost_frac

        # Portfolio-level cost = slippage * |position change|
        exec_costs = (total_slippage_frac * abs_changes).where(trade_mask, 0.0)

        # ── Collect stats ───────────────────────────────────────────────
        stats.total_position_changes = int(trade_mask.sum())
        trade_slippages = (total_slippage_frac[trade_mask] * 10_000).tolist()
        stats.slippage_per_trade = trade_slippages
        stats.total_slippage_bps = sum(trade_slippages)

        return adj_positions, exec_costs, stats
