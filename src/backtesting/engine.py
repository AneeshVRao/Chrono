"""
Backtesting engine — simulates trades over time with transaction costs.

Core principle: at time t, the engine only has access to information up to t.
Signals generated at close of day t are executed at open of day t+1 (next-bar execution).

Designed so ML model predictions can plug directly into the signal interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.backtesting.metrics import MetricsCalculator, PerformanceReport
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Trade:
    """Record of a single trade."""
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    direction: int  # +1 long, -1 short
    shares: float
    pnl: float
    return_pct: float
    cost: float


@dataclass
class BacktestResult:
    """Complete result of a backtest run."""
    strategy_name: str
    ticker: str
    equity_curve: pd.Series        # portfolio value over time
    daily_returns: pd.Series       # daily strategy returns (after costs)
    positions: pd.Series           # position at each timestep (+1, 0, -1)
    signals: pd.Series             # raw signal at each timestep
    trades: list[Trade]            # individual trade records
    report: PerformanceReport      # performance metrics
    benchmark_returns: pd.Series   # buy-and-hold returns


class BacktestEngine:
    """
    Event-driven backtesting engine.

    Signal convention:
        +1 = go long / stay long
         0 = flat / no position
        -1 = go short / stay short

    Any strategy (rule-based or ML) must produce a pd.Series of signals
    aligned to the DataFrame index. The engine handles execution.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        slippage_bps: float = 5.0,
        commission_bps: float = 5.0,
    ) -> None:
        self.initial_capital = initial_capital
        self.cost_rate = (slippage_bps + commission_bps) / 10_000  # total per-trade cost
        self.metrics_calc = MetricsCalculator()

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "BacktestEngine":
        tc = cfg.get("transaction_costs", {})
        return cls(
            initial_capital=cfg.get("initial_capital", 100_000.0),
            slippage_bps=tc.get("slippage_bps", 5.0),
            commission_bps=tc.get("commission_bps", 5.0),
        )

    def run(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        strategy_name: str = "Strategy",
        ticker: str = "UNKNOWN",
    ) -> BacktestResult:
        """
        Run backtest on a DataFrame using pre-computed signals.

        Args:
            df: OHLCV DataFrame with DatetimeIndex (must have 'close' column).
            signals: Series of {+1, 0, -1} aligned to df.index.
                     Signal at time t → execution at t+1 open (simulated as t+1 close).
            strategy_name: identifier for the strategy.
            ticker: ticker symbol.

        Returns:
            BacktestResult with equity curve, returns, trades, and metrics.
        """
        logger.info(f"Running backtest: {strategy_name} on {ticker} ({len(df)} bars)")

        # Align signals to DataFrame
        signals = signals.reindex(df.index).fillna(0).astype(int).clip(-1, 1)

        # -- Position tracking -------------------------------------------------
        # Signal at t -> position change at t+1 (next-bar execution)
        positions = signals.shift(1).fillna(0).astype(int)

        # Detect position changes (trades)
        position_changes = positions.diff().fillna(0)

        # -- Returns calculation ------------------------------------------------
        # Daily asset return: close[t]/close[t-1] - 1
        asset_returns = df["close"].pct_change().fillna(0)

        # Strategy return = position * asset return
        strategy_returns = positions * asset_returns

        # Transaction costs: proportional to abs(position change)
        # Going 0->+1 costs 1x, going +1->-1 costs 2x (close long + open short)
        trade_costs = position_changes.abs() * self.cost_rate
        strategy_returns = strategy_returns - trade_costs

        # -- Equity curve -------------------------------------------------------
        equity = self.initial_capital * (1 + strategy_returns).cumprod()

        # -- Trade extraction ---------------------------------------------------
        trades = self._extract_trades(df, positions, strategy_returns)

        # Count actual position entries for logging
        entries = (position_changes != 0) & (positions != 0)
        n_actual_trades = int(entries.sum())

        # -- Benchmark (buy-and-hold) -------------------------------------------
        benchmark_returns = asset_returns.copy()

        # -- Performance metrics (pass positions for accurate trade counting) ---
        report = self.metrics_calc.compute(
            strategy_returns, strategy_name, ticker, positions=positions
        )

        logger.info(
            f"  Completed: {n_actual_trades} trade entries, "
            f"Sharpe={report.sharpe_ratio:.3f}, "
            f"Return={report.total_return:.2%}"
        )

        return BacktestResult(
            strategy_name=strategy_name,
            ticker=ticker,
            equity_curve=equity,
            daily_returns=strategy_returns,
            positions=positions,
            signals=signals,
            trades=trades,
            report=report,
            benchmark_returns=benchmark_returns,
        )

    def _extract_trades(
        self, df: pd.DataFrame, positions: pd.Series, returns: pd.Series
    ) -> list[Trade]:
        """Extract individual trade records from position series."""
        trades: list[Trade] = []
        prices = df["close"]

        in_trade = False
        entry_date = None
        entry_price = 0.0
        direction = 0

        for i in range(len(positions)):
            pos = positions.iloc[i]
            date = positions.index[i]
            price = prices.iloc[i]

            if not in_trade and pos != 0:
                # Open new trade
                in_trade = True
                entry_date = date
                entry_price = price
                direction = int(pos)

            elif in_trade and (pos == 0 or pos != direction):
                # Close trade
                exit_price = price
                shares = self.initial_capital / (entry_price + 1e-10)
                raw_return = direction * (exit_price / entry_price - 1)
                cost = 2 * self.cost_rate  # entry + exit
                pnl = shares * entry_price * (raw_return - cost)

                trades.append(Trade(
                    entry_date=entry_date,
                    exit_date=date,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    direction=direction,
                    shares=shares,
                    pnl=pnl,
                    return_pct=raw_return - cost,
                    cost=cost * shares * entry_price,
                ))

                # If changing direction (not going flat), open new trade immediately
                if pos != 0:
                    in_trade = True
                    entry_date = date
                    entry_price = price
                    direction = int(pos)
                else:
                    in_trade = False

        # Close any open trade at end
        if in_trade:
            exit_price = prices.iloc[-1]
            shares = self.initial_capital / (entry_price + 1e-10)
            raw_return = direction * (exit_price / entry_price - 1)
            cost = 2 * self.cost_rate
            trades.append(Trade(
                entry_date=entry_date,
                exit_date=positions.index[-1],
                entry_price=entry_price,
                exit_price=exit_price,
                direction=direction,
                shares=shares,
                pnl=shares * entry_price * (raw_return - cost),
                return_pct=raw_return - cost,
                cost=cost * shares * entry_price,
            ))

        return trades

    def run_benchmark(
        self, df: pd.DataFrame, ticker: str = "UNKNOWN"
    ) -> BacktestResult:
        """Run buy-and-hold benchmark for comparison."""
        signals = pd.Series(1, index=df.index, dtype=int)
        return self.run(df, signals, strategy_name="Buy & Hold", ticker=ticker)
