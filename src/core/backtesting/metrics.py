"""
Performance metrics for backtesting.
All metrics are computed from return series — never from raw prices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

TRADING_DAYS_PER_YEAR = 252


@dataclass
class PerformanceReport:
    """Container for all backtest performance metrics."""
    strategy_name: str
    ticker: str
    start_date: str
    end_date: str
    total_return: float
    cagr: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration_days: int
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_return: float

    def to_dict(self) -> dict:
        return {
            "Strategy": self.strategy_name,
            "Ticker": self.ticker,
            "Period": f"{self.start_date} -> {self.end_date}",
            "Total Return": f"{self.total_return:.2%}",
            "CAGR": f"{self.cagr:.2%}",
            "Volatility (Ann.)": f"{self.annualized_volatility:.2%}",
            "Sharpe Ratio": f"{self.sharpe_ratio:.3f}",
            "Sortino Ratio": f"{self.sortino_ratio:.3f}",
            "Max Drawdown": f"{self.max_drawdown:.2%}",
            "Max DD Duration": f"{self.max_drawdown_duration_days} days",
            "Calmar Ratio": f"{self.calmar_ratio:.3f}",
            "Win Rate": f"{self.win_rate:.2%}",
            "Profit Factor": f"{self.profit_factor:.3f}",
            "Total Trades": self.total_trades,
            "Avg Trade Return": f"{self.avg_trade_return:.4%}",
        }

    def __repr__(self) -> str:
        lines = [f"\n{'=' * 50}", f"  {self.strategy_name} -- {self.ticker}", f"{'=' * 50}"]
        for k, v in self.to_dict().items():
            lines.append(f"  {k:<22s} {v}")
        lines.append(f"{'=' * 50}")
        return "\n".join(lines)


class MetricsCalculator:
    """Compute risk-adjusted performance metrics from strategy returns."""

    def __init__(self, risk_free_rate: float = 0.04) -> None:
        """risk_free_rate: annualized (e.g., 0.04 = 4%)."""
        self.rf_daily = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1

    def compute(
        self,
        returns: pd.Series,
        strategy_name: str = "Strategy",
        ticker: str = "PORTFOLIO",
        positions: pd.Series | None = None,
    ) -> PerformanceReport:
        """Compute full performance report from daily strategy returns.

        Args:
            returns: daily strategy returns (after costs).
            positions: position series for trade counting. If None,
                       falls back to counting non-zero return days.
        """
        returns = returns.dropna()

        if len(returns) < 2:
            logger.warning("Insufficient returns data for metrics")
            return self._empty_report(strategy_name, ticker)

        # -- Basic returns --
        total_return = (1 + returns).prod() - 1
        n_years = len(returns) / TRADING_DAYS_PER_YEAR

        # CAGR: guard against negative total return (can't raise negative to fractional power)
        if total_return > -1.0:
            cagr = (1 + total_return) ** (1 / max(n_years, 1e-6)) - 1
        else:
            cagr = -1.0  # total loss

        # -- Volatility --
        ann_vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

        # -- Sharpe: guard against zero/near-zero std --
        ret_std = returns.std()
        if ret_std < 1e-8 or len(returns) < 10:
            sharpe = 0.0
        else:
            excess_mean = returns.mean() - self.rf_daily
            sharpe = (excess_mean / ret_std) * np.sqrt(TRADING_DAYS_PER_YEAR)

        # -- Sortino: use downside deviation of excess returns --
        excess = returns - self.rf_daily
        downside_excess = excess[excess < 0]
        if len(downside_excess) > 1:
            downside_std = downside_excess.std()
            if downside_std < 1e-8:
                sortino = 0.0
            else:
                sortino = (excess.mean() / downside_std) * np.sqrt(TRADING_DAYS_PER_YEAR)
        else:
            sortino = 0.0

        # -- Drawdown --
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.cummax()
        drawdowns = cum_returns / rolling_max - 1
        max_dd = drawdowns.min()

        # Max drawdown duration
        dd_duration = self._max_drawdown_duration(drawdowns)

        # -- Calmar --
        calmar = cagr / (abs(max_dd) + 1e-10)

        # -- Trade stats: count actual position transitions, not daily returns --
        if positions is not None:
            position_changes = positions.diff().fillna(0)
            # A trade occurs when position changes (entry or exit)
            # Count round-trip trades: each entry+exit pair = 1 trade
            entries = (position_changes != 0) & (positions != 0)
            total_trades = int(entries.sum())

            # Per-trade PnL: segment returns by position periods
            trade_returns = self._compute_trade_returns(returns, positions)
            if len(trade_returns) > 0:
                winning = [r for r in trade_returns if r > 0]
                losing = [r for r in trade_returns if r < 0]
                win_rate = len(winning) / max(len(trade_returns), 1)
                profit_factor = (
                    sum(winning) / (abs(sum(losing)) + 1e-10)
                    if losing else float("inf")
                )
                avg_trade = np.mean(trade_returns)
            else:
                win_rate = 0.0
                profit_factor = 0.0
                avg_trade = 0.0
        else:
            # Fallback: count non-zero return days (less accurate)
            active_days = returns[returns != 0]
            total_trades = len(active_days)
            winning = active_days[active_days > 0]
            losing = active_days[active_days < 0]
            win_rate = len(winning) / max(total_trades, 1)
            profit_factor = (
                winning.sum() / (abs(losing.sum()) + 1e-10)
                if len(losing) > 0 else float("inf")
            )
            avg_trade = active_days.mean() if total_trades > 0 else 0.0

        return PerformanceReport(
            strategy_name=strategy_name,
            ticker=ticker,
            start_date=str(returns.index[0].date()),
            end_date=str(returns.index[-1].date()),
            total_return=total_return,
            cagr=cagr,
            annualized_volatility=ann_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            max_drawdown_duration_days=dd_duration,
            calmar_ratio=calmar,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            avg_trade_return=avg_trade,
        )

    def _compute_trade_returns(
        self, returns: pd.Series, positions: pd.Series
    ) -> list[float]:
        """Compute per-trade cumulative returns by segmenting position periods."""
        trade_returns: list[float] = []
        in_trade = False
        cum_ret = 0.0

        for i in range(len(positions)):
            pos = positions.iloc[i]
            ret = returns.iloc[i]

            if not in_trade and pos != 0:
                in_trade = True
                cum_ret = ret
            elif in_trade and pos != 0:
                cum_ret = (1 + cum_ret) * (1 + ret) - 1
            elif in_trade and pos == 0:
                trade_returns.append(cum_ret)
                in_trade = False
                cum_ret = 0.0

        # Close any still-open trade
        if in_trade:
            trade_returns.append(cum_ret)

        return trade_returns

    def _max_drawdown_duration(self, drawdowns: pd.Series) -> int:
        """Longest streak where portfolio was below previous peak."""
        is_dd = drawdowns < 0
        if not is_dd.any():
            return 0

        # Group consecutive drawdown periods
        groups = (~is_dd).cumsum()
        dd_groups = groups[is_dd]
        if dd_groups.empty:
            return 0
        return int(dd_groups.value_counts().max())

    def _empty_report(self, name: str, ticker: str) -> PerformanceReport:
        return PerformanceReport(
            strategy_name=name, ticker=ticker,
            start_date="N/A", end_date="N/A",
            total_return=0, cagr=0, annualized_volatility=0,
            sharpe_ratio=0, sortino_ratio=0, max_drawdown=0,
            max_drawdown_duration_days=0, calmar_ratio=0,
            win_rate=0, profit_factor=0, total_trades=0, avg_trade_return=0,
        )

    def compare(self, reports: list[PerformanceReport]) -> pd.DataFrame:
        """Create comparison table across strategies."""
        rows = [r.to_dict() for r in reports]
        return pd.DataFrame(rows).set_index("Strategy")
