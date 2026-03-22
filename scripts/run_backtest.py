"""
run_backtest.py — Phase 2 entry point.

Usage:
    python scripts/run_backtest.py                      # all tickers, all strategies
    python scripts/run_backtest.py --ticker AAPL        # single ticker
    python scripts/run_backtest.py --config path.yaml   # custom config
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Fix sys.path to allow src imports (since we are in scripts/ dir)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config_loader import Config
from src.utils.logger import setup_logging
from src.pipeline.backtest_runner import BacktestRunner
from src.core.backtesting.metrics import MetricsCalculator

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quant ML — Phase 2: Backtesting Engine")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--ticker", type=str, default=None, help="Run single ticker (e.g., AAPL)")
    return parser.parse_args()


def print_equity_curve_summary(result) -> None:
    """Print text-based equity curve summary."""
    eq = result.equity_curve

    # Quarterly snapshots
    quarterly = eq.resample("QE").last().dropna()
    print(f"\n  Equity Curve ({result.strategy_name} -- {result.ticker}):")
    print(f"  {'Date':<14s} {'Value':>12s} {'Cum Return':>12s}")
    print(f"  {'-' * 40}")
    for date, val in quarterly.items():
        cum_ret = val / result.equity_curve.iloc[0] - 1
        print(f"  {str(date.date()):<14s} ${val:>11,.0f} {cum_ret:>11.2%}")


def print_walk_forward_results(fold_reports, strategy_name, ticker) -> None:
    """Print per-fold walk-forward validation results."""
    print(f"\n{'=' * 60}")
    print(f"  WALK-FORWARD VALIDATION: {strategy_name} -- {ticker}")
    print(f"{'=' * 60}")

    for i, report in enumerate(fold_reports):
        print(f"\n  Fold {i+1} ({report.start_date} -> {report.end_date})")
        print(f"    Sharpe: {report.sharpe_ratio:.3f}  |  "
              f"Return: {report.total_return:.2%}  |  "
              f"MaxDD: {report.max_drawdown:.2%}  |  "
              f"Trades: {report.total_trades}")

    # Summary across folds
    if fold_reports:
        avg_sharpe = np.mean([r.sharpe_ratio for r in fold_reports])
        avg_return = np.mean([r.total_return for r in fold_reports])
        avg_dd = np.mean([r.max_drawdown for r in fold_reports])
        print(f"\n  {'-' * 50}")
        print(f"  Walk-Forward Average:")
        print(f"    Sharpe: {avg_sharpe:.3f}  |  Return: {avg_return:.2%}  |  MaxDD: {avg_dd:.2%}")


def main() -> None:
    args = parse_args()
    cfg = Config(args.config)

    setup_logging(level=cfg.log_level, log_dir=cfg.project_root / "logs")

    runner = BacktestRunner(cfg)
    
    t0 = time.time()
    
    tickers_to_run = [args.ticker.upper()] if args.ticker else cfg.tickers

    all_results, all_reports = runner.run_all(tickers=tickers_to_run)

    # ── Walk-forward validation (keep printing logic isolated) ────
    for ticker in tickers_to_run:
        for strategy in runner.strategies:
            fold_reports = runner.run_walk_forward(ticker, strategy)
            print_walk_forward_results(fold_reports, strategy.name, ticker)

    # ── Summary ────────────────────────────────────────────────────────
    elapsed = time.time() - t0

    print(f"\n\n{'=' * 70}")
    print(f"  BACKTEST RESULTS SUMMARY")
    print(f"{'=' * 70}\n")

    # Comparison table
    calc = MetricsCalculator()
    comparison = calc.compare(all_reports)
    print(comparison.to_string())

    # Equity curve summaries (first ticker only to keep output manageable)
    for result in all_results[:3]:  # first ticker: momentum, mean_rev, benchmark
        print_equity_curve_summary(result)

    print(f"\n{'=' * 70}")
    print(f"  Pipeline completed in {elapsed:.1f}s")
    print(f"  Tickers: {len(tickers_to_run)}  |  Strategies: {len(runner.strategies) + 1}")
    print(f"{'=' * 70}")

    # ── Phase 3 connection point ───────────────────────────────────────
    print(f"\n>> PHASE 3 CONNECTION POINT:")
    print(f"   To plug in ML predictions, create a class that inherits BaseStrategy")
    print(f"   and implements generate_signals(df) -> pd.Series of {{+1, 0, -1}}.")
    print(f"   The backtesting engine accepts any signal source -- rule-based or ML.")
    print(f"   Example:")
    print(f"     class MLStrategy(BaseStrategy):")
    print(f"         def generate_signals(self, df):")
    print(f"             predictions = self.model.predict(df[feature_cols])")
    print(f"             return pd.Series(np.sign(predictions), index=df.index)")

if __name__ == "__main__":
    main()
