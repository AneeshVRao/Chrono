"""
run_backtest.py — Phase 2 entry point.

Runs baseline strategies (Momentum, Mean Reversion) on Phase 1 feature data.
Produces performance reports, equity curves, and a comparison table.

Usage:
    python run_backtest.py                      # all tickers, all strategies
    python run_backtest.py --ticker AAPL        # single ticker
    python run_backtest.py --skip-fetch         # skip data re-download
    python run_backtest.py --config path.yaml   # custom config
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.config_loader import Config
from src.utils.logger import setup_logging, get_logger
from src.backtesting.engine import BacktestEngine, BacktestResult
from src.backtesting.splitter import WalkForwardSplitter
from src.backtesting.metrics import MetricsCalculator, PerformanceReport
from src.strategies.momentum import MomentumStrategy
from src.strategies.mean_reversion import MeanReversionStrategy

import pandas as pd
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quant ML — Phase 2: Backtesting Engine")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--ticker", type=str, default=None, help="Run single ticker (e.g., AAPL)")
    return parser.parse_args()


def load_feature_data(features_dir: Path, ticker: str) -> pd.DataFrame:
    """Load feature-engineered data for a ticker."""
    path = features_dir / f"{ticker}_features.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {path}\n"
            "Run `python run_pipeline.py` first to generate features."
        )
    df = pd.read_parquet(path, engine="pyarrow")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df


def print_equity_curve_summary(result: BacktestResult) -> None:
    """Print text-based equity curve summary."""
    eq = result.equity_curve
    daily_ret = result.daily_returns

    # Quarterly snapshots
    quarterly = eq.resample("QE").last().dropna()
    print(f"\n  Equity Curve ({result.strategy_name} -- {result.ticker}):")
    print(f"  {'Date':<14s} {'Value':>12s} {'Cum Return':>12s}")
    print(f"  {'-' * 40}")
    for date, val in quarterly.items():
        cum_ret = val / result.equity_curve.iloc[0] - 1
        print(f"  {str(date.date()):<14s} ${val:>11,.0f} {cum_ret:>11.2%}")


def print_walk_forward_results(
    splitter: WalkForwardSplitter,
    df: pd.DataFrame,
    engine: BacktestEngine,
    strategy,
    ticker: str,
) -> list[PerformanceReport]:
    """Run walk-forward validation and print per-fold results."""
    print(f"\n{'=' * 60}")
    print(f"  WALK-FORWARD VALIDATION: {strategy.name} -- {ticker}")
    print(f"{'=' * 60}")

    fold_reports: list[PerformanceReport] = []

    for train_df, test_df, window in splitter.iter_splits(df):
        print(f"\n  {window}")

        # Generate signals on test data (using features computed from past only)
        signals = strategy.generate_signals(test_df)
        result = engine.run(test_df, signals, strategy.name, ticker)
        fold_reports.append(result.report)

        print(f"    Sharpe: {result.report.sharpe_ratio:.3f}  |  "
              f"Return: {result.report.total_return:.2%}  |  "
              f"MaxDD: {result.report.max_drawdown:.2%}  |  "
              f"Trades: {result.report.total_trades}")

    # Summary across folds
    if fold_reports:
        avg_sharpe = np.mean([r.sharpe_ratio for r in fold_reports])
        avg_return = np.mean([r.total_return for r in fold_reports])
        avg_dd = np.mean([r.max_drawdown for r in fold_reports])
        print(f"\n  {'-' * 50}")
        print(f"  Walk-Forward Average:")
        print(f"    Sharpe: {avg_sharpe:.3f}  |  Return: {avg_return:.2%}  |  MaxDD: {avg_dd:.2%}")

    return fold_reports


def main() -> None:
    args = parse_args()
    cfg = Config(args.config)

    setup_logging(level=cfg.log_level, log_dir=cfg.project_root / "logs")
    logger = get_logger("backtest")

    bt_cfg = cfg.backtesting_params

    logger.info("=" * 70)
    logger.info("  QUANT ML TRADING SYSTEM — Phase 2: Backtesting")
    logger.info("=" * 70)

    t0 = time.time()

    # ── Determine tickers ──────────────────────────────────────────────
    tickers = [args.ticker.upper()] if args.ticker else cfg.tickers

    # ── Initialize components ──────────────────────────────────────────
    engine = BacktestEngine.from_config(bt_cfg)
    splitter = WalkForwardSplitter.from_config(bt_cfg)

    strategies = [
        MomentumStrategy(bt_cfg.get("strategies", {}).get("momentum")),
        MeanReversionStrategy(bt_cfg.get("strategies", {}).get("mean_reversion")),
    ]

    # ── Run backtests ──────────────────────────────────────────────────
    all_reports: list[PerformanceReport] = []
    all_results: list[BacktestResult] = []

    for ticker in tickers:
        logger.info(f"\n▶ Processing {ticker}...")

        try:
            df = load_feature_data(cfg.features_dir, ticker)
        except FileNotFoundError as e:
            logger.error(str(e))
            continue

        # ── Full-sample backtest (for equity curve) ────────────────
        for strategy in strategies:
            signals = strategy.generate_signals(df)
            result = engine.run(df, signals, strategy.name, ticker)
            all_results.append(result)
            all_reports.append(result.report)

        # ── Benchmark ──────────────────────────────────────────────
        benchmark = engine.run_benchmark(df, ticker)
        all_results.append(benchmark)
        all_reports.append(benchmark.report)

        # ── Walk-forward validation ────────────────────────────────
        for strategy in strategies:
            print_walk_forward_results(splitter, df, engine, strategy, ticker)

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
    print(f"  Tickers: {len(tickers)}  |  Strategies: {len(strategies) + 1}")
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
