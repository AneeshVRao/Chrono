"""
Backtest runner orchestrator.
Coordinates: load features -> generate signals -> run engine -> report.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config_loader import Config
from src.utils.logger import get_logger
from src.core.backtesting.engine import BacktestEngine, BacktestResult
from src.core.backtesting.splitter import WalkForwardSplitter
from src.core.backtesting.metrics import MetricsCalculator, PerformanceReport
from src.core.strategies.momentum import MomentumStrategy
from src.core.strategies.mean_reversion import MeanReversionStrategy
from src.core.strategies.ml_strategy import MLStrategy
from src.core.strategies.base import BaseStrategy


class BacktestRunner:
    """Orchestrates backtesting across tickers and strategies."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.logger = get_logger("pipeline.backtest")

        bt_cfg = cfg.backtesting_params
        self.engine = BacktestEngine.from_config(bt_cfg)
        self.splitter = WalkForwardSplitter.from_config(bt_cfg)

        self.strategies: list[BaseStrategy] = [
            MomentumStrategy(bt_cfg.get("strategies", {}).get("momentum")),
            MeanReversionStrategy(bt_cfg.get("strategies", {}).get("mean_reversion")),
            MLStrategy("LogisticRegression"),
            MLStrategy("RandomForest"),
        ]

    def load_features(self, ticker: str) -> pd.DataFrame:
        """Load feature-engineered data for a ticker."""
        path = self.cfg.features_dir / f"{ticker}_features.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Feature file not found: {path}\n"
                "Run `python scripts/run_pipeline.py` first."
            )
        df = pd.read_parquet(path, engine="pyarrow")
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df

    def run_single_ticker(
        self, ticker: str
    ) -> tuple[list[BacktestResult], list[PerformanceReport]]:
        """Run all strategies + benchmark on one ticker."""
        df = self.load_features(ticker)
        results: list[BacktestResult] = []
        reports: list[PerformanceReport] = []

        for strategy in self.strategies:
            signals = strategy.generate_signals(df)
            result = self.engine.run(df, signals, strategy.name, ticker)
            results.append(result)
            reports.append(result.report)

        # Benchmark
        benchmark = self.engine.run_benchmark(df, ticker)
        results.append(benchmark)
        reports.append(benchmark.report)

        return results, reports

    def run_walk_forward(
        self, ticker: str, strategy: BaseStrategy
    ) -> list[PerformanceReport]:
        """Run walk-forward validation for one strategy on one ticker."""
        df = self.load_features(ticker)
        fold_reports: list[PerformanceReport] = []

        for train_df, test_df, window in self.splitter.iter_splits(df):
            signals = strategy.generate_signals(test_df)
            result = self.engine.run(test_df, signals, strategy.name, ticker)
            fold_reports.append(result.report)

        return fold_reports

    def run_all(
        self, tickers: list[str] | None = None
    ) -> tuple[list[BacktestResult], list[PerformanceReport]]:
        """Run full backtest on all tickers."""
        tickers = tickers or self.cfg.tickers

        self.logger.info("=" * 70)
        self.logger.info("  BACKTEST RUNNER -- strategies x tickers")
        self.logger.info("=" * 70)

        t0 = time.time()
        all_results: list[BacktestResult] = []
        all_reports: list[PerformanceReport] = []

        for ticker in tickers:
            self.logger.info(f">> Processing {ticker}...")
            try:
                results, reports = self.run_single_ticker(ticker)
                all_results.extend(results)
                all_reports.extend(reports)
            except FileNotFoundError as e:
                self.logger.error(str(e))
                continue

        elapsed = time.time() - t0
        self.logger.info(f"Backtest complete in {elapsed:.1f}s -- "
                         f"{len(tickers)} tickers, {len(self.strategies) + 1} strategies")
        return all_results, all_reports
