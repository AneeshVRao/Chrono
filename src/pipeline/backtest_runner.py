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
from src.portfolio.portfolio_manager import PortfolioManager
from src.risk.risk_manager import RiskManager
from src.data.fetcher import DataFetcher
from src.portfolio.beta_neutralizer import BetaNeutralizer


class BacktestRunner:
    """Orchestrates backtesting across tickers and strategies."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.logger = get_logger("pipeline.backtest")

        bt_cfg = cfg.backtesting_params
        exec_cfg = cfg.execution_model
        self.engine = BacktestEngine.from_config(bt_cfg, exec_cfg=exec_cfg)
        self.splitter = WalkForwardSplitter.from_config(bt_cfg)

        self.strategies: list[BaseStrategy] = [
            MomentumStrategy(bt_cfg.get("strategies", {}).get("momentum")),
            MeanReversionStrategy(bt_cfg.get("strategies", {}).get("mean_reversion")),
            MLStrategy("LogisticRegression"),
            MLStrategy("RandomForest"),
            MLStrategy("XGBoost"),
            MLStrategy("Ensemble"),
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
        """Run full portfolio backtest on all tickers."""
        tickers = tickers or self.cfg.tickers

        self.logger.info("=" * 70)
        self.logger.info("  PORTFOLIO RUNNER -- Multi-Asset Execution")
        self.logger.info("=" * 70)

        t0 = time.time()
        
        # Fetch SPY for Beta Neutralization
        self.logger.info("Fetching SPY for Beta Neutralization...")
        spy_fetcher = DataFetcher(tickers=["SPY"], start=self.cfg.start_date, end=self.cfg.end_date, interval=self.cfg.interval, output_dir=self.cfg.raw_dir)
        try:
            spy_df = spy_fetcher.fetch_single("SPY")
            spy_returns = spy_df["close"].pct_change().fillna(0) if not spy_df.empty else pd.Series(dtype=float)
        except Exception as e:
            self.logger.warning(f"Failed to fetch SPY: {e}")
            spy_returns = pd.Series(dtype=float)
            
        beta_neutralizer = BetaNeutralizer(spy_returns, window=60) if not spy_returns.empty else None

        # Load all ticker dataframes securely
        dfs: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            try:
                dfs[ticker] = self.load_features(ticker)
            except FileNotFoundError:
                self.logger.warning(f"Features missing for {ticker}, skipping.")
                
        if not dfs:
            self.logger.error("No valid ticker data frames located.")
            return [], []

        # Pre-compute returns for all assets to feed Risk Parity volatility scaling
        returns_dict = {t: df["close"].pct_change().fillna(0) for t, df in dfs.items()}
        
        # Pre-compute betas for all assets if neutralizer is available
        asset_betas_dict = {}
        if beta_neutralizer:
            for t, ret in returns_dict.items():
                asset_betas_dict[t] = beta_neutralizer.compute_asset_beta(ret)
        asset_betas_df = pd.DataFrame(asset_betas_dict) if asset_betas_dict else None
        
        # Initialize engines — portfolio allocation (CVaR by default, configurable)
        portfolio_cfg = self.cfg.get("portfolio", {})
        allocation_type = portfolio_cfg.get("allocation_type", "cvar")
        cvar_params = portfolio_cfg.get("cvar", {})
        portfolio_manager = PortfolioManager(
            allocation_type=allocation_type,
            returns_data=returns_dict,
            cvar_params=cvar_params,
        )
        risk_manager = RiskManager(self.cfg.get("risk_management", {}))

        all_results: list[BacktestResult] = []
        all_reports: list[PerformanceReport] = []

        # Process each strategy across the entire portfolio
        for strategy in self.strategies:
            self.logger.info(f">> Executing {strategy.name} Portfolio...")
            raw_signals = {}
            risk_adj_signals = {}
            
            # Step 1 & 2: Generate raw signals and apply per-asset Risk Rules
            for ticker, df in dfs.items():
                raw = strategy.generate_signals(df)
                returns = returns_dict[ticker]
                risk_adj = risk_manager.apply_rules(df, raw, returns)
                raw_signals[ticker] = raw
                risk_adj_signals[ticker] = risk_adj
                
            # Step 3: Portfolio Allocation Weights
            weights_df = portfolio_manager.allocate(risk_adj_signals)
            
            # Step 4: Backtest executions per weight
            strategy_daily_returns = None
            
            for ticker, df in dfs.items():
                alloc_signal = weights_df[ticker]
                result = self.engine.run(df, alloc_signal, strategy.name, ticker)
                
                # Combine returns globally
                if strategy_daily_returns is None:
                    strategy_daily_returns = result.daily_returns.copy()
                else:
                    strategy_daily_returns += result.daily_returns
                    
            # Compute Master Portfolio Metrics
            if strategy_daily_returns is not None:
                pf_report = self.engine.metrics_calc.compute(
                    strategy_daily_returns, strategy.name, "PORTFOLIO", None
                )
                all_reports.append(pf_report)
                self.logger.info(f"   [{strategy.name} PORTFOLIO] Return: {pf_report.total_return:.2%} | Sharpe: {pf_report.sharpe_ratio:.3f} | MaxDD: {pf_report.max_drawdown:.2%}")
                
                # Beta Neutralization & Alpha Decomposition
                if beta_neutralizer and asset_betas_df is not None:
                    hedge_position = beta_neutralizer.apply_hedge(weights_df, asset_betas_df)
                    
                    # Align SPY returns and apply hedge position (signal at t executed at t+1)
                    spy_returns_aligned = spy_returns.reindex(strategy_daily_returns.index).fillna(0)
                    hedge_returns = hedge_position.shift(1).fillna(0).reindex(strategy_daily_returns.index).fillna(0) * spy_returns_aligned
                    
                    hedged_returns = strategy_daily_returns + hedge_returns
                    
                    hedged_report = self.engine.metrics_calc.compute(
                        hedged_returns, f"{strategy.name} (Beta-Neutral)", "PORTFOLIO", None
                    )
                    all_reports.append(hedged_report)
                    
                    # Log comparison and alpha contribution
                    pf_beta_series = beta_neutralizer.compute_asset_beta(strategy_daily_returns)
                    pf_beta = pf_beta_series.mean()
                    
                    unhedged_decomp = beta_neutralizer.alpha_decomposition(strategy_daily_returns, pf_beta_series)
                    alpha_sum = unhedged_decomp['alpha'].sum()
                    total_sum = unhedged_decomp['total_return'].sum()
                    alpha_contrib_pct = (alpha_sum / total_sum) * 100 if total_sum != 0 else 0
                    
                    self.logger.info(f"   [BETA NEUTRALIZATION] {strategy.name}")
                    self.logger.info(f"      Hedged Return: {hedged_report.total_return:.2%} | Hedged Sharpe: {hedged_report.sharpe_ratio:.3f}")
                    self.logger.info(f"      Validation -> Initial Sharpe: {pf_report.sharpe_ratio:.3f} vs Hedged Sharpe: {hedged_report.sharpe_ratio:.3f}")
                    self.logger.info(f"      Avg Unhedged Pf Beta vs SPY: {pf_beta:.3f} | Alpha Contribution to Return: {alpha_contrib_pct:.1f}%")

        # Benchmark Global Portfolio (Buy & Hold equally weighted)
        self.logger.info(f">> Compiling Global Benchmark (Buy & Hold)...")
        b_raw_signals = {t: pd.Series(1.0, index=df.index) for t, df in dfs.items()}
        b_weights = portfolio_manager.allocate(b_raw_signals)
        bench_daily_returns = None
        for ticker, df in dfs.items():
            alloc_sig = b_weights[ticker]
            b_res = self.engine.run(df, alloc_sig, "Buy & Hold", ticker)
            bench_daily_returns = b_res.daily_returns.copy() if bench_daily_returns is None else bench_daily_returns + b_res.daily_returns
            
        if bench_daily_returns is not None:
            bench_report = self.engine.metrics_calc.compute(
                bench_daily_returns, "Buy & Hold", "PORTFOLIO", None
            )
            all_reports.append(bench_report)
            self.logger.info(f"   [BENCHMARK PORTFOLIO] Return: {bench_report.total_return:.2%} | Sharpe: {bench_report.sharpe_ratio:.3f} | MaxDD: {bench_report.max_drawdown:.2%}")

        elapsed = time.time() - t0
        self.logger.info(f"Backtest complete in {elapsed:.1f}s -- "
                         f"{len(tickers)} tickers, Portfolio Engine Integrated")

        # Log aggregate execution model stats if enabled
        exec_cfg = self.cfg.execution_model
        if exec_cfg.get("use_slippage_model", False):
            self.logger.info("=" * 70)
            self.logger.info("  EXECUTION MODEL SUMMARY")
            self.logger.info("=" * 70)
            self.logger.info(f"  Config: ADV window={exec_cfg.get('adv_window', 20)}, "
                             f"max participation={exec_cfg.get('max_adv_participation', 0.05):.0%}, "
                             f"spread_k={exec_cfg.get('spread_k', 0.15)}, "
                             f"impact_exp={exec_cfg.get('impact_exponent', 2.0)}")

        return all_results, all_reports
