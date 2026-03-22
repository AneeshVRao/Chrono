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
import pandas as pd


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

    tickers_to_run = [args.ticker.upper()] if args.ticker else cfg.tickers

    # ── Phase 4, 5 & 6: Portfolio Execution & Sensitivities ──────────
    t0 = time.time()
    
    # 1. Base Portfolio Run (Default costs)
    runner = BacktestRunner(cfg)
    all_results, all_reports = runner.run_all(tickers=tickers_to_run)

    # 2. Extract Ensemble and Benchmark for Comparison
    ensemble_res = next((r for r in all_results if r.strategy_name == "ML_Strategy(Ensemble)"), None)
    bench_res = next((r for r in all_results if r.strategy_name == "Buy & Hold"), None)

    alpha_ann = 0.0
    info_ratio = 0.0

    if ensemble_res and bench_res:
        ens_ret = ensemble_res.daily_returns
        idx = ens_ret.index.intersection(bench_res.daily_returns.index)
        
        diff = ens_ret.loc[idx] - bench_res.daily_returns.loc[idx]
        tracking_error = diff.std() * np.sqrt(252)
        
        alpha_ann = (ensemble_res.report.cagr - bench_res.report.cagr)
        info_ratio = alpha_ann / tracking_error if tracking_error > 1e-8 else 0.0

    # 3. Cost Sensitivity Analysis
    costs = {
        "Low (0.1%)": {"slippage_bps": 5.0, "commission_bps": 5.0},
        "Medium (0.5%)": {"slippage_bps": 25.0, "commission_bps": 25.0},
        "High (1.0%)": {"slippage_bps": 50.0, "commission_bps": 50.0}
    }
    
    sensitivity_reports = []
    print(f"\n============================================================")
    print(f"  PHASE 5: COST SENSITIVITY ANALYSIS (Ensemble Portfolio)")
    print(f"============================================================")
    for cost_name, cost_dict in costs.items():
        cfg._cfg["backtesting"]["transaction_costs"] = cost_dict
        temp_runner = BacktestRunner(cfg)
        
        # Filter down to just Ensemble to save time 
        temp_runner.strategies = [s for s in temp_runner.strategies if s.name == "ML_Strategy(Ensemble)"]
        _, reports = temp_runner.run_all(tickers=tickers_to_run)
        
        if reports:
            ens_rep = reports[0]
            sensitivity_reports.append({
                "Cost Regime": cost_name,
                "CAGR": f"{ens_rep.cagr:.2%}",
                "Sharpe": f"{ens_rep.sharpe_ratio:.3f}",
                "Max DD": f"{ens_rep.max_drawdown:.2%}",
            })
            
    sens_df = pd.DataFrame(sensitivity_reports).set_index("Cost Regime")
    print(sens_df.to_string())

    # ── Summary ────────────────────────────────────────────────────────
    elapsed = time.time() - t0

    print(f"\n\n{'=' * 80}")
    print(f"  PHASE 7: FINAL PORTFOLIO AUDIT REPORT")
    print(f"{'=' * 80}\n")

    # Overall Portfolio Metrics Comparison
    calc = MetricsCalculator()
    comparison = calc.compare(all_reports)
    print(comparison.to_string())

    print(f"\n{'-' * 80}")
    print(f"  PHASE 6: BENCHMARK COMPARISON (Ensemble vs Equal-Weight B&H)")
    print(f"{'-' * 80}")
    print(f"  Annualized Alpha:   {alpha_ann:+.2%}")
    print(f"  Information Ratio:  {info_ratio:+.3f}")

    print(f"\n{'=' * 80}")
    print(f"  Portfolio Pipeline Engine completed in {elapsed:.1f}s")
    print(f"  Multi-Asset Executed on {len(tickers_to_run)} tickers.")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    main()
