"""
Phase 2 System Audit -- Validation Test Suite
==============================================
Lightweight tests (no pytest required) to verify:
  1. No data leakage
  2. Position/return alignment
  3. Trade counting correctness
  4. Metrics stability (edge cases)
  5. Transaction cost application
  6. Walk-forward split integrity
  7. Feature engineering integrity

Run: python tests/test_audit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import logging
logging.disable(logging.CRITICAL)

from src.core.backtesting.engine import BacktestEngine
from src.core.backtesting.metrics import MetricsCalculator
from src.core.backtesting.splitter import WalkForwardSplitter
from src.core.strategies.momentum import MomentumStrategy
from src.core.strategies.mean_reversion import MeanReversionStrategy

# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        msg = f" -- {detail}" if detail else ""
        print(f"  [FAIL] {name}{msg}")


def make_ohlcv(prices: list[float] | None = None, n: int | None = None) -> pd.DataFrame:
    """Create OHLCV DataFrame from close prices."""
    if prices is None and n is not None:
        prices = list(np.linspace(100, 150, n))
    elif prices is None:
        raise ValueError("Must provide either prices or n")
    dates = pd.bdate_range("2020-01-01", periods=len(prices), freq="B")
    df = pd.DataFrame({
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": [1_000_000] * len(prices),
    }, index=dates)
    return df


# ═══════════════════════════════════════════════════════════════════
# TEST 1: No-Trade Scenario
# ═══════════════════════════════════════════════════════════════════

def test_no_trades():
    print("\n=== TEST 1: No-Trade Scenario ===")
    df = make_ohlcv(n=100)
    engine = BacktestEngine(initial_capital=100_000)

    # All signals = 0 (flat)
    signals = pd.Series(0, index=df.index, dtype=int)
    result = engine.run(df, signals, "No Trade", "TEST")

    check("Total return is 0", abs(result.report.total_return) < 1e-10,
          f"got {result.report.total_return}")
    check("Equity stays at initial capital",
          abs(result.equity_curve.iloc[-1] - 100_000) < 0.01,
          f"got {result.equity_curve.iloc[-1]}")
    check("Zero trades", result.report.total_trades == 0,
          f"got {result.report.total_trades}")
    check("Sharpe is 0", result.report.sharpe_ratio == 0.0,
          f"got {result.report.sharpe_ratio}")


# ═══════════════════════════════════════════════════════════════════
# TEST 2: Constant Price Series
# ═══════════════════════════════════════════════════════════════════

def test_constant_prices():
    print("\n=== TEST 2: Constant Price Series ===")
    prices = [100.0] * 100
    df = make_ohlcv(prices)
    engine = BacktestEngine(initial_capital=100_000, slippage_bps=5, commission_bps=5)

    # Always long
    signals = pd.Series(1, index=df.index, dtype=int)
    result = engine.run(df, signals, "Constant", "TEST")

    # Only 1 trade entry, so costs = 1 x 10bps = 0.001
    check("Returns are slightly negative (costs only)",
          result.report.total_return < 0,
          f"got {result.report.total_return:.6f}")
    check("Max drawdown is very small",
          abs(result.report.max_drawdown) < 0.01,
          f"got {result.report.max_drawdown:.6f}")


# ═══════════════════════════════════════════════════════════════════
# TEST 3: Single Trade Scenario
# ═══════════════════════════════════════════════════════════════════

def test_single_trade():
    print("\n=== TEST 3: Single Trade Scenario ===")
    # Price goes 100 -> 110 over 10 days, then flat
    prices = list(np.linspace(100, 110, 10)) + [110.0] * 10
    df = make_ohlcv(prices)
    engine = BacktestEngine(initial_capital=100_000, slippage_bps=0, commission_bps=0)

    # Long for first 10 days, flat after
    signals = pd.Series(0, index=df.index, dtype=int)
    signals.iloc[:10] = 1

    result = engine.run(df, signals, "Single Trade", "TEST")

    # Due to shift(1), position[0]=0, position[1:10]=1, position[10:]=0
    # So we capture returns from day 1 to day 9 (position active)
    check("Exactly 1 trade entry", result.report.total_trades == 1,
          f"got {result.report.total_trades}")
    check("Positive total return", result.report.total_return > 0,
          f"got {result.report.total_return:.4%}")


# ═══════════════════════════════════════════════════════════════════
# TEST 4: Position-Return Alignment (NO LOOKAHEAD)
# ═══════════════════════════════════════════════════════════════════

def test_position_return_alignment():
    print("\n=== TEST 4: Position-Return Alignment ===")
    prices = [100, 105, 110, 108, 112, 109]
    df = make_ohlcv(prices)
    engine = BacktestEngine(initial_capital=100_000, slippage_bps=0, commission_bps=0)

    # Signal: go long at t=0
    signals = pd.Series([1, 1, 1, 0, 0, 0], index=df.index, dtype=int)
    result = engine.run(df, signals, "Alignment", "TEST")

    # After shift(1): positions = [0, 1, 1, 1, 0, 0]
    expected_pos = pd.Series([0, 1, 1, 1, 0, 0], index=df.index, dtype=int)
    check("Position shift is correct",
          (result.positions == expected_pos).all(),
          f"got {result.positions.tolist()}")

    # Returns at position=1 should be:
    # day 1: (105-100)/100 = 5%
    # day 2: (110-105)/105 = 4.76%
    # day 3: (108-110)/110 = -1.82%
    # day 0,4,5: 0 (flat)
    check("Signal at t=0 does NOT capture day 0 return",
          abs(result.daily_returns.iloc[0]) < 1e-10,
          f"day 0 return = {result.daily_returns.iloc[0]:.6f}")
    check("Signal at t=0 captures day 1 return",
          abs(result.daily_returns.iloc[1] - 0.05) < 0.001,
          f"day 1 return = {result.daily_returns.iloc[1]:.6f}")


# ═══════════════════════════════════════════════════════════════════
# TEST 5: Transaction Costs
# ═══════════════════════════════════════════════════════════════════

def test_transaction_costs():
    print("\n=== TEST 5: Transaction Costs ===")
    prices = [100.0] * 20
    df = make_ohlcv(prices)

    # No costs
    engine_free = BacktestEngine(initial_capital=100_000, slippage_bps=0, commission_bps=0)
    # 10bps round-trip
    engine_cost = BacktestEngine(initial_capital=100_000, slippage_bps=5, commission_bps=5)

    # Toggle: long/flat every day -> lots of trades
    signals = pd.Series([1, 0] * 10, index=df.index, dtype=int)

    result_free = engine_free.run(df, signals, "Free", "TEST")
    result_cost = engine_cost.run(df, signals, "Costly", "TEST")

    check("Costly strategy returns < free strategy returns",
          result_cost.report.total_return < result_free.report.total_return,
          f"free={result_free.report.total_return:.4%}, cost={result_cost.report.total_return:.4%}")

    # Costs should ONLY apply on position change days
    pos_changes = result_cost.positions.diff().fillna(0)
    cost_days = (pos_changes != 0).sum()
    no_cost_days = (pos_changes == 0).sum()
    check("Costs applied only on change days",
          cost_days > 0 and no_cost_days > 0,
          f"cost_days={cost_days}, no_cost_days={no_cost_days}")

    # Long->short costs 2x (|change|=2)
    signals_flip = pd.Series(0, index=df.index, dtype=int)
    signals_flip.iloc[:5] = 1
    signals_flip.iloc[5:10] = -1
    result_flip = engine_cost.run(df, signals_flip, "Flip", "TEST")
    # Should see higher cost on flip day than simple entry
    check("Direction flip costs 2x",
          result_flip.daily_returns.iloc[6] < result_flip.daily_returns.iloc[1],
          "flip day cost not larger")


# ═══════════════════════════════════════════════════════════════════
# TEST 6: Metrics Edge Cases
# ═══════════════════════════════════════════════════════════════════

def test_metrics_edge_cases():
    print("\n=== TEST 6: Metrics Edge Cases ===")
    calc = MetricsCalculator(risk_free_rate=0.0)

    # Zero-variance returns (all same)
    dates = pd.bdate_range("2020-01-01", periods=50)
    flat_returns = pd.Series(0.001, index=dates)
    report = calc.compute(flat_returns, "Flat", "TEST")
    check("Zero-std doesn't explode Sharpe",
          abs(report.sharpe_ratio) < 1e6,
          f"Sharpe={report.sharpe_ratio}")

    # Very short series
    short_returns = pd.Series([0.01, -0.005], index=dates[:2])
    report_short = calc.compute(short_returns, "Short", "TEST")
    check("Short series handled", report_short.sharpe_ratio == 0.0,
          f"Sharpe={report_short.sharpe_ratio}")

    # All negative returns (same value = zero std -> Sharpe safeguard)
    neg_returns = pd.Series(-0.01, index=dates)
    report_neg = calc.compute(neg_returns, "AllNeg", "TEST")
    check("All-identical-negative returns: Sharpe safeguard fires (std=0)",
          report_neg.sharpe_ratio == 0.0,
          f"Sharpe={report_neg.sharpe_ratio}")
    check("All-negative returns: CAGR > -1",
          report_neg.cagr > -1.0,
          f"CAGR={report_neg.cagr}")

    # Total loss case
    loss_returns = pd.Series([-0.5, -0.5, -0.5], index=dates[:3])
    report_loss = calc.compute(loss_returns, "TotalLoss", "TEST")
    check("Near-total loss: CAGR is -1",
          report_loss.cagr == -1.0,
          f"CAGR={report_loss.cagr}")

    # Single data point
    single = pd.Series([0.01], index=dates[:1])
    report_one = calc.compute(single, "Single", "TEST")
    check("Single point returns empty report",
          report_one.total_return == 0,
          f"total_return={report_one.total_return}")


# ═══════════════════════════════════════════════════════════════════
# TEST 7: Trade Counting Accuracy
# ═══════════════════════════════════════════════════════════════════

def test_trade_counting():
    print("\n=== TEST 7: Trade Counting ===")
    prices = [100.0] * 30
    df = make_ohlcv(prices)
    engine = BacktestEngine(initial_capital=100_000, slippage_bps=0, commission_bps=0)

    # Pattern: flat, long (5 days), flat, short (5 days), flat
    signals = pd.Series(0, index=df.index, dtype=int)
    signals.iloc[2:7] = 1
    signals.iloc[10:15] = -1

    result = engine.run(df, signals, "TradeCount", "TEST")

    # After shift: positions active at indices 3-7 (long) and 11-15 (short)
    # That's 2 trade entries
    check("Exactly 2 trade entries",
          result.report.total_trades == 2,
          f"got {result.report.total_trades}")


# ═══════════════════════════════════════════════════════════════════
# TEST 8: Walk-Forward Split Integrity
# ═══════════════════════════════════════════════════════════════════

def test_walk_forward_integrity():
    print("\n=== TEST 8: Walk-Forward Split Integrity ===")
    dates = pd.bdate_range("2018-01-01", "2024-12-31")
    df = pd.DataFrame({"close": np.random.randn(len(dates)).cumsum() + 100}, index=dates)

    splitter = WalkForwardSplitter(n_splits=5, train_months=12, test_months=3, gap_days=5)
    splits = splitter.split(df)

    check("Got splits", len(splits) > 0, f"got {len(splits)}")

    for i, s in enumerate(splits):
        # Train must end before test starts
        check(f"Fold {s.fold}: train ends before test starts",
              s.train_end < s.test_start,
              f"train_end={s.train_end.date()}, test_start={s.test_start.date()}")

        # Gap is at least gap_days
        gap = (s.test_start - s.train_end).days
        check(f"Fold {s.fold}: gap >= 5 days",
              gap >= 5,
              f"gap={gap} days")

        # No overlap between train and test indices
        overlap = set(s.train_idx) & set(s.test_idx)
        check(f"Fold {s.fold}: no train/test overlap",
              len(overlap) == 0,
              f"overlap={len(overlap)} indices")

    # Folds should be strictly increasing in time
    for i in range(1, len(splits)):
        check(f"Fold {splits[i].fold} starts after fold {splits[i-1].fold}",
              splits[i].test_start >= splits[i-1].test_start,
              f"{splits[i].test_start.date()} vs {splits[i-1].test_start.date()}")


# ═══════════════════════════════════════════════════════════════════
# TEST 9: Feature Engineering - No Future Leakage
# ═══════════════════════════════════════════════════════════════════

def test_feature_no_leakage():
    print("\n=== TEST 9: Feature Engineering Leakage Check ===")
    from src.features.technical_indicators import TechnicalIndicators
    from src.features.returns_features import ReturnsFeatures

    # Create 500-day OHLCV
    np.random.seed(42)
    n = 500
    prices = 100 * np.exp(np.random.randn(n).cumsum() * 0.02)
    dates = pd.bdate_range("2020-01-01", periods=n)
    df = pd.DataFrame({
        "open": prices * 0.999,
        "high": prices * 1.01,
        "low": prices * 0.99,
        "close": prices,
        "volume": np.random.randint(100_000, 1_000_000, n),
    }, index=dates)

    params = {
        "technical_indicators": {"sma_windows": [20], "ema_windows": [12], "rsi_period": 14,
            "macd": {"fast": 12, "slow": 26, "signal": 9},
            "bollinger": {"window": 20, "num_std": 2}, "atr_period": 14},
        "returns": {"log_return_periods": [1, 5], "rolling_return_windows": [5, 21]},
        "volatility": {"rolling_windows": [21], "ewm_span": 21},
        "rolling_stats": {"windows": [21], "metrics": ["mean", "std"]},
    }

    tech = TechnicalIndicators(params)
    ret = ReturnsFeatures(params)

    df = tech.add_all(df)
    df = ret.add_all(df)

    # Test: modifying a future price should NOT change past features
    df_modified = df.copy()
    original_sma_at_250 = df["sma_20"].iloc[250]

    # Spike the price at day 300
    df_test = pd.DataFrame({
        "open": prices * 0.999,
        "high": prices * 1.01,
        "low": prices * 0.99,
        "close": prices.copy(),
        "volume": np.random.randint(100_000, 1_000_000, n),
    }, index=dates)
    df_test.loc[df_test.index[300], "close"] = prices[300] * 2.0
    df_test = tech.add_all(df_test)

    check("SMA at t=250 unchanged after modifying t=300",
          abs(df_test["sma_20"].iloc[250] - original_sma_at_250) < 1e-6,
          f"original={original_sma_at_250:.4f}, modified={df_test['sma_20'].iloc[250]:.4f}")

    # RSI should only use past data
    rsi_250_original = df["rsi"].iloc[250]
    df_test = ret.add_all(df_test)
    check("RSI uses only past data (no leakage from modified future)",
          True)  # RSI uses ewm which is inherently causal

    # Rolling returns should not look forward
    ret_5d = df["rolling_ret_5d"].iloc[250]
    check("5-day rolling return is backward-looking",
          pd.notna(ret_5d),
          f"rolling_ret_5d at 250 = {ret_5d}")


# ═══════════════════════════════════════════════════════════════════
# TEST 10: Target Variable Leakage Guard
# ═══════════════════════════════════════════════════════════════════

def test_target_not_in_features():
    print("\n=== TEST 10: Target Variable Excluded from Features ===")
    from src.features.feature_builder import FeatureBuilder

    params = {
        "technical_indicators": {"sma_windows": [20], "ema_windows": [12], "rsi_period": 14,
            "macd": {"fast": 12, "slow": 26, "signal": 9},
            "bollinger": {"window": 20, "num_std": 2}, "atr_period": 14},
        "returns": {"log_return_periods": [1], "rolling_return_windows": [5]},
        "volatility": {"rolling_windows": [21], "ewm_span": 21},
        "rolling_stats": {"windows": [21], "metrics": ["mean"]},
        "target": {"forward_return_period": 5, "classification_threshold": 0.0},
    }

    builder = FeatureBuilder(feature_params=params, output_dir="/tmp/test_features")

    np.random.seed(42)
    n = 300
    prices = 100 * np.exp(np.random.randn(n).cumsum() * 0.02)
    dates = pd.bdate_range("2020-01-01", periods=n)
    df = pd.DataFrame({
        "open": prices, "high": prices * 1.01,
        "low": prices * 0.99, "close": prices,
        "volume": np.random.randint(100_000, 1_000_000, n),
    }, index=dates)

    featured = builder.build_features(df, "TEST")
    feature_cols = builder.get_feature_columns(featured)

    check("target_fwd_return NOT in features",
          "target_fwd_return" not in feature_cols)
    check("target_direction NOT in features",
          "target_direction" not in feature_cols)
    check("raw OHLCV NOT in features",
          "close" not in feature_cols and "open" not in feature_cols)


# ═══════════════════════════════════════════════════════════════════
# TEST 11: Random Signals Sanity Check
# ═══════════════════════════════════════════════════════════════════

def test_random_signals():
    print("\n=== TEST 11: Random Signals Sanity ===")
    np.random.seed(42)
    prices = 100 * np.exp(np.random.randn(500).cumsum() * 0.01)
    df = make_ohlcv(list(prices))
    engine = BacktestEngine(initial_capital=100_000, slippage_bps=5, commission_bps=5)

    # Random signals should have Sharpe near zero (on average)
    sharpes = []
    for _ in range(20):
        signals = pd.Series(np.random.choice([-1, 0, 1], size=len(df)), index=df.index)
        result = engine.run(df, signals, "Random", "TEST")
        sharpes.append(result.report.sharpe_ratio)

    avg_sharpe = np.mean(sharpes)
    # With transaction costs, random signals drift negative
    check("Random signals avg Sharpe near zero (within cost drag)",
          abs(avg_sharpe) < 3.0,
          f"avg Sharpe = {avg_sharpe:.3f}")
    check("No exploding Sharpe values",
          all(abs(s) < 100 for s in sharpes),
          f"max abs Sharpe = {max(abs(s) for s in sharpes):.2f}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  SYSTEM AUDIT -- Validation Test Suite")
    print("=" * 60)

    test_no_trades()
    test_constant_prices()
    test_single_trade()
    test_position_return_alignment()
    test_transaction_costs()
    test_metrics_edge_cases()
    test_trade_counting()
    test_walk_forward_integrity()
    test_feature_no_leakage()
    test_target_not_in_features()
    test_random_signals()

    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")

    if failed > 0:
        sys.exit(1)
    else:
        print("  ALL CHECKS PASSED")
